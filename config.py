"""
Everything you tune lives here. Edit FEATURE_SETS to try new combinations,
BACKTEST for trading rules, GRID for sweeps. Every backtest run is logged to
results/trials.csv so you can report how many variants you tried.
"""

PATHS = dict(
    canonical="data/canonical",                    # output of `python run.py build`
    tm_duckdb="transfermarkt-datasets.duckdb",
    fd_games="football_data_games.parquet",        # soccerdata MatchHistory output
    kalshi_markets="big5_markets.parquet",
    kalshi_candles="big5_candles.parquet",
    xg_shots="xg_shots.parquet",                   # optional, see data.load_xg_shots
    aliases="team_aliases.csv",                    # optional manual name fixes
    results="results",
)

# Mapping clock time -> match minute. Check these against a few real matches.
TIME = dict(
    first_half_clock=47,     # clock minutes from kickoff to end of 1st half (45 + ~2 stoppage)
    halftime_break=15,       # clock minutes of halftime
    max_minute=97,           # match minute after which we call the match over
    event_lag=2,             # minutes after an event before the model is allowed to "know" it
    kickoff_from_close=122,  # fallback only: kickoff ~= market close_time minus this many minutes
)

SPLIT = dict(
    val_start="2024-07-01",  # training matches before this, validation from here to first Kalshi match
    oos_frac=0.20,           # most recent 20% of Kalshi matches are out-of-sample (track rule)
    tm_start_season=2015,
)

# "closing" odds are only known at kickoff, so they can't be used for pre-match trades.
# The backtest blocks pre-kickoff trades automatically when ODDS == "closing".
ODDS = "closing"             # or "opening"

TRAIN_MINUTES = list(range(0, 91, 5))   # snapshot minutes per training match
RECENT_XG_WINDOW = 15                    # match minutes for "recent pressure" xG features

# --- feature combinations: names come from features.py ---
FEATURE_SETS = {
    "score_only": ["goal_diff", "gd_x_elapsed", "level_x_elapsed", "elapsed"],
    "baseline": ["goal_diff", "gd_x_elapsed", "gd_over_sqrt_left", "level_x_elapsed",
                 "elapsed", "red_diff", "red_diff_x_left",
                 "pre_logit_ha", "pre_logit_x_left", "pre_logit_draw"],
    "baseline_value": ["goal_diff", "gd_x_elapsed", "gd_over_sqrt_left", "level_x_elapsed",
                       "elapsed", "red_diff", "red_diff_x_left",
                       "pre_logit_ha", "pre_logit_x_left", "pre_logit_draw",
                       "value_logratio", "value_x_left"],
    "baseline_xg": ["goal_diff", "gd_x_elapsed", "gd_over_sqrt_left", "level_x_elapsed",
                    "elapsed", "red_diff", "red_diff_x_left",
                    "pre_logit_ha", "pre_logit_x_left", "pre_logit_draw",
                    "xg_diff", "xg_diff_x_left"],
    # baseline_xg + |gd| / level-score shape: fixes late level-game draw underprediction
    # (validation logloss 0.7532 -> 0.7507, paired t = -7.4). late_tilt_x_elapsed tested: no gain (t = -0.4).
    # yellow_diff + yellow_diff_x_left tested on top: slightly worse (0.7507 -> 0.7512, t = +1.7), not used.
    # recent pressure (xg_recent_diff, xg_recent_x_left, comeback_pressure; last 15 min) tested on top:
    # no gain (0.7507 -> 0.7507, t = -0.6); the pressure effect is already captured by cumulative xG + odds.
    "xg_draw": ["goal_diff", "gd_x_elapsed", "gd_over_sqrt_left", "level_x_elapsed",
                "elapsed", "red_diff", "red_diff_x_left",
                "pre_logit_ha", "pre_logit_x_left", "pre_logit_draw",
                "xg_diff", "xg_diff_x_left",
                "level_over_sqrt_left", "abs_gd", "abs_gd_over_sqrt_left"],
    # xg_draw + expected goals from the Over/Under 2.5 market (Poisson lambda): leads judged against
    # goals still expected. Validation logloss 0.7507 -> 0.7493 (paired t = -2.8).
    "xg_draw_ou": ["goal_diff", "gd_x_elapsed", "gd_over_sqrt_left", "level_x_elapsed",
                   "elapsed", "red_diff", "red_diff_x_left",
                   "pre_logit_ha", "pre_logit_x_left", "pre_logit_draw",
                   "xg_diff", "xg_diff_x_left",
                   "level_over_sqrt_left", "abs_gd", "abs_gd_over_sqrt_left",
                   "pre_total_goals", "exp_goals_left", "gd_over_sqrt_egl", "abs_gd_over_sqrt_egl", "level_x_egl"],
    # xg_draw_ou + per-team expected goals (O/U total split by the 1X2 odds) fed through a Poisson
    # in-play model: live P(H)/P(D)/P(A) from current score + each team's goals still expected.
    # Validation logloss 0.7493 -> 0.7487 (paired t = -2.1).
    "xg_draw_pois": ["goal_diff", "gd_x_elapsed", "gd_over_sqrt_left", "level_x_elapsed",
                     "elapsed", "red_diff", "red_diff_x_left",
                     "pre_logit_ha", "pre_logit_x_left", "pre_logit_draw",
                     "xg_diff", "xg_diff_x_left",
                     "level_over_sqrt_left", "abs_gd", "abs_gd_over_sqrt_left",
                     "pre_total_goals", "exp_goals_left", "gd_over_sqrt_egl", "abs_gd_over_sqrt_egl", "level_x_egl",
                     "pois_logit_ha", "pois_logit_draw", "trailer_egl"],
    "full": ["goal_diff", "gd_x_elapsed", "gd_over_sqrt_left", "level_x_elapsed",
             "elapsed", "total_goals", "red_diff", "red_diff_x_left",
             "pre_logit_ha", "pre_logit_x_left", "pre_logit_draw",
             "value_logratio", "value_x_left", "h2h_gd", "xg_diff", "xg_diff_x_left"],
}

# default model: xg_draw_ou (chosen over xg_draw_pois: simpler, better on Kalshi-period games). kind: "logit" or "ols"
# retrain: "none" = train once on matches before the first Kalshi match; "monthly" = walk-forward,
# retrain each month on all matches before that month (football data only)
MODEL = dict(feature_set="xg_draw_ou", kind="logit", C=1.0, retrain="monthly")

BACKTEST = dict(
    threshold=0.08,      # required edge, in probability points, AFTER fees (chosen in-sample, trials 4-12)
    max_spread=0.06,     # skip quotes wider than this (dollars on a $1 contract)
    min_minute=0,        # earliest decision: negative = clock minutes before kickoff
    max_minute=85,       # latest decision, in match minutes
    skip_halftime=True,
    fee_rate=0.07,       # Kalshi taker fee ~ rate * P * (1-P) per contract. VERIFY on Kalshi's fee schedule.
    slippage=0.0,        # extra cost per contract in dollars
    cost_mult=1.0,       # set 2.0 for the "double the costs" check (done automatically in reports)
    exec_next_bar=True,  # signal on bar t, fill at bar t+1 quotes
    one_per="ticker",    # "ticker" = max one trade per outcome market, "match" = one per match
    cooldown=3,          # minutes to stand aside after a sharp price jump in the match (0 = off)
    jump=0.07,           # mid move in one bar (dollars) that counts as a jump; goals move ~0.15-0.20
)

# keys that are also BACKTEST keys (threshold, cooldown, ...) override the trading rule per combo
GRID = dict(
    feature_set=["xg_draw_ou"],
    kind=["logit"],
    C=[1.0],
    threshold=[0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15],
    cooldown=[3],
)
