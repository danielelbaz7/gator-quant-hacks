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
    kickoff_from_close=115,  # fallback only: kickoff ~= market close_time minus this many minutes
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
    "full": ["goal_diff", "gd_x_elapsed", "gd_over_sqrt_left", "level_x_elapsed",
             "elapsed", "total_goals", "red_diff", "red_diff_x_left",
             "pre_logit_ha", "pre_logit_x_left", "pre_logit_draw",
             "value_logratio", "value_x_left", "h2h_gd", "xg_diff", "xg_diff_x_left"],
}

MODEL = dict(kind="logit", C=1.0)   # kind: "logit" (multinomial logistic) or "ols" (linear probability)

BACKTEST = dict(
    threshold=0.04,      # required edge, in probability points, AFTER fees
    max_spread=0.06,     # skip quotes wider than this (dollars on a $1 contract)
    min_minute=0,        # earliest decision: negative = clock minutes before kickoff
    max_minute=85,       # latest decision, in match minutes
    skip_halftime=True,
    fee_rate=0.07,       # Kalshi taker fee ~ rate * P * (1-P) per contract. VERIFY on Kalshi's fee schedule.
    slippage=0.0,        # extra cost per contract in dollars
    cost_mult=1.0,       # set 2.0 for the "double the costs" check (done automatically in reports)
    exec_next_bar=True,  # signal on bar t, fill at bar t+1 quotes
    one_per="ticker",    # "ticker" = max one trade per outcome market, "match" = one per match
)

GRID = dict(
    feature_set=["score_only", "baseline", "baseline_value"],
    kind=["logit"],
    C=[0.1, 1.0],
    threshold=[0.03, 0.05, 0.08],
)
