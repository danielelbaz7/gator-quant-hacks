"""
Feature registry. Every feature is a function of a "snapshot" DataFrame with
these canonical columns (missing ones are treated as NaN, and rows with NaN in
a selected feature are dropped from training / never traded):

  minute                      match minute (0 = kickoff / pre-match)
  home_goals, away_goals      goals known at that minute
  home_reds,  away_reds       red cards known at that minute
  home_xg,    away_xg         cumulative xG known at that minute (optional)
  p_home, p_draw, p_away      pre-match bookmaker probabilities, vig removed
  home_value, away_value      starting XI market value in EUR (optional)
  h2h_gd                      avg goal diff (home team view) in last meetings (optional)

Add a feature: write a function, decorate it with @feature("name"), then use
"name" in a FEATURE_SETS entry in config.py.
"""
import numpy as np
import pandas as pd

FEATURES = {}


def feature(name):
    def deco(f):
        FEATURES[name] = f
        return f
    return deco


def col(df, name):
    if name in df:
        return df[name].astype(float)
    return pd.Series(np.nan, index=df.index)


def _elapsed(df):
    return col(df, "minute").clip(0, 95) / 90


def _left(df):
    return (1 - _elapsed(df)).clip(lower=0)


def _gd(df):
    return col(df, "home_goals") - col(df, "away_goals")


@feature("elapsed")
def f_elapsed(df):
    return _elapsed(df)


@feature("time_left")
def f_time_left(df):
    return _left(df)


@feature("goal_diff")
def f_goal_diff(df):
    return _gd(df)


@feature("gd_x_elapsed")
def f_gd_x_elapsed(df):
    """A lead is worth more the later it is."""
    return _gd(df) * _elapsed(df)


@feature("gd_over_sqrt_left")
def f_gd_over_sqrt_left(df):
    """Lead relative to remaining goal 'noise' (variance of remaining goals ~ time left)."""
    return _gd(df) / np.sqrt(_left(df) + 1 / 90)


@feature("level_x_elapsed")
def f_level_x_elapsed(df):
    """Level score late in the game -> draw more likely."""
    return (_gd(df) == 0).astype(float) * _elapsed(df)


@feature("level_over_sqrt_left")
def f_level_over_sqrt_left(df):
    """Level score locks in sharply near the end (draw signal grows like 1/sqrt(time left))."""
    return (_gd(df) == 0).astype(float) / np.sqrt(_left(df) + 1 / 90)


@feature("level_over_left")
def f_level_over_left(df):
    """Even sharper end-of-match lock-in for level scores (1 / time left)."""
    return (_gd(df) == 0).astype(float) / (_left(df) + 1 / 90)


@feature("abs_gd")
def f_abs_gd(df):
    """Size of the lead regardless of side: draws depend on |gd|, which signed features can't express."""
    return _gd(df).abs()


@feature("abs_gd_over_sqrt_left")
def f_abs_gd_over_sqrt_left(df):
    return _gd(df).abs() / np.sqrt(_left(df) + 1 / 90)


@feature("late_tilt_x_elapsed")
def f_late_tilt_x_elapsed(df):
    """Home minus away tendency for late goals (share of goals scored + conceded-by-opponent after
    75', previous 38 games), mattering more as the match gets late."""
    home = col(df, "home_late_for") + col(df, "away_late_against")
    away = col(df, "away_late_for") + col(df, "home_late_against")
    return (home - away) * _elapsed(df)


@feature("total_goals")
def f_total_goals(df):
    return col(df, "home_goals") + col(df, "away_goals")


@feature("red_diff")
def f_red_diff(df):
    """Positive = away team has more red cards (good for home)."""
    return col(df, "away_reds") - col(df, "home_reds")


@feature("red_diff_x_left")
def f_red_diff_x_left(df):
    return f_red_diff(df) * _left(df)


@feature("xg_diff")
def f_xg_diff(df):
    return col(df, "home_xg") - col(df, "away_xg")


@feature("xg_diff_x_left")
def f_xg_diff_x_left(df):
    return f_xg_diff(df) * _left(df)


@feature("xg_recent_diff")
def f_xg_recent_diff(df):
    """Home minus away xG in the last RECENT_XG_WINDOW match minutes (current pressure)."""
    return col(df, "home_xg_recent") - col(df, "away_xg_recent")


@feature("xg_recent_x_left")
def f_xg_recent_x_left(df):
    return f_xg_recent_diff(df) * _left(df)


@feature("comeback_pressure")
def f_comeback_pressure(df):
    """Recent xG of the trailing team minus the leader's (0 when level): > 0 = losing side is pushing."""
    return -np.sign(_gd(df)) * f_xg_recent_diff(df)


@feature("pre_logit_ha")
def f_pre_logit_ha(df):
    """Pre-match strength gap: log(P(home) / P(away))."""
    return np.log(col(df, "p_home") / col(df, "p_away"))


@feature("pre_logit_x_left")
def f_pre_logit_x_left(df):
    """Pre-match strength matters less as the clock runs down."""
    return f_pre_logit_ha(df) * _left(df)


@feature("pre_logit_draw")
def f_pre_logit_draw(df):
    p = col(df, "p_draw")
    return np.log(p / (1 - p))


# Expected total goals from the Over/Under 2.5 market: solve P(Poisson(lam) >= 3) = p_over25.
_LAM = np.linspace(0.3, 7.0, 2000)
_POVER = 1 - np.exp(-_LAM) * (1 + _LAM + _LAM ** 2 / 2)


def _lambda(df):
    p = col(df, "p_over25")
    return pd.Series(np.interp(p, _POVER, _LAM), index=df.index).where(p.notna())


def _egl(df):
    """Expected goals still to come = pre-match expected total x fraction of match left."""
    return _lambda(df) * _left(df)


@feature("pre_total_goals")
def f_pre_total_goals(df):
    return _lambda(df)


@feature("exp_goals_left")
def f_exp_goals_left(df):
    return _egl(df)


@feature("gd_over_sqrt_egl")
def f_gd_over_sqrt_egl(df):
    """Lead relative to the goal 'noise' still expected (Poisson sd ~ sqrt(expected goals left))."""
    return _gd(df) / np.sqrt(_egl(df) + 0.05)


@feature("abs_gd_over_sqrt_egl")
def f_abs_gd_over_sqrt_egl(df):
    return _gd(df).abs() / np.sqrt(_egl(df) + 0.05)


@feature("level_x_egl")
def f_level_x_egl(df):
    """Level score with many goals still expected -> draw less safe."""
    return (_gd(df) == 0).astype(float) * _egl(df)


# Per-team expected goals: split the O/U total so that independent Poisson(lam_h), Poisson(lam_a)
# reproduce the 1X2 market's P(home) - P(away). Solved once on a grid, then interpolated per row.
def _team_lambda_table():
    from scipy.stats import poisson
    T = np.linspace(1.0, 6.0, 101)
    S = np.linspace(-4.0, 4.0, 401)
    g = np.arange(15)
    diff = np.full((len(T), len(S)), np.nan)
    for i, t in enumerate(T):
        lh, la = np.clip((t + S) / 2, 1e-3, None), np.clip((t - S) / 2, 1e-3, None)
        ph, pa = poisson.pmf(g[None, :], lh[:, None]), poisson.pmf(g[None, :], la[:, None])
        joint = ph[:, :, None] * pa[:, None, :]                     # [s, home goals, away goals]
        hw = np.tril(np.ones((15, 15)), -1).astype(bool)            # home goals > away goals
        diff[i] = joint[:, hw].sum(1) - joint[:, hw.T].sum(1)
    return T, S, diff


_TL = None


def _team_lambdas(df):
    global _TL
    if _TL is None:
        _TL = _team_lambda_table()
    T, S, diff = _TL
    tot = _lambda(df).to_numpy()
    target = (col(df, "p_home") - col(df, "p_away")).to_numpy()
    ti = np.clip(np.round((np.nan_to_num(tot, nan=2.7) - T[0]) / (T[1] - T[0])).astype(int), 0, len(T) - 1)
    sup = np.full(len(df), np.nan)
    for k in np.unique(ti):                                         # P(H)-P(A) is increasing in supremacy
        r = ti == k
        sup[r] = np.interp(target[r], diff[k], S)
    lh, la = (tot + sup) / 2, (tot - sup) / 2
    return pd.Series(np.clip(lh, 0.05, None), index=df.index), pd.Series(np.clip(la, 0.05, None), index=df.index)


def _poisson_inplay(df):
    """P(home win), P(draw), P(away win) from the current score and each team's expected goals left."""
    from scipy.stats import skellam
    lh, la = _team_lambdas(df)
    left = _left(df)
    mh, ma = (lh * left + 1e-6).to_numpy(), (la * left + 1e-6).to_numpy()
    gd = _gd(df).to_numpy()
    p_draw = skellam.pmf(-gd, mh, ma)                    # home_left - away_left == -gd
    p_away = skellam.cdf(-gd - 1, mh, ma)                # home_left - away_left <  -gd
    p_home = 1 - p_draw - p_away
    clip = lambda x: np.clip(x, 1e-6, 1 - 1e-6)
    return clip(p_home), clip(p_draw), clip(p_away)


@feature("pois_logit_ha")
def f_pois_logit_ha(df):
    """Structural Poisson in-play model: log(P(home win) / P(away win))."""
    ph, pd_, pa = _poisson_inplay(df)
    return pd.Series(np.log(ph / pa), index=df.index).where(col(df, "p_over25").notna())


@feature("pois_logit_draw")
def f_pois_logit_draw(df):
    ph, pd_, pa = _poisson_inplay(df)
    return pd.Series(np.log(pd_ / (1 - pd_)), index=df.index).where(col(df, "p_over25").notna())


@feature("trailer_egl")
def f_trailer_egl(df):
    """Expected goals left for the team that is behind (0 when level): direct comeback potential."""
    lh, la = _team_lambdas(df)
    gd = _gd(df)
    return (np.where(gd < 0, lh, np.where(gd > 0, la, 0.0)) * _left(df)).where(col(df, "p_over25").notna())


@feature("value_logratio")
def f_value_logratio(df):
    return np.log(col(df, "home_value") / col(df, "away_value"))


@feature("value_x_left")
def f_value_x_left(df):
    return f_value_logratio(df) * _left(df)


@feature("h2h_gd")
def f_h2h_gd(df):
    return col(df, "h2h_gd")


def build_X(df, names):
    missing = [n for n in names if n not in FEATURES]
    if missing:
        raise KeyError(f"unknown features {missing}; available: {sorted(FEATURES)}")
    X = pd.DataFrame({n: FEATURES[n](df) for n in names}, index=df.index)
    return X.replace([np.inf, -np.inf], np.nan)
