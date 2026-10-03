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
