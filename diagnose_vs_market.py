"""
Diagnostic: is the model a better or worse forecaster than the Kalshi price?
Scores both on the same in-sample candles (never OOS) with binary log loss per outcome market.
Kalshi price = mid of bid/ask. Nothing is trained or tuned here.

    python diagnose_vs_market.py --set baseline
"""
import argparse
import pickle

import numpy as np
import pandas as pd

import config
from run import kalshi_frame, load, predict_outcome


def logloss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


ap = argparse.ArgumentParser()
ap.add_argument("--set", default="baseline")
ap.add_argument("--kind", default=config.MODEL["kind"])
ap.add_argument("--C", type=float, default=config.MODEL["C"])
ap.add_argument("--event_lag", type=float, default=None, help="override TIME[\"event_lag\"] (minutes)")
a = ap.parse_args()
if a.event_lag is not None:
    config.TIME["event_lag"] = a.event_lag

matches, events, candles, kstart, oos_start = load(config.PATHS["canonical"])
c = kalshi_frame(candles, matches, events, config.PATHS["canonical"])
c = c[(c.date < oos_start) & c.phase.isin(["1H", "2H"])].copy()          # in-sample, in play
with open(f"models/{a.set}_{a.kind}_C{a.C}.pkl", "rb") as fh:
    model = pickle.load(fh)
c["p_model"] = predict_outcome(model, c)
c["p_market"] = (c.bid + c.ask) / 2
c = c[c.p_model.notna() & c.bid.notna() & c.ask.notna() & (c.ask - c.bid <= 0.10)]
c["ll_model"], c["ll_market"] = logloss(c.p_model, c.won), logloss(c.p_market, c.won)
c["bucket"] = pd.cut(c.minute, [-1, 15, 30, 45, 60, 75, 90, 200],
                     labels=["0-15", "16-30", "31-45", "46-60", "61-75", "76-90", "90+"])
out = c.groupby("bucket", observed=True).agg(
    rows=("won", "size"), model=("ll_model", "mean"), market=("ll_market", "mean"))
out["model_minus_market"] = out.model - out.market
out.loc["ALL"] = [len(c), c.ll_model.mean(), c.ll_market.mean(), c.ll_model.mean() - c.ll_market.mean()]
print("\nbinary log loss per outcome market, in-sample in-play candles (lower = better)")
print("model_minus_market > 0  ->  Kalshi is the better forecaster\n")
print(out.round(4).to_string())
lg = c.groupby("league").agg(model=("ll_model", "mean"), market=("ll_market", "mean"))
lg["model_minus_market"] = lg.model - lg.market
print("\nby league:\n", lg.round(4).to_string())
