"""
Walk-forward report (in-sample only): one model per month, trained on all matches dated before
that month (football data only). For each month: training size, the model's log loss vs the
Kalshi mid on that month's in-play candles, and the trades / P&L from the walk-forward backtest
(results/trades_IS_<set>.csv written by `python run.py backtest --retrain monthly`).

    python diagnose_walkforward.py
"""
import argparse

import numpy as np
import pandas as pd

import config
from data import training_snapshots
from model import WinProbModel
from run import kalshi_frame, load, predict_outcome

ap = argparse.ArgumentParser()
ap.add_argument("--set", default=config.MODEL["feature_set"])
a = ap.parse_args()

matches, events, candles, kstart, oos_start = load(config.PATHS["canonical"])
c = kalshi_frame(candles, matches, events, config.PATHS["canonical"])
c = c[(c.date < oos_start)].copy()
feats = config.FEATURE_SETS[a.set]
snaps = training_snapshots(matches, events, oos_start, config.TRAIN_MINUTES)
c["month"] = pd.to_datetime(c.date).dt.to_period("M").dt.to_timestamp()
ll = lambda p, y: -(y * np.log(np.clip(p, 1e-4, 1 - 1e-4)) + (1 - y) * np.log(np.clip(1 - p, 1e-4, 1 - 1e-4)))

rows = []
for ms in sorted(c.month.unique()):
    tr = snaps[snaps.date < ms]
    model = WinProbModel(feats, config.MODEL["kind"], config.MODEL["C"]).fit(tr)
    x = c[(c.month == ms) & c.phase.isin(["1H", "2H"]) & c.bid.notna() & c.ask.notna() & (c.ask - c.bid <= 0.10)]
    p = predict_outcome(model, x)
    ok = p.notna()
    rows.append(dict(month=f"{ms:%Y-%m}", trained_on_matches=tr.match_id.nunique(),
                     train_until=str((ms - pd.Timedelta(days=1)).date()), games=x.match_id.nunique(),
                     model_logloss=ll(p[ok], x.won[ok]).mean(),
                     kalshi_logloss=ll(((x.bid + x.ask) / 2)[ok], x.won[ok]).mean()))
r = pd.DataFrame(rows).set_index("month")
r["gap"] = r.model_logloss - r.kalshi_logloss

try:
    t = pd.read_csv(f"{config.PATHS['results']}/trades_IS_{a.set}.csv")
    t["month"] = pd.to_datetime(t.t, utc=True).dt.tz_convert(None).dt.to_period("M").astype(str)
    g = t.groupby("month").agg(trades=("pnl", "size"), pnl=("pnl", "sum"), hit=("pnl", lambda s: (s > 0).mean()))
    r = r.join(g).fillna({"trades": 0, "pnl": 0})
    r["cum_pnl"] = r.pnl.cumsum()
except FileNotFoundError:
    print("(no trades file; run `python run.py backtest --retrain monthly` first)")

print(f"\nWALK-FORWARD, {a.set}: one model per month (gap > 0 = Kalshi forecasts better)\n")
print(r.round(4).to_string())
