"""
Diagnostic (no model): is there any simple edge in Kalshi prices themselves? In-sample matches only.

1. Calibration: does a contract priced at X win X% of the time? (favourite-longshot bias check)
2. Naive strategies at fixed match minutes: buy YES on the favourite / the draw / the longshot
   at the ask, hold to settlement, pay the Kalshi fee. P&L per contract.

    python diagnose_market.py
"""
import numpy as np
import pandas as pd

import config
from backtest import kalshi_fee
from run import kalshi_frame, load

matches, events, candles, kstart, oos_start = load(config.PATHS["canonical"])
c = kalshi_frame(candles, matches, events, config.PATHS["canonical"])
c = c[(c.date < oos_start) & c.bid.notna() & c.ask.notna() & (c.ask - c.bid <= 0.10)].copy()
c["mid"] = (c.bid + c.ask) / 2

# one quote per market at each decision point: first quoted bar at/after that match minute
points = {"kickoff": 0, "15'": 15, "30'": 30, "2H start": 46, "60'": 60, "75'": 75, "85'": 85}
snap = []
for name, mm in points.items():
    x = c[(c.phase.isin(["1H", "2H"])) & (c.minute >= mm) & (c.minute < mm + 5)]
    x = x.sort_values("t").groupby("ticker").head(1)
    snap.append(x.assign(point=name))
snap = pd.concat(snap)
# keep match-points where all three outcomes are quoted
full = snap.groupby(["match_id", "point"]).outcome.transform("nunique") == 3
snap = snap[full]

# 1. calibration
bins = [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, 1]
snap["bin"] = pd.cut(snap.mid, bins)
cal = snap.groupby("bin", observed=True).agg(n=("won", "size"), price=("mid", "mean"), won=("won", "mean"))
cal["won_minus_price"] = cal.won - cal.price
se = np.sqrt(cal.won * (1 - cal.won) / cal.n)
cal["t"] = cal.won_minus_price / se.replace(0, np.nan)
print("\n1) CALIBRATION of Kalshi mid prices (in-sample, all decision points pooled)")
print("   won_minus_price < 0 -> contracts at that price win less often than the price implies (overpriced)")
print(cal.round(3).to_string())


# 2. naive strategies
def strat(pick):
    rows = []
    for (mid_, pt), g in snap.groupby(["match_id", "point"]):
        r = g.loc[pick(g)]
        fee = kalshi_fee(r.ask, config.BACKTEST["fee_rate"])
        rows.append(dict(point=pt, pnl=r.won - r.ask - fee, price=r.ask))
    return pd.DataFrame(rows)


def summary(df):
    g = df.groupby("point").pnl
    out = pd.DataFrame({"trades": g.size(), "avg_price": df.groupby("point").price.mean(),
                        "pnl_per_contract": g.mean(), "t": g.mean() / g.std() * np.sqrt(g.size()),
                        "total": g.sum()}).reindex(list(points))
    out.loc["ALL"] = [len(df), df.price.mean(), df.pnl.mean(),
                      df.pnl.mean() / df.pnl.std() * np.sqrt(len(df)), df.pnl.sum()]
    return out.round(3)


for name, pick in [("FAVOURITE (highest price)", lambda g: g.mid.idxmax()),
                   ("DRAW", lambda g: g.index[g.outcome == "draw"][0]),
                   ("LONGSHOT (lowest price)", lambda g: g.mid.idxmin())]:
    print(f"\n2) Buy YES on the {name} at the ask, hold to the end, after fees ({config.BACKTEST['fee_rate']} * P * (1-P))")
    print(summary(strat(pick)).to_string())
