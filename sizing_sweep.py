"""
Position-sizing sweep on a fixed set of IN-SAMPLE signals (the trades of a flat backtest run).
Which trades fire does not depend on sizing, so sizing rules are simulated on that trade list:

  trust     fraction of Kelly. Equivalent to shrinking the model's probability toward the price:
            p_used = price + trust * (p_model - price)  ->  stake = trust * full-Kelly stake
  pos_cap   max stake on one trade as a fraction of current equity (e.g. 0.25 = a quarter of the portfolio)
  dd_stop   circuit breaker: stop trading for good once equity falls this far below its peak
  sized at the FILL price (next bar); if the model no longer sees an edge there, the order is cancelled
  compounding: stakes are a fraction of current realised equity; positions settle at the match end

Every combination is appended to results/trials.csv (sizing variants of the given signal set).

    python run.py backtest            # writes results/trades_IS_<set>.csv (signals)
    python sizing_sweep.py
"""
import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd

import config
from backtest import kalshi_fee

ap = argparse.ArgumentParser()
ap.add_argument("--set", default=config.MODEL["feature_set"])
ap.add_argument("--bankroll", type=float, default=10_000)
ap.add_argument("--sample", choices=["IS", "OOS"], default="IS")
ap.add_argument("--trust", type=float, default=None, help="single rule instead of the sweep")
ap.add_argument("--pos_cap", type=float, default=1.0)
a = ap.parse_args()
if a.sample == "OOS" and a.trust is None:
    raise SystemExit("OOS: pass ONE rule chosen in-sample (--trust, --pos_cap); no sweeps on OOS")

t = pd.read_csv(f"{config.PATHS['results']}/trades_{a.sample}_{a.set}.csv")
m = pd.read_parquet(f"{config.PATHS['canonical']}/matches.parquet").set_index("match_id")
t["t"] = pd.to_datetime(t.t, utc=True)
t["settle"] = pd.to_datetime(m.loc[t.match_id, "kickoff"].to_numpy(), utc=True) + pd.Timedelta(minutes=125)
buy = t.side == 1
t["q"] = np.where(buy, t.fill, 1 - t.fill)                  # price paid per contract for our side
t["pw"] = np.where(buy, t.p, 1 - t.p)                       # model prob our side wins
t["win"] = np.where(buy, t.won == 1, t.won == 0)
t["fee"] = kalshi_fee(t.fill, config.BACKTEST["fee_rate"])
t["kelly"] = (t.pw - t.q - t.fee) / (1 - t.q)               # full-Kelly fraction at the fill, after fee
t = t.sort_values("t").reset_index(drop=True)


def simulate(trust, dd_stop, bank, pos_cap=1.0):
    cash, peak, mdd, low, stopped, n = bank, bank, 0.0, bank, False, 0
    open_pos = []                                            # (settle_time, payout)
    for r in t.itertuples():
        for s, pay in [x for x in open_pos if x[0] <= r.t]:  # settle finished matches first
            cash += pay
        open_pos = [x for x in open_pos if x[0] > r.t]
        peak = max(peak, cash)
        mdd = max(mdd, 1 - cash / peak)
        low = min(low, cash)
        if dd_stop and 1 - cash / peak >= dd_stop:
            stopped = True
        if stopped or r.kelly <= 0:
            continue
        stake = min(trust * r.kelly * cash, pos_cap * cash, cash)   # position cap, never more than cash
        k = stake / (r.q + r.fee)
        if k <= 0:
            continue
        cash -= k * (r.q + r.fee)
        open_pos.append((r.settle, k * (1.0 if r.win else 0.0)))
        n += 1
    for s, pay in open_pos:
        cash += pay
    peak = max(peak, cash)
    mdd = max(mdd, 1 - cash / peak)
    return dict(trades=n, final=cash, ret=cash / bank - 1, low=min(low, cash), max_dd=mdd, stopped=stopped)


if a.trust is not None:                                     # single pre-chosen rule
    import json
    r = simulate(a.trust, None, a.bankroll, a.pos_cap)
    print(f"{a.sample}: {len(t)} signals, trust {a.trust}, pos_cap {a.pos_cap:.0%}, ${a.bankroll:,.0f} start")
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    if a.sample == "OOS":
        with open(f"{config.PATHS['results']}/oos_runs.log", "a") as fh:
            fh.write(json.dumps(dict(time=datetime.now().isoformat(timespec="seconds"), set=a.set,
                                     note="POST-HOC sizing on the official OOS signals (2nd OOS look)",
                                     trust=a.trust, pos_cap=a.pos_cap, bankroll=a.bankroll, result=r), default=str) + "\n")
    raise SystemExit

print(f"{len(t)} in-sample signals; {(t.kelly > 0).sum()} still have an edge at the fill price "
      f"({(t.kelly <= 0).sum()} cancelled). Bankroll ${a.bankroll:,.0f}, compounding.\n")
rows = []
for trust in [1.0, 0.5, 0.25, 0.1, 0.05]:
    for cap in [1.0, 0.25, 0.10, 0.05, 0.02]:
        r = simulate(trust, None, a.bankroll, cap)
        rows.append(dict(trust=trust, pos_cap="none" if cap == 1.0 else f"{cap:.0%}", dd_stop="none", **r))
out = pd.DataFrame(rows)
out["ret_over_dd"] = out.ret / out.max_dd
show = out.assign(final=out.final.map("${:,.0f}".format), ret=out.ret.map("{:+.1%}".format),
                  low=out.low.map("${:,.0f}".format), max_dd=out.max_dd.map("{:.1%}".format),
                  ret_over_dd=out.ret_over_dd.round(2)).drop(columns=["dd_stop", "stopped"])
print(show.to_string(index=False))

f = f"{config.PATHS['results']}/trials.csv"
old = pd.read_csv(f)
now = datetime.now().isoformat(timespec="seconds")
new = [{**config.BACKTEST, "trial": len(old) + i + 1, "time": now, "sample": "IS", "feature_set": a.set,
        "kind": config.MODEL["kind"], "C": config.MODEL["C"], "retrain": config.MODEL["retrain"],
        "sizing": "kelly_compound", "kelly_frac": r.trust, "bankroll": a.bankroll, "dd_stop": r.dd_stop,
        "pos_cap": r.pos_cap,
        "net_trades": r.trades, "net_pnl_total": round(r.final - a.bankroll, 2), "max_dd_pct": round(r.max_dd, 4),
        "note": "sizing sweep on in-sample signals of the flat walk-forward run"} for i, r in enumerate(out.itertuples())]
pd.concat([old, pd.DataFrame(new)], ignore_index=True).to_csv(f, index=False)
print(f"\n[{len(new)} sizing variants logged as trials #{len(old) + 1}-#{len(old) + len(new)} in {f}]")
