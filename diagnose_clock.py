"""
Diagnostic: does our goal timing line up with reality?
For each goal (in-sample matches only), look at the scoring team's Kalshi "win" price around
the time we think the goal happened. If our timing is right, the price jumps at offset ~0.
If it jumps at +k minutes, our clock runs k minutes ahead of the real match (=> the model
would see goals before they happen).
  old = Transfermarkt minute + fixed config.TIME mapping
  new = ESPN wall-clock goal time (what the backtest uses when available)

    python diagnose_clock.py
"""
import numpy as np
import pandas as pd

import config
from data import load_live_events
from run import load

matches, events, candles, kstart, oos_start = load(config.PATHS["canonical"])
FH, HT = config.TIME["first_half_clock"], config.TIME["halftime_break"]
m = matches.set_index("match_id")
ids = set(candles.match_id) & set(m.index[m.date < oos_start])

g = events[(events.kind == "goal") & events.match_id.isin(ids)].copy()
g = g[g.minute.between(5, 40) | g.minute.between(50, 85)]             # avoid stoppage-time ambiguity
g["half"] = np.where(g.minute <= 45, "1H", "2H")
g["clock"] = np.where(g.half == "1H", g.minute, g.minute - 45 + FH + HT)
g["t_goal"] = pd.to_datetime(m.loc[g.match_id, "kickoff_utc"].to_numpy(), utc=True) + pd.to_timedelta(g.clock, "min")
g["src"] = "old"
live = load_live_events()
if live is not None:
    lv = live[(live.kind == "goal") & live.match_id.isin(ids)].rename(columns={"t": "t_goal"})
    lv["clock"] = (lv.t_goal - pd.to_datetime(m.loc[lv.match_id, "kickoff"].to_numpy(), utc=True)).dt.total_seconds() / 60
    lv = lv[lv.clock.between(5, 40) | lv.clock.between(66, 105)]
    lv["half"], lv["src"] = np.where(lv.clock < 45, "1H", "2H"), "new"
    g = pd.concat([g, lv[["match_id", "side", "half", "t_goal", "src"]]])

c = candles[candles.outcome.isin(["home", "away"])].copy()
c["mid"] = (c.bid + c.ask) / 2
c = c.dropna(subset=["mid"])
c["t"] = pd.to_datetime(c.t, utc=True)
by = {k: v.set_index("t").mid.sort_index() for k, v in c.groupby(["match_id", "outcome"])}

offs = np.arange(-10, 11)
rows = []
for r in g.itertuples():
    s = by.get((r.match_id, r.side))
    if s is None:
        continue
    grid = r.t_goal + pd.to_timedelta(offs, "min")
    v = s.reindex(s.index.union(grid)).ffill().loc[grid].to_numpy()
    if np.isnan(v[0]) or np.isnan(v[-1]):
        continue
    rows.append({"src": r.src, "half": r.half, **{int(o): x - v[0] for o, x in zip(offs, v)}})
d = pd.DataFrame(rows)
print(f"\n{len(d)} goals. Median move in scoring team's win price (cents) vs 10 min before our goal time:")
tab = (d.groupby(["src", "half"])[list(offs)].median() * 100).round(1)
print(tab.to_string())
for (s, h), row in tab.iterrows():
    print(f"  {s} {h}: biggest 1-minute jump lands at offset {row.diff().idxmax():+d} min")
