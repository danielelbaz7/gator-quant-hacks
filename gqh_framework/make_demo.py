"""Synthetic canonical data so the whole pipeline can be run end to end without real data.
python make_demo.py  ->  data/demo/{matches,events,candles}.parquet"""
import os
import numpy as np
import pandas as pd
from scipy.stats import poisson

rng = np.random.default_rng(0)
N_TRAIN, N_KAL = 4000, 400
leagues = ["EPL", "LALIGA", "BUNDESLIGA", "SERIEA", "LIGUE1"]


def outcome_probs(lh, la, mins, hg, ag):
    """P(H,D,A) given full-match rates, minute, current score (Poisson remaining goals)."""
    left = np.clip(90 - mins, 0, 90) / 90
    k = np.arange(11)
    ph = poisson.pmf(k[None, :], (lh * left)[:, None])
    pa = poisson.pmf(k[None, :], (la * left)[:, None])
    diff = (hg - ag)[:, None, None] + k[None, :, None] - k[None, None, :]
    joint = ph[:, :, None] * pa[:, None, :]
    return np.stack([(joint * (diff > 0)).sum((1, 2)), (joint * (diff == 0)).sum((1, 2)),
                     (joint * (diff < 0)).sum((1, 2))], 1)


rows, evs = [], []
dates = np.r_[pd.date_range("2019-08-01", "2025-12-31", periods=N_TRAIN),
              pd.date_range("2026-02-01", "2026-09-20", periods=N_KAL)]
for i, d in enumerate(dates):
    lh, la = np.exp(rng.normal(0.35, 0.3)), np.exp(rng.normal(0.1, 0.3))
    gh, ga = rng.poisson(lh), rng.poisson(la)
    for side, n in (("home", gh), ("away", ga)):
        for mnt in rng.uniform(1, 94, n):
            evs.append(("m%d" % i, float(int(mnt)), side, "goal", np.nan))
    if rng.random() < 0.12:
        evs.append(("m%d" % i, float(rng.integers(10, 88)), rng.choice(["home", "away"]), "red", np.nan))
    p = outcome_probs(np.array([lh]), np.array([la]), np.array([0.]), np.array([0]), np.array([0]))[0]
    p = np.clip(p * np.exp(rng.normal(0, 0.05, 3)), 0.01, None); p /= p.sum()
    res = "H" if gh > ga else "A" if ga > gh else "D"
    rows.append(dict(match_id="m%d" % i, league=leagues[i % 5], date=pd.Timestamp(d).normalize(),
                     home="T%d" % rng.integers(100), away="T%d" % rng.integers(100),
                     home_goals=gh, away_goals=ga, result=res, p_home=p[0], p_draw=p[1], p_away=p[2],
                     home_value=np.exp(rng.normal(19 + lh, .3)), away_value=np.exp(rng.normal(19 + la, .3)),
                     h2h_gd=rng.normal(0, 1), lh=lh, la=la,
                     kickoff=pd.Timestamp(d).tz_localize("UTC") + pd.Timedelta(hours=15)))
m = pd.DataFrame(rows)
e = pd.DataFrame(evs, columns=["match_id", "minute", "side", "kind", "xg"])

# Kalshi candles for the last N_KAL matches: true in-play prob + noise, spread 2-4c
cs = []
for r in m.iloc[N_TRAIN:].itertuples():
    clock = np.arange(-60, 116)
    FH, HT = 47, 15
    mins = np.where(clock < 0, 0, np.where(clock <= FH, clock, np.where(clock <= FH + HT, 45, 45 + clock - FH - HT))).astype(float)
    g = e[(e.match_id == r.match_id) & (e.kind == "goal")]
    hg = np.searchsorted(np.sort(g[g.side == "home"].minute), mins, side="right")
    ag = np.searchsorted(np.sort(g[g.side == "away"].minute), mins, side="right")
    P = outcome_probs(np.full(len(mins), r.lh), np.full(len(mins), r.la), mins, hg, ag)
    for k, (out, won) in enumerate([("home", r.result == "H"), ("draw", r.result == "D"), ("away", r.result == "A")]):
        mid = np.clip(P[:, k] + rng.normal(0, 0.04, len(mins)), 0.01, 0.99)
        half = rng.choice([0.01, 0.015, 0.02], len(mins))
        q = pd.DataFrame(dict(ticker=f"{r.match_id}-{out}", match_id=r.match_id, outcome=out, won=int(won),
                              t=r.kickoff + pd.to_timedelta(clock, unit="min"),
                              bid=np.round(mid - half, 2), ask=np.round(mid + half, 2)))
        q.loc[rng.random(len(q)) < 0.3, ["bid", "ask"]] = np.nan
        cs.append(q)
c = pd.concat(cs, ignore_index=True)
os.makedirs("data/demo", exist_ok=True)
m.drop(columns=["lh", "la"]).to_parquet("data/demo/matches.parquet")
e.to_parquet("data/demo/events.parquet")
c.to_parquet("data/demo/candles.parquet")
print(len(m), "matches", len(e), "events", len(c), "candles -> data/demo")
