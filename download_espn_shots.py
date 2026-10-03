"""
Shot-by-shot data from ESPN match commentary for every big-5 match since 2016/17
(dates come from the Transfermarkt matches in data/canonical, so run `python run.py build` first).

    python download_espn_shots.py

Output (git-ignored): espn_shots.parquet, one row per attempt or goal:
  espn_id, league, home, away, scheduled, period, clock_s (match-clock seconds), minute (displayed
  minute, stoppage time folded into 45/90), team, play_type, text, wallclock
Reruns skip games already fetched.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from config import PATHS
from download_espn import ESPN_LEAGUE, get, scoreboard

OUT = "espn_shots.parquet"
START = "2016-07-01"


def shots(g):
    d = get(f"{ESPN_LEAGUE[g.league]}/summary", event=g.espn_id) or {}
    rows = []
    for c in d.get("commentary", []):
        p = c.get("play") or {}
        typ = (p.get("type") or {}).get("type", "")
        text = c.get("text", "")
        if not (("shot" in typ) or ("goal" in typ) or ("penalty" in typ) or text.startswith(("Attempt", "Goal!", "Penalty"))):
            continue
        if "own-goal" in typ or text.startswith("Own Goal"):
            continue
        disp = (c.get("time") or {}).get("displayValue", "") or ""
        base = pd.to_numeric(disp.split("'")[0], errors="coerce")
        rows.append(dict(espn_id=g.espn_id, league=g.league, home=g.home, away=g.away, scheduled=g.scheduled,
                         period=(p.get("period") or {}).get("number"), clock_s=(c.get("time") or {}).get("value"),
                         minute=base, team=(p.get("team") or {}).get("displayName"), play_type=typ,
                         text=text, wallclock=p.get("wallclock")))
    if not rows:   # keep a marker so reruns know this game was fetched (it just had no commentary)
        rows.append(dict(espn_id=g.espn_id, league=g.league, home=g.home, away=g.away, scheduled=g.scheduled,
                         play_type="__none__"))
    return rows


def main(workers=8):
    m = pd.read_parquet(f"{PATHS['canonical']}/matches.parquet")
    m = m[pd.to_datetime(m.date) >= START]
    items = sorted({(lg, d.strftime("%Y%m%d")) for lg, d in zip(m.league, pd.to_datetime(m.date))})
    print(f"ESPN scoreboards: {len(items)} league-days since {START}")
    with ThreadPoolExecutor(workers) as ex:
        games = pd.DataFrame([r for rows in ex.map(scoreboard, items) for r in rows]).drop_duplicates("espn_id")
    games = games[games.status == "STATUS_FULL_TIME"].reset_index(drop=True)
    print(f"  {len(games)} finished games")

    done = pd.read_parquet(OUT) if os.path.exists(OUT) else None
    todo = games[~games.espn_id.isin(done.espn_id)] if done is not None else games
    print(f"ESPN commentary: fetching {len(todo)} games ({workers} workers)...")
    rows = []
    with ThreadPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(shots, todo.itertuples()), 1):
            rows.extend(r)
            if i % 1000 == 0:
                print(f"  {i}/{len(todo)}")
    out = pd.concat([done, pd.DataFrame(rows)]) if done is not None else pd.DataFrame(rows)
    out.to_parquet(OUT)
    real = out[out.play_type != "__none__"]
    print(f"  saved {len(real)} shots from {real.espn_id.nunique()} games "
          f"({out.espn_id.nunique() - real.espn_id.nunique()} games had no commentary)")


if __name__ == "__main__":
    main()
