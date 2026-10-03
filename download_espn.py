"""
Real match timelines from ESPN's public API: actual kickoff, halftime, 2nd-half start and
full-time wall-clock times, plus goal and red-card times. Used to map Kalshi candle times to
the real match clock (stoppage time and halftime length vary per game).

    python download_espn.py

Only dates that have Kalshi markets are fetched (from big5_markets.parquet).
Outputs (git-ignored): espn_games.parquet, espn_events.parquet. Reruns skip games already fetched.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from config import PATHS

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
ESPN_LEAGUE = {"EPL": "eng.1", "LALIGA": "esp.1", "BUNDESLIGA": "ger.1", "SERIEA": "ita.1", "LIGUE1": "fra.1"}
KALSHI_LEAGUE = {"KXEPLGAME": "EPL", "KXLALIGAGAME": "LALIGA", "KXBUNDESLIGAGAME": "BUNDESLIGA",
                 "KXSERIEAGAME": "SERIEA", "KXLIGUE1GAME": "LIGUE1"}
S = requests.Session()


def get(path, **params):
    for _ in range(4):
        try:
            r = S.get(f"{BASE}/{path}", params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
    return None


def kalshi_dates():
    mk = pd.read_parquet(PATHS["kalshi_markets"])
    mk["league"] = mk.series.map(KALSHI_LEAGUE)
    d = mk.event_ticker.str.extract(r"-(\d{2})([A-Z]{3})(\d{2})")
    tick = pd.to_datetime("20" + d[0] + d[1].str.title() + d[2], format="%Y%b%d", errors="coerce")
    close = pd.to_datetime(mk.close_time, format="ISO8601").dt.tz_convert(None).dt.normalize()
    out = set()
    for lg, a, b in zip(mk.league, tick, close):    # ticker date (scheduled) and close date (played)
        for day in {a, b, a - pd.Timedelta(days=1)}:
            if pd.notna(day):
                out.add((lg, day.strftime("%Y%m%d")))
    return sorted(out)


def scoreboard(item):
    lg, day = item
    d = get(f"{ESPN_LEAGUE[lg]}/scoreboard", dates=day) or {}
    rows = []
    for e in d.get("events", []):
        c = e["competitions"][0]
        t = {x["homeAway"]: x for x in c["competitors"]}
        if "home" not in t or "away" not in t:
            continue
        rows.append(dict(espn_id=e["id"], league=lg, scheduled=e["date"], status=c["status"]["type"]["name"],
                         home=t["home"]["team"]["displayName"], away=t["away"]["team"]["displayName"],
                         home_id=t["home"]["team"]["id"], away_id=t["away"]["team"]["id"],
                         home_score=t["home"].get("score"), away_score=t["away"].get("score")))
    return rows


def timeline(g):
    d = get(f"{ESPN_LEAGUE[g.league]}/summary", event=g.espn_id) or {}
    return [dict(espn_id=g.espn_id, type=k["type"]["type"], period=k.get("period", {}).get("number"),
                 clock=k.get("clock", {}).get("displayValue"), wallclock=k.get("wallclock"),
                 team_id=(k.get("team") or {}).get("id"), scoring=bool(k.get("scoringPlay")))
            for k in d.get("keyEvents", []) if k["type"]["type"] not in ("substitution", "yellow-card")]


def main(workers=8):
    items = kalshi_dates()
    print(f"ESPN scoreboards: {len(items)} league-days")
    with ThreadPoolExecutor(workers) as ex:
        games = pd.DataFrame([r for rows in ex.map(scoreboard, items) for r in rows])
    games = games.drop_duplicates("espn_id")
    games = games[games.status == "STATUS_FULL_TIME"].reset_index(drop=True)
    games.to_parquet("espn_games.parquet")
    print(f"  {len(games)} finished games")

    done = pd.read_parquet("espn_events.parquet") if os.path.exists("espn_events.parquet") else None
    todo = games[~games.espn_id.isin(done.espn_id)] if done is not None else games
    print(f"ESPN timelines: fetching {len(todo)} games ({workers} workers)...")
    rows = []
    with ThreadPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(timeline, todo.itertuples()), 1):
            rows.extend(r)
            if i % 300 == 0:
                print(f"  {i}/{len(todo)}")
    ev = pd.concat([done, pd.DataFrame(rows)]) if done is not None else pd.DataFrame(rows)
    ev.to_parquet("espn_events.parquet")
    have = ev[ev.type == "start-2nd-half"].espn_id.nunique()
    print(f"  saved {len(ev)} events; {have}/{len(games)} games have a 2nd-half start time")


if __name__ == "__main__":
    main()
