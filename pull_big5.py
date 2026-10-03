"""
Pull Kalshi game-result markets + candles for the big-5 European leagues.
Needs kalshi_soccer_pull.py in the same folder. UNTESTED, so run --list first.

    python pull_big5.py --list                     # step 1: see which series it finds
    python pull_big5.py                            # step 2: pull everything it found
    python pull_big5.py --series KXEPLGAME KXFOO   # or pull explicit series only
    python pull_big5.py --combine                  # merge per-series files

Each series is saved to data/<SERIES>_{markets,candles}.parquet as soon as it
finishes, and finished series are skipped on rerun, so interrupting is safe.
"""
import argparse, glob, os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import kalshi_soccer_pull as k

LEAGUES = {
    "Premier League": ["EPL", "PREMIER LEAGUE"],
    "La Liga": ["LALIGA", "LA LIGA"],
    "Ligue 1": ["LIGUE1", "LIGUE 1"],
    "Bundesliga": ["BUNDESLIGA"],
    "Serie A": ["SERIEA", "SERIE A"],
}


def candidates():
    d = k.get("/series", category="Sports", tags="Soccer")
    rows = []
    for s in d.get("series", []):
        t, title = s.get("ticker", ""), s.get("title", "") or ""
        hay = f"{t} {title}".upper()
        for league, keys in LEAGUES.items():
            if any(x in hay for x in keys):
                rows.append((league, t, title))
                break
    return sorted(rows)


def pull_series(s, period, days, workers, outdir):
    cpath = f"{outdir}/{s}_candles.parquet"
    if os.path.exists(cpath):
        print(f"{s}: already done, skipping")
        return
    ms = k.settled_markets(s, 10**6)
    seen, uniq = set(), []
    for tier, m in ms:                      # dedupe in case live/historical overlap
        if m["ticker"] not in seen:
            seen.add(m["ticker"])
            uniq.append((tier, m))
    print(f"{s}: {len(uniq)} settled markets, fetching candles ({workers} workers)...")

    def one(item):
        tier, m = item
        try:
            close = int(pd.Timestamp(m["close_time"]).timestamp())
            cs = k.candles(s, m["ticker"], tier, close - 86400 * days, close, period)
            return m, tier, [{"ticker": m["ticker"], **k.flatten(c)} for c in cs], None
        except Exception as e:
            return m, tier, [], e

    mrows, crows, failed = [], [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (m, tier, rows, err) in enumerate(ex.map(one, uniq), 1):
            mrows.append({"series": s, "tier": tier, **{f: m.get(f) for f in (
                "ticker", "event_ticker", "title", "yes_sub_title", "status",
                "result", "open_time", "close_time", "volume")}})
            crows.extend(rows)
            if err:
                failed += 1
                if failed <= 3:
                    print(f"  fail {m.get('ticker')}: {err}")
            if i % 200 == 0:
                print(f"  {s}: {i}/{len(uniq)}")
    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame(mrows).to_parquet(f"{outdir}/{s}_markets.parquet")
    pd.DataFrame(crows).to_parquet(cpath)
    print(f"{s}: saved {len(mrows)} markets, {len(crows)} candles, {failed} failed")


def combine(outdir):
    m = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{outdir}/*_markets.parquet")])
    c = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{outdir}/*_candles.parquet")])
    m.to_parquet("big5_markets.parquet")
    c.to_parquet("big5_candles.parquet")
    print(f"big5_markets.parquet: {len(m)} rows | big5_candles.parquet: {len(c)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--series", nargs="*")
    ap.add_argument("--period", type=int, default=1)
    ap.add_argument("--days-before-close", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--outdir", default="data")
    a = ap.parse_args()

    if a.combine:
        return combine(a.outdir)

    cands = candidates()
    if a.list:
        print("league | ticker | title   (* = looks like a game-result series)")
        for league, t, title in cands:
            print(f"{league:15s} | {t:28s} | {title}{'  *' if t.endswith('GAME') else ''}")
        print("\nWatch for lookalikes: 'Serie A' may also match Brazil/Ecuador, "
              "'Bundesliga' may match Austria. Pass --series explicitly if so.")
        return

    chosen = a.series or [t for _, t, _ in cands if t.endswith("GAME")]
    print("pulling:", chosen)
    for s in chosen:
        pull_series(s, a.period, a.days_before_close, a.workers, a.outdir)
    combine(a.outdir)


if __name__ == "__main__":
    main()