"""
Quick Kalshi soccer collector (public endpoints, no API key).

UNTESTED: written from docs/snippets and run nowhere yet. Run it with
--probe first, check the printed JSON keys, and fix field names if needed.

    pip install requests pandas pyarrow
    python kalshi_soccer_pull.py --probe                 # see what exists (fast)
    python kalshi_soccer_pull.py --series KXEPLGAME --max-markets 50
    python kalshi_soccer_pull.py --max-markets 300       # all soccer series
"""
import argparse, json, time
import requests
import pandas as pd

BASE = "https://api.elections.kalshi.com/trade-api/v2"
S = requests.Session()


def get(path, **params):
    for attempt in range(5):
        r = S.get(BASE + path, params={k: v for k, v in params.items() if v is not None}, timeout=30)
        if r.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"rate limited: {path}")


def paged(path, key, **params):
    cursor = None
    while True:
        d = get(path, cursor=cursor, limit=200, **params)
        yield from d.get(key, [])
        cursor = d.get("cursor")
        if not cursor:
            return


def soccer_series():
    d = get("/series", category="Sports", tags="Soccer")
    return [s["ticker"] for s in d.get("series", [])]


def settled_markets(series, max_markets):
    """Live tier first, then historical tier (older than ~3 months)."""
    out = []
    for m in paged("/markets", "markets", series_ticker=series, status="settled"):
        out.append(("live", m))
        if len(out) >= max_markets:
            return out
    try:
        for m in paged("/historical/markets", "markets", series_ticker=series):
            out.append(("historical", m))
            if len(out) >= max_markets:
                break
    except requests.HTTPError as e:
        print(f"  historical markets query failed for {series}: {e} (check docs for params)")
    return out


def candles(series, ticker, tier, start_ts, end_ts, period=60):
    if tier == "historical":
        path = f"/historical/markets/{ticker}/candlesticks"
    else:
        path = f"/series/{series}/markets/{ticker}/candlesticks"
    d = get(path, start_ts=start_ts, end_ts=end_ts, period_interval=period)
    return d.get("candlesticks", [])


def flatten(c):
    row = {"end_ts": c.get("end_period_ts"), "volume": c.get("volume"),
           "open_interest": c.get("open_interest")}
    for side in ("yes_bid", "yes_ask", "price"):
        for k, v in (c.get(side) or {}).items():
            row[f"{side}_{k}"] = v
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--series", nargs="*")
    ap.add_argument("--max-markets", type=int, default=50)
    ap.add_argument("--period", type=int, default=60, help="minutes: 1, 60 or 1440")
    ap.add_argument("--days-before-close", type=int, default=3)
    a = ap.parse_args()

    series_list = a.series or soccer_series()
    print(f"{len(series_list)} soccer series: {series_list[:15]}")

    if a.probe:
        print("cutoffs:", json.dumps(get("/historical/cutoff"), indent=1))
        for s in series_list[:3]:
            ms = settled_markets(s, 5)
            print(s, "-> sample markets:", len(ms))
            if ms:
                tier, m = ms[0]
                print(tier, json.dumps(m, indent=1)[:1500])
                close = pd.Timestamp(m["close_time"]).timestamp()
                cs = candles(s, m["ticker"], tier, int(close) - 86400 * 3, int(close), a.period)
                print("candle sample:", json.dumps(cs[:1], indent=1))
        return

    mrows, crows = [], []
    for s in series_list:
        ms = settled_markets(s, a.max_markets)
        print(f"{s}: {len(ms)} settled markets")
        for tier, m in ms:
            mrows.append({"series": s, "tier": tier, **{k: m.get(k) for k in (
                "ticker", "event_ticker", "title", "yes_sub_title", "status",
                "result", "open_time", "close_time", "volume")}})
            try:
                close = int(pd.Timestamp(m["close_time"]).timestamp())
                for c in candles(s, m["ticker"], tier, close - 86400 * a.days_before_close, close, a.period):
                    crows.append({"ticker": m["ticker"], **flatten(c)})
            except Exception as e:
                print(f"  skip {m.get('ticker')}: {e}")
            time.sleep(0.1)

    pd.DataFrame(mrows).to_parquet("kalshi_soccer_markets.parquet")
    pd.DataFrame(crows).to_parquet("kalshi_soccer_candles.parquet")
    print(f"saved {len(mrows)} markets, {len(crows)} candles")


if __name__ == "__main__":
    main()
