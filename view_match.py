"""
Look at one match from the Kalshi pull.

    python -m pip install matplotlib
    python view_match.py --list              # list matches with volume
    python view_match.py                     # plot the most-traded match
    python view_match.py --event KXEPLGAME-26FEB18WOLARS
    python view_match.py --summary           # coverage/spread stats for all markets
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt

m = pd.read_parquet("kalshi_soccer_markets.parquet")
c = pd.read_parquet("kalshi_soccer_candles.parquet")

def pick(name):
    """Use the *_dollars column where filled, else the cents column / 100."""
    d = pd.to_numeric(c[f"{name}_dollars"], errors="coerce")
    k = pd.to_numeric(c[name], errors="coerce")
    if k.max() > 1.5:      # real cents (0-100) -> convert to 0-1
        k = k / 100
    return d.fillna(k)     # result is always a 0-1 probability

c["volume"] = pd.to_numeric(c["volume"], errors="coerce")
c["bid"] = pick("yes_bid_close")
c["ask"] = pick("yes_ask_close")
c["px"] = pick("price_close")
c["time"] = pd.to_datetime(c["end_ts"], unit="s", utc=True)
c["spread"] = c["ask"] - c["bid"]
print("non-null share  bid:", round(c.bid.notna().mean(), 3),
      " ask:", round(c.ask.notna().mean(), 3), " price:", round(c.px.notna().mean(), 3))
print("ranges (should be within 0-1)  bid:", (c.bid.min(), c.bid.max()),
      " ask:", (c.ask.min(), c.ask.max()), " price:", (c.px.min(), c.px.max()))
c = c.merge(m[["ticker", "event_ticker", "yes_sub_title", "result"]], on="ticker", how="left")

ap = argparse.ArgumentParser()
ap.add_argument("--event")
ap.add_argument("--list", action="store_true")
ap.add_argument("--summary", action="store_true")
a = ap.parse_args()

per_event = c.groupby("event_ticker").agg(
    volume=("volume", "sum"), start=("time", "min"), end=("time", "max"), n_candles=("time", "size"))

if a.list:
    print(per_event.sort_values("volume", ascending=False).to_string())
    raise SystemExit

if a.summary:
    s = c.groupby("ticker").agg(
        result=("result", "first"),
        volume=("volume", "sum"),
        quoted=("spread", lambda x: x.notna().mean()),
        median_spread=("spread", "median"),
    )
    print(s.describe().to_string())
    print("\nmedian spread by outcome result:\n", s.groupby("result").median_spread.median())
    raise SystemExit

event = a.event or per_event.volume.idxmax()
d = c[c.event_ticker == event]
print(event)
print(d.groupby(["ticker", "yes_sub_title"]).agg(
    result=("result", "first"), volume=("volume", "sum"),
    quoted=("spread", lambda x: x.notna().mean()),
    median_spread=("spread", "median")).to_string())

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1]})
for (tick, label), g in d.groupby(["ticker", "yes_sub_title"]):
    g = g.sort_values("time")
    line, = ax1.plot(g.time, g.px.ffill(), lw=1, label=f"{label} ({g.result.iloc[0]})")
    ax1.fill_between(g.time, g.bid, g.ask, color=line.get_color(), alpha=0.3)
    ax2.bar(g.time, g.volume, width=0.0007, color=line.get_color(), alpha=0.6)
from matplotlib.ticker import PercentFormatter
ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
ax1.set_ylim(0, 1)
ax1.set_ylabel("Implied probability (line) with bid-ask band")
ax1.set_title(event)
ax1.legend()
ax2.set_ylabel("volume / min")
plt.tight_layout()
plt.show()