"""
Reproduce the note's headline numbers for the locked config (config.MODEL / config.BACKTEST) and
draw the equity curve. Re-computes the in-sample walk-forward run and the official OOS run
(same signals, nothing tuned here, nothing logged as a new trial).

    python report.py      ->  results/figures/equity_<set>.png, results/summary_<set>.csv
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from backtest import metrics, run_backtest
from run import bt_params, kalshi_frame, load, walkforward_probs

# reference palette (light): categorical slots 1-2, text and surface tokens
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
NET, X2 = "#2a78d6", "#eb6834"

FSET, M = config.MODEL["feature_set"], config.MODEL


class _A:                                                   # bt_params reads CLI-style overrides
    pass


def run(c, cost_mult):
    p = {**bt_params(_A()), "cost_mult": config.BACKTEST["cost_mult"] * cost_mult}
    return run_backtest(c, PROB.loc[c.index], **p)


matches, events, candles, kstart, oos_start = load(config.PATHS["canonical"])
c = kalshi_frame(candles, matches, events, config.PATHS["canonical"])
is_c, oos_c = c[c.date < oos_start], c[c.date >= oos_start]
# in-sample and OOS predicted separately: each month's model only sees matches before that month
PROB = pd.concat([walkforward_probs(matches, events, is_c, FSET, M["kind"], M["C"]),
                  walkforward_probs(matches, events, oos_c, FSET, M["kind"], M["C"])])

res = {}
for name, sample in (("in-sample", is_c), ("out-of-sample", oos_c)):
    for mult in (1, 2):
        tr = run(sample, mult).sort_values("t")
        res[(name, mult)] = tr
summary = pd.DataFrame([{"sample": s, "costs": f"x{k}", **metrics(tr)} for (s, k), tr in res.items()]).drop(columns="label")
os.makedirs(f"{config.PATHS['results']}/figures", exist_ok=True)
summary.to_csv(f"{config.PATHS['results']}/summary_{FSET}.csv", index=False)
print(summary.to_string(index=False))

# ---------------------------------------------------------------- equity curve
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ends = {}
for mult, color, label in ((1, NET, "After fees"), (2, X2, "After 2x fees")):
    tr = pd.concat([res[("in-sample", mult)], res[("out-of-sample", mult)]])
    t = pd.to_datetime(tr.t, utc=True).dt.tz_convert(None)
    eq = tr.pnl.cumsum().to_numpy()
    ax.plot(np.r_[t.iloc[0], t], np.r_[0, eq], color=color, lw=2, drawstyle="steps-post", label=label,
            solid_capstyle="round", zorder=3)
    ends[mult] = (t.iloc[-1], eq[-1], color, label)

oos0 = pd.Timestamp(oos_start)
ax.axvspan(oos0, ends[1][0] + pd.Timedelta(days=4), color=INK2, alpha=0.06, lw=0, zorder=0)
ax.axvline(oos0, color=INK2, lw=1, ls=(0, (3, 3)), zorder=1)
ax.axhline(0, color=INK2, lw=0.8, zorder=1)
xt = ax.get_xaxis_transform()                              # x in data, y in axes fraction
ax.text(oos0 - pd.Timedelta(days=5), 0.03, "in-sample (walk-forward) ←", ha="right", va="bottom", transform=xt, color=INK2, fontsize=8)
ax.text(oos0 + pd.Timedelta(days=4), 0.03, "→ out-of-sample", ha="left", va="bottom", transform=xt, color=INK2, fontsize=8)
for mult, (tx, y, color, label) in ends.items():         # direct labels at the line ends, in text ink
    ax.annotate(f"{'+' if y >= 0 else '-'}${abs(y):.2f}", (tx, y), xytext=(6, 0), textcoords="offset points", va="center",
                color=INK, fontsize=8.5, fontweight="bold")

isn, oon = res[("in-sample", 1)], res[("out-of-sample", 1)]
mi, mo = metrics(isn), metrics(oon)
ax.set_title(f"{FSET}: cumulative P&L, 1 contract per trade", loc="left", color=INK, fontsize=11, fontweight="bold", pad=18)
ax.text(0, 1.02, f"In-sample {mi['trades']} trades, t = {mi['t_stat']:.2f}   |   out-of-sample {mo['trades']} trades, "
        f"t = {mo['t_stat']:.2f}   |   threshold {config.BACKTEST['threshold']}, cooldown {config.BACKTEST['cooldown']} min",
        transform=ax.transAxes, color=INK2, fontsize=7.5)
ax.set_ylabel("Cumulative P&L ($ per contract)")
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(length=0)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.set_xlim(right=ends[1][0] + pd.Timedelta(days=30))
leg = ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK)
fig.tight_layout()
out = f"{config.PATHS['results']}/figures/equity_{FSET}.png"
fig.savefig(out, facecolor=SURFACE)
print("saved", out)
