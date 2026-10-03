"""
Trading rule (per Kalshi outcome market, e.g. "Arsenal wins"):

  p   = model probability for that outcome at bar t (state known at t minus event lag)
  buy YES  if p - ask - cost(ask) > threshold      (market too cheap)
  buy NO   if bid - p - cost(bid) > threshold      (market too expensive; same as selling YES)

Fill at the NEXT bar's quotes (exec_next_bar), hold to settlement.
Cooldown: no trade within `cooldown` minutes after any market of the match moved its mid by
more than `jump` in one bar. A live trader sees this move; it usually means an event our
(lagged) data feed hasn't delivered yet.
YES pays $1 if the outcome happens. One contract per trade, so P&L is in dollars per contract.
"""
import numpy as np
import pandas as pd


def kalshi_fee(price, rate):
    """Approximate taker fee per contract: rate * P * (1 - P). Check Kalshi's current schedule."""
    return rate * price * (1 - price)


def last_jump_age(c, jump):
    """Minutes since any market of the same match last moved its mid by more than `jump` in
    one quoted bar (up to and including bar t, so it is known at decision time). NaN = never."""
    mid = (c.bid + c.ask) / 2
    moved = (mid.groupby(c.ticker).diff().abs() > jump).to_numpy()
    t = pd.to_datetime(c.t, utc=True)
    jumps = pd.DataFrame({"match_id": c.match_id[moved].to_numpy(), "tj": t[moved].to_numpy()})
    q = pd.DataFrame({"match_id": c.match_id.to_numpy(), "t": t.to_numpy(), "i": np.arange(len(c))})
    out = pd.merge_asof(q.sort_values("t"), jumps.sort_values("tj"), left_on="t", right_on="tj",
                        by="match_id", direction="backward")
    age = (out.t - out.tj).dt.total_seconds() / 60
    return pd.Series(age.to_numpy(), index=out.i.to_numpy()).sort_index().to_numpy()


def run_backtest(c, prob, threshold, max_spread, min_minute, max_minute, skip_halftime,
                 fee_rate, slippage, cost_mult, exec_next_bar, one_per, cooldown=0, jump=0.07,
                 sizing="flat", size_cap=1.0, bankroll=100.0, kelly_frac=0.25, block_prematch=False):
    c = c.assign(p=prob).sort_values(["ticker", "t"]).reset_index(drop=True)
    if exec_next_bar:
        g = c.groupby("ticker")
        c["xbid"], c["xask"] = g.bid.shift(-1), g.ask.shift(-1)
    else:
        c["xbid"], c["xask"] = c.bid, c.ask

    signed = np.where(c.phase == "pre", c.clock, c.minute)       # pre-match = negative clock minutes
    lo = max(min_minute, 0) if block_prematch else min_minute
    ok = (c.bid.notna() & c.ask.notna() & c.p.notna()
          & ((c.ask - c.bid) <= max_spread)
          & (signed >= lo) & (signed <= max_minute)
          & (c.phase != "post"))
    if skip_halftime:
        ok &= c.phase != "HT"
    if cooldown > 0:   # market just jumped (likely a goal/card our feed hasn't delivered yet): stand aside
        age = last_jump_age(c, jump)
        ok &= ~(age <= cooldown)

    cost = lambda px: cost_mult * (kalshi_fee(px, fee_rate) + slippage)
    edge_buy = c.p - c.ask - cost(c.ask)
    edge_sell = c.bid - c.p - cost(c.bid)
    side = np.where(ok & (edge_buy > threshold), 1, np.where(ok & (edge_sell > threshold), -1, 0))
    c["side"] = side
    sig = c[c.side != 0]
    key = "ticker" if one_per == "ticker" else "match_id"
    sig = sig.groupby(key, sort=False).head(1)                    # first signal only

    px = np.where(sig.side == 1, sig.xask, sig.xbid)
    tr = sig.assign(fill=px).dropna(subset=["fill"]).copy()
    buy = tr.side == 1
    tr["edge_at_signal"] = np.where(buy, edge_buy[tr.index], edge_sell[tr.index])
    # position size, decided from signal-bar information only (edge, model p, quotes at bar t)
    if sizing == "edge":            # 1 contract at the threshold, more for bigger edges
        tr["contracts"] = np.clip(tr.edge_at_signal / threshold, 1.0, size_cap)
    elif sizing == "kelly":         # fractional Kelly stake on a fixed bankroll, in contracts
        q = np.where(buy, tr.ask, 1 - tr.bid)                     # price of the side we buy
        pw = np.where(buy, tr.p, 1 - tr.p)                        # model prob that side wins
        f = np.clip((pw - q) / (1 - q), 0, 1)
        tr["contracts"] = np.clip(kelly_frac * f * bankroll / q, 1.0, size_cap)
    else:
        tr["contracts"] = 1.0
    tr["capital"] = np.where(buy, tr.fill, 1 - tr.fill) * tr.contracts
    tr["costs"] = cost(tr.fill) * tr.contracts
    tr["pnl"] = np.where(buy, tr.won - tr.fill, tr.fill - tr.won) * tr.contracts - tr.costs
    return tr[["match_id", "ticker", "league", "outcome", "t", "phase", "minute", "side",
               "p", "bid", "ask", "fill", "won", "contracts", "capital", "costs", "pnl", "edge_at_signal"]]


def metrics(tr, label=""):
    if tr.empty:
        return dict(label=label, trades=0)
    day = tr.assign(d=pd.to_datetime(tr.t).dt.tz_convert(None).dt.normalize()).groupby("d").pnl.sum()
    day = day.reindex(pd.date_range(day.index.min(), day.index.max()), fill_value=0)
    cum = day.cumsum()
    sharpe = day.mean() / day.std() * np.sqrt(365) if day.std() > 0 else np.nan
    return dict(
        label=label,
        trades=len(tr),
        contracts=round(tr.contracts.sum(), 1) if "contracts" in tr else len(tr),
        matches=tr.match_id.nunique(),
        hit_rate=round((tr.pnl > 0).mean(), 3),
        pnl_total=round(tr.pnl.sum(), 2),
        pnl_per_trade=round(tr.pnl.mean(), 4),
        return_on_capital=round(tr.pnl.sum() / tr.capital.sum(), 4),
        avg_cost=round(tr.costs.mean(), 4),
        sharpe_daily_ann=round(sharpe, 2),
        max_drawdown=round((cum - cum.cummax()).min(), 2),
        t_stat=round(tr.pnl.mean() / tr.pnl.std() * np.sqrt(len(tr)), 2) if len(tr) > 1 else np.nan,
    )


def breakdown(tr, by):
    if tr.empty:
        return pd.DataFrame()
    return tr.groupby(by, observed=True).agg(
        trades=("pnl", "size"), pnl=("pnl", "sum"), per_trade=("pnl", "mean"),
        hit=("pnl", lambda x: (x > 0).mean())).round(3)
