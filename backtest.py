"""
Trading rule (per Kalshi outcome market, e.g. "Arsenal wins"):

  p   = model probability for that outcome at bar t (state known at t minus event lag)
  buy YES  if p - ask - cost(ask) > threshold      (market too cheap)
  buy NO   if bid - p - cost(bid) > threshold      (market too expensive; same as selling YES)

Fill at the NEXT bar's quotes (exec_next_bar), hold to settlement.
YES pays $1 if the outcome happens. One contract per trade, so P&L is in dollars per contract.
"""
import numpy as np
import pandas as pd


def kalshi_fee(price, rate):
    """Approximate taker fee per contract: rate * P * (1 - P). Check Kalshi's current schedule."""
    return rate * price * (1 - price)


def run_backtest(c, prob, threshold, max_spread, min_minute, max_minute, skip_halftime,
                 fee_rate, slippage, cost_mult, exec_next_bar, one_per, block_prematch=False):
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
    tr["capital"] = np.where(buy, tr.fill, 1 - tr.fill)
    tr["costs"] = cost(tr.fill)
    tr["pnl"] = np.where(buy, tr.won - tr.fill, tr.fill - tr.won) - tr.costs
    tr["edge_at_signal"] = np.where(buy, edge_buy[tr.index], edge_sell[tr.index])
    return tr[["match_id", "ticker", "league", "outcome", "t", "phase", "minute", "side",
               "p", "bid", "ask", "fill", "won", "capital", "costs", "pnl", "edge_at_signal"]]


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
