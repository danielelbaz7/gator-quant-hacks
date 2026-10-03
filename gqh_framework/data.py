"""
Turns raw sources into three canonical tables in data/canonical/:

  matches.parquet  one row per match: match_id, league, date, kickoff_utc, home, away,
                   home_goals, away_goals, result (H/D/A), p_home/p_draw/p_away,
                   home_value/away_value, h2h_gd
  events.parquet   match_id, minute, side (home/away), kind (goal/red/shot), xg
  candles.parquet  Kalshi: match_id, ticker, outcome (home/draw/away), t (UTC), bid, ask, won

The model and backtest only ever read these, so if a raw source has different
column names than assumed below, fix the adapter and nothing else changes.
Adapters marked ASSUMED were written from docs, not run on the real files.
"""
import os
import re
import difflib
import unicodedata

import numpy as np
import pandas as pd

from config import PATHS, TIME, SPLIT

LEAGUE_TM = {"GB1": "EPL", "ES1": "LALIGA", "L1": "BUNDESLIGA", "IT1": "SERIEA", "FR1": "LIGUE1"}
LEAGUE_FD = {"ENG-Premier League": "EPL", "ESP-La Liga": "LALIGA", "GER-Bundesliga": "BUNDESLIGA",
             "ITA-Serie A": "SERIEA", "FRA-Ligue 1": "LIGUE1"}
LEAGUE_KALSHI = {"KXEPLGAME": "EPL", "KXLALIGAGAME": "LALIGA", "KXBUNDESLIGAGAME": "BUNDESLIGA",
                 "KXSERIEAGAME": "SERIEA", "KXLIGUE1GAME": "LIGUE1"}

# --------------------------------------------------------------------------- names
STOP = {"fc", "afc", "cf", "sc", "ac", "as", "ssc", "ss", "us", "rc", "rcd", "cd", "ud", "sd",
        "sv", "vfl", "vfb", "tsg", "fsv", "bsc", "ogc", "losc", "club", "de", "calcio", "the",
        "1", "04", "05", "1899", "1846", "1848", "1907", "1909", "1910", "1913", "and", "hotspur",
        "football", "balompie", "futbol"}
_ALIASES = None


def _aliases():
    """team_aliases.csv with columns raw,target lets you force tricky names together."""
    global _ALIASES
    if _ALIASES is None:
        _ALIASES = {}
        if os.path.exists(PATHS["aliases"]):
            a = pd.read_csv(PATHS["aliases"])
            _ALIASES = {_basic(r): _basic(t) for r, t in zip(a.raw, a.target)}
    return _ALIASES


def _basic(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(w for w in s.split() if w not in STOP)


def norm(name):
    b = _basic(name)
    return _aliases().get(b, b)


def sim(a, b):
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    r = difflib.SequenceMatcher(None, a, b).ratio()
    if a in b or b in a:
        r = max(r, 0.9)
    return r


def match_games(left, right, tol_days=1, min_score=1.3, allow_swap=False):
    """left/right have key, league, date, home, away. Returns left.key -> (right.key, swapped, score)."""
    R = right.assign(_h=right.home.map(norm), _a=right.away.map(norm),
                     _d=pd.to_datetime(right.date).dt.normalize())
    by = {k: g for k, g in R.groupby(["league", "_d"])}
    out = {}
    for row in left.itertuples(index=False):
        d = pd.Timestamp(row.date).normalize()
        h, a = norm(row.home), norm(row.away)
        best = (0, None, False)
        for dd in range(-tol_days, tol_days + 1):
            g = by.get((row.league, d + pd.Timedelta(days=dd)))
            if g is None:
                continue
            for rk, rh, ra in zip(g.key, g._h, g._a):
                s = sim(h, rh) + sim(a, ra)
                if s > best[0]:
                    best = (s, rk, False)
                if allow_swap:
                    s2 = sim(h, ra) + sim(a, rh)
                    if s2 > best[0]:
                        best = (s2, rk, True)
        if best[0] >= min_score:
            out[row.key] = (best[1], best[2], best[0])
    return out


# --------------------------------------------------------------------------- Transfermarkt (ASSUMED schema)
def load_transfermarkt(path=None, start_season=None):
    import duckdb
    path = path or PATHS["tm_duckdb"]
    start_season = start_season or SPLIT["tm_start_season"]
    con = duckdb.connect(path, read_only=True)
    comps = "(" + ",".join(f"'{c}'" for c in LEAGUE_TM) + ")"
    g = con.execute(f"""
        select game_id, competition_id, season, date, home_club_id, away_club_id,
               home_club_goals, away_club_goals, home_club_name, away_club_name
        from games where competition_id in {comps} and season >= {start_season}""").df()
    ev = con.execute(f"""
        select e.game_id, e.minute, e.type, e.club_id, e.description
        from game_events e join games g on e.game_id = g.game_id
        where g.competition_id in {comps} and g.season >= {start_season}""").df()

    g = g.dropna(subset=["home_club_goals", "away_club_goals"])
    m = pd.DataFrame({
        "match_id": "tm" + g.game_id.astype(str),
        "league": g.competition_id.map(LEAGUE_TM),
        "season": g.season,
        "date": pd.to_datetime(g.date),
        "home": g.home_club_name, "away": g.away_club_name,
        "home_id": g.home_club_id, "away_id": g.away_club_id,
        "home_goals": g.home_club_goals.astype(int), "away_goals": g.away_club_goals.astype(int),
    })
    m["result"] = np.select([m.home_goals > m.away_goals, m.home_goals < m.away_goals], ["H", "A"], "D")

    ev = ev.merge(g[["game_id", "home_club_id", "away_club_id"]], on="game_id")
    ev["match_id"] = "tm" + ev.game_id.astype(str)
    ev["side"] = np.where(ev.club_id == ev.home_club_id, "home",
                          np.where(ev.club_id == ev.away_club_id, "away", None))
    desc = ev.description.fillna("").str.lower()
    goals = ev[ev.type.str.lower().str.startswith("goal")].copy()
    reds = ev[ev.type.str.lower().str.startswith("card") & desc.str.contains("red|second yellow")].copy()
    goals["own"] = desc[goals.index].str.contains("own")
    goals["kind"], reds["kind"] = "goal", "red"

    # Own goals: unclear which club_id Transfermarkt attaches. Try both conventions per match
    # and keep whichever reproduces the final score; drop matches where neither does.
    goals, bad = _resolve_own_goals(goals, m)
    m = m[~m.match_id.isin(bad)]
    events = pd.concat([goals, reds])[["match_id", "minute", "side", "kind"]].dropna(subset=["side"])
    events["minute"] = pd.to_numeric(events.minute, errors="coerce")
    events["xg"] = np.nan

    m = m.merge(_squad_values(con, g), on="match_id", how="left")
    m["h2h_gd"] = _h2h(m)
    return m, events.dropna(subset=["minute"])


def _resolve_own_goals(goals, m):
    flip = goals.side.map({"home": "away", "away": "home"})
    variants = [goals.side, pd.Series(np.where(goals.own, flip, goals.side), index=goals.index)]
    final = m.set_index("match_id")[["home_goals", "away_goals"]]
    keep = []
    for mid, gg in goals.groupby("match_id"):
        if mid not in final.index:
            continue
        tgt = final.loc[mid]
        for v in variants:
            s = v.loc[gg.index]
            if (s == "home").sum() == tgt.home_goals and (s == "away").sum() == tgt.away_goals:
                keep.append(gg.assign(side=s))
                break
    kept = pd.concat(keep) if keep else goals.iloc[:0]
    zero = set(final[(final.home_goals == 0) & (final.away_goals == 0)].index)
    bad = set(final.index) - set(kept.match_id) - zero
    print(f"  [tm] goal timelines reconcile with final score for "
          f"{1 - len(bad) / max(len(final), 1):.1%} of matches ({len(bad)} dropped)")
    return kept, bad


def _squad_values(con, g):
    """Sum of starting XI market values on the day before the match. Skipped if tables differ."""
    try:
        lu = con.execute("""select game_id, club_id, player_id, type from game_lineups""").df()
        val = con.execute("""select player_id, date, market_value_in_eur from player_valuations""").df()
    except Exception as e:
        print(f"  [tm] squad values skipped ({e})")
        return pd.DataFrame(columns=["match_id", "home_value", "away_value"])
    lu = lu[lu.type.astype(str).str.contains("start")].merge(
        g[["game_id", "date", "home_club_id", "away_club_id"]], on="game_id")
    lu["date"] = pd.to_datetime(lu.date)
    val["date"] = pd.to_datetime(val.date)
    lu = pd.merge_asof(lu.sort_values("date"), val.sort_values("date"), on="date", by="player_id",
                       direction="backward", allow_exact_matches=False)
    lu["side"] = np.where(lu.club_id == lu.home_club_id, "home_value", "away_value")
    sv = lu.groupby(["game_id", "side"]).market_value_in_eur.sum(min_count=8).unstack()
    sv = sv.reset_index()
    sv["match_id"] = "tm" + sv.game_id.astype(str)
    return sv[["match_id"] + [c for c in ("home_value", "away_value") if c in sv]]


def _h2h(m, last_n=5):
    """Average goal difference (current home team's view) over the previous meetings."""
    out = pd.Series(np.nan, index=m.index)
    hist = {}
    for i, r in m.sort_values("date").iterrows():
        key = frozenset((r.home_id, r.away_id))
        past = hist.get(key, [])
        if past:
            out[i] = np.mean([gd if h == r.home_id else -gd for h, gd in past[-last_n:]])
        hist.setdefault(key, []).append((r.home_id, r.home_goals - r.away_goals))
    return out


# --------------------------------------------------------------------------- Football-Data odds (ASSUMED schema)
ODDS_CLOSING = [("PSCH", "PSCD", "PSCA"), ("AvgCH", "AvgCD", "AvgCA"), ("B365CH", "B365CD", "B365CA")]
ODDS_OPENING = [("PSH", "PSD", "PSA"), ("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A")]


def load_football_data(path=None, which="closing"):
    df = pd.read_parquet(path or PATHS["fd_games"]).reset_index()
    cols = {c.lower(): c for c in df.columns}
    need = {k: cols.get(k) for k in ("league", "date", "home_team", "away_team")}
    if None in need.values():
        raise KeyError(f"football-data columns not as expected: {list(df.columns)[:40]}")
    triples = ODDS_CLOSING + ODDS_OPENING if which == "closing" else ODDS_OPENING
    triples = [t for t in triples if all(c in df for c in t)]
    if which == "closing" and not any(t in ODDS_CLOSING for t in triples):
        print("  [fd] no closing odds columns found, falling back to opening odds")
    P = pd.DataFrame(np.nan, index=df.index, columns=["p_home", "p_draw", "p_away"])
    for t in triples:                               # first available bookmaker per row
        inv = 1 / df[list(t)].astype(float)
        p = inv.div(inv.sum(axis=1), axis=0)        # proportional vig removal
        fill = P.p_home.isna() & p.notna().all(axis=1)
        P.loc[fill] = p[fill].to_numpy()
    out = pd.DataFrame({
        "key": df.index, "league": df[need["league"]].map(LEAGUE_FD),
        "date": pd.to_datetime(df[need["date"]]),
        "home": df[need["home_team"]], "away": df[need["away_team"]],
    }).join(P)
    tcol = cols.get("time")
    if tcol:   # football-data times are believed to be UK local time; verify on a few matches
        local = pd.to_datetime(out.date.dt.strftime("%Y-%m-%d") + " " + df[tcol].astype(str),
                               errors="coerce")
        out["kickoff_utc"] = local.dt.tz_localize("Europe/London", ambiguous="NaT",
                                                  nonexistent="NaT").dt.tz_convert("UTC")
    return out


# --------------------------------------------------------------------------- xG shots (optional)
def load_xg_shots(path=None):
    """Expected columns: league, date, home, away, minute, team, xg (one row per shot).
    'team' must equal the home or away name in that row. Produce this from Understat
    (soccerdata.Understat shot events) and save as parquet."""
    path = path or PATHS["xg_shots"]
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


# --------------------------------------------------------------------------- Kalshi
def _pick(c, name):
    d = pd.to_numeric(c.get(f"{name}_dollars"), errors="coerce") if f"{name}_dollars" in c else None
    k = pd.to_numeric(c[name], errors="coerce") if name in c else None
    if k is not None and k.max() > 1.5:
        k = k / 100
    if d is None:
        return k
    return d if k is None else d.fillna(k)


def load_kalshi(matches, markets_path=None, candles_path=None):
    mk = pd.read_parquet(markets_path or PATHS["kalshi_markets"])
    mk["league"] = mk.series.map(LEAGUE_KALSHI)
    mk = mk[mk.league.notna()].copy()
    dt = mk.event_ticker.str.extract(r"-(\d{2})([A-Z]{3})(\d{2})")
    mk["date"] = pd.to_datetime("20" + dt[0] + dt[1].str.title() + dt[2], format="%Y%b%d", errors="coerce")
    mk["label"] = mk.yes_sub_title.fillna("")
    mk["is_draw"] = mk.label.str.lower().isin(["tie", "draw"])

    ev = []
    for e, g in mk.groupby("event_ticker"):
        teams = g.loc[~g.is_draw, "label"].tolist()
        if len(teams) == 2:
            ev.append(dict(key=e, league=g.league.iloc[0], date=g.date.iloc[0],
                           home=teams[0], away=teams[1]))
    ev = pd.DataFrame(ev)
    right = matches.rename(columns={"match_id": "key"})[["key", "league", "date", "home", "away"]]
    link = match_games(ev, right, allow_swap=True)
    print(f"  [kalshi] matched {len(link)}/{len(ev)} events to matches "
          f"(unmatched -> add rows to {PATHS['aliases']})")
    if len(ev) - len(link):
        print("   e.g.", ev[~ev.key.isin(link)].head(5)[["key", "home", "away"]].to_dict("records"))

    mm = matches.set_index("match_id")
    rows = []
    for e, (mid, _, _) in link.items():
        g = mk[mk.event_ticker == e]
        hn, an = norm(mm.at[mid, "home"]), norm(mm.at[mid, "away"])
        for r in g.itertuples():
            if r.is_draw:
                out = "draw"
            else:
                out = "home" if sim(norm(r.label), hn) >= sim(norm(r.label), an) else "away"
            rows.append(dict(ticker=r.ticker, match_id=mid, outcome=out,
                             won=int(str(r.result).lower() == "yes"),
                             close_time=pd.to_datetime(r.close_time, utc=True)))
    tick = pd.DataFrame(rows)

    c = pd.read_parquet(candles_path or PATHS["kalshi_candles"])
    c = c[c.ticker.isin(tick.ticker)]
    out = pd.DataFrame({
        "ticker": c.ticker.to_numpy(),
        "t": pd.to_datetime(c.end_ts, unit="s", utc=True).to_numpy(),
        "bid": _pick(c, "yes_bid_close").to_numpy(),
        "ask": _pick(c, "yes_ask_close").to_numpy(),
    })
    out = out.merge(tick, on="ticker")
    # sanity: the winning outcome should be the only "won" market per match
    chk = out.groupby("match_id").apply(lambda g: g.drop_duplicates("ticker").won.sum(),
                                        include_groups=False)
    print(f"  [kalshi] matches with exactly one winning market: {(chk == 1).mean():.1%}")
    return out


# --------------------------------------------------------------------------- build all
def build_canonical():
    os.makedirs(PATHS["canonical"], exist_ok=True)
    print("loading Transfermarkt...")
    m, events = load_transfermarkt()
    print(f"  {len(m)} matches, {len(events)} goal/red events")

    print("loading Football-Data odds...")
    from config import ODDS
    fd = load_football_data(which=ODDS)
    link = match_games(fd, m.rename(columns={"match_id": "key"})[["key", "league", "date", "home", "away"]])
    fd["match_id"] = fd.key.map(lambda k: link.get(k, (None,))[0])
    print(f"  odds matched for {fd.match_id.notna().sum()}/{len(fd)} football-data rows")
    keep = ["match_id", "p_home", "p_draw", "p_away"] + (["kickoff_utc"] if "kickoff_utc" in fd else [])
    m = m.merge(fd.dropna(subset=["match_id"])[keep].drop_duplicates("match_id"), on="match_id", how="left")

    xg = load_xg_shots()
    if xg is not None:
        r = m.rename(columns={"match_id": "key"})[["key", "league", "date", "home", "away"]]
        xk = xg.drop_duplicates(["league", "date", "home", "away"]).reset_index(drop=True)
        xk["key"] = xk.index
        lk = match_games(xk, r)
        xk["match_id"] = xk.key.map(lambda k: lk.get(k, (None,))[0])
        xg = xg.merge(xk[["league", "date", "home", "away", "match_id"]], on=["league", "date", "home", "away"])
        xg["side"] = np.where(xg.team == xg.home, "home", "away")
        shots = xg.dropna(subset=["match_id"]).assign(kind="shot")[["match_id", "minute", "side", "kind", "xg"]]
        events = pd.concat([events, shots])
        print(f"  xG shots attached for {shots.match_id.nunique()} matches")

    print("loading Kalshi...")
    candles = load_kalshi(m)
    # kickoff: football-data time if present, else market close minus a fixed duration
    ko = m.set_index("match_id").get("kickoff_utc")
    close = candles.groupby("match_id").close_time.max()
    fallback = close - pd.Timedelta(minutes=TIME["kickoff_from_close"])
    if ko is not None:
        both = pd.concat([ko, fallback], axis=1, keys=["fd", "close"]).dropna()
        if len(both):
            diff = (both["close"] - both["fd"]).dt.total_seconds() / 60
            print(f"  [kickoff] close-based minus football-data kickoff: median {diff.median():.0f} min, "
                  f"IQR {diff.quantile(.25):.0f}..{diff.quantile(.75):.0f}. "
                  f"If this isn't ~0, tune TIME['kickoff_from_close'].")
    kick = (ko if ko is not None else pd.Series(dtype="datetime64[ns, UTC]")).reindex(close.index)
    src = np.where(kick.notna(), "football-data", "close_time")
    kick = kick.fillna(fallback)
    m = m.merge(pd.DataFrame({"match_id": kick.index, "kickoff": kick.to_numpy(), "kickoff_src": src}),
                on="match_id", how="left")
    print(f"  kickoff sources: {pd.Series(src).value_counts().to_dict()}")

    m.to_parquet(f"{PATHS['canonical']}/matches.parquet")
    events.to_parquet(f"{PATHS['canonical']}/events.parquet")
    candles.drop(columns=["close_time"]).to_parquet(f"{PATHS['canonical']}/candles.parquet")
    print("saved canonical tables to", PATHS["canonical"])


def load_canonical(d=None):
    d = d or PATHS["canonical"]
    return (pd.read_parquet(f"{d}/matches.parquet"), pd.read_parquet(f"{d}/events.parquet"),
            pd.read_parquet(f"{d}/candles.parquet"))


# --------------------------------------------------------------------------- match state
def _index_events(events):
    idx = {}
    for mid, g in events.groupby("match_id"):
        d = {}
        for side in ("home", "away"):
            s = g[g.side == side]
            for kind in ("goal", "red"):
                d[(side, kind)] = np.sort(s.loc[s.kind == kind, "minute"].to_numpy(float))
            sh = s[s.kind == "shot"].sort_values("minute")
            d[(side, "xg")] = (sh.minute.to_numpy(float), np.cumsum(sh.xg.to_numpy(float)))
        d["has_xg"] = (g.kind == "shot").any()
        idx[mid] = d
    return idx


def state_at(idx, match_ids, cutoffs):
    """Counts of events with minute <= cutoff, per row. cutoffs are match minutes."""
    match_ids, cutoffs = np.asarray(match_ids), np.asarray(cutoffs, float)
    out = {k: np.zeros(len(cutoffs)) for k in
           ("home_goals", "away_goals", "home_reds", "away_reds")}
    out["home_xg"] = np.full(len(cutoffs), np.nan)
    out["away_xg"] = np.full(len(cutoffs), np.nan)
    order = np.argsort(match_ids, kind="stable")
    ids_sorted = match_ids[order]
    bounds = np.flatnonzero(np.r_[True, ids_sorted[1:] != ids_sorted[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        rows = order[a:b]
        d = idx.get(ids_sorted[a])
        if d is None:
            continue
        q = cutoffs[rows]
        for side in ("home", "away"):
            out[f"{side}_goals"][rows] = np.searchsorted(d[(side, "goal")], q, side="right")
            out[f"{side}_reds"][rows] = np.searchsorted(d[(side, "red")], q, side="right")
            if d["has_xg"]:
                mins, cum = d[(side, "xg")]
                k = np.searchsorted(mins, q, side="right")
                out[f"{side}_xg"][rows] = np.where(k > 0, cum[np.maximum(k - 1, 0)], 0.0)
    return pd.DataFrame(out)


PREMATCH_COLS = ["p_home", "p_draw", "p_away", "home_value", "away_value", "h2h_gd"]


def training_snapshots(matches, events, before, minutes):
    """One row per (match, minute) for matches dated before `before`; only events strictly
    before the snapshot minute are known."""
    m = matches[pd.to_datetime(matches.date) < pd.Timestamp(before)]
    rep = m.loc[m.index.repeat(len(minutes))].reset_index(drop=True)
    rep["minute"] = np.tile(minutes, len(m)).astype(float)
    st = state_at(_index_events(events[events.match_id.isin(m.match_id)]),
                  rep.match_id, rep.minute - 0.5)
    cols = ["match_id", "league", "date", "minute", "result"] + [c for c in PREMATCH_COLS if c in rep]
    return pd.concat([rep[cols], st], axis=1)


def clock_to_match(t, kickoff):
    """Clock time -> (match minute, phase, signed clock minutes since kickoff)."""
    clock = (pd.to_datetime(t, utc=True) - pd.to_datetime(kickoff, utc=True)).dt.total_seconds().to_numpy() / 60
    FH, HT = TIME["first_half_clock"], TIME["halftime_break"]
    minute = np.where(clock < 0, 0.0, np.where(clock <= FH, clock,
                      np.where(clock <= FH + HT, 45.0, 45.0 + (clock - FH - HT))))
    phase = np.select([clock < 0, clock <= FH, clock <= FH + HT, minute <= TIME["max_minute"]],
                      ["pre", "1H", "HT", "2H"], "post")
    return minute, phase, clock


def candle_snapshots(candles, matches, events):
    """Attach match minute, phase and the match state known at each candle (with event lag)."""
    m = matches.set_index("match_id")
    c = candles.join(m[["league", "date", "kickoff"] + [x for x in PREMATCH_COLS if x in m]],
                     on="match_id")
    c = c[c.kickoff.notna()].reset_index(drop=True)
    c["minute"], c["phase"], c["clock"] = clock_to_match(c.t, c.kickoff)
    cutoff = np.where(c.phase == "pre", -1.0, c.minute - TIME["event_lag"])
    st = state_at(_index_events(events[events.match_id.isin(c.match_id.unique())]),
                  c.match_id, cutoff)
    return pd.concat([c, st], axis=1)
