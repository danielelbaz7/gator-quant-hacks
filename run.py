"""
python run.py build                                   # raw sources -> data/canonical/
python run.py fit      --set baseline                 # train, validation scores, calibration
python run.py backtest --set baseline --threshold 0.05
python run.py grid                                    # sweep config.GRID (in-sample only)
python run.py oos      --set baseline --threshold 0.05   # ONCE, at the very end

Add --data data/demo to any command to use the synthetic demo data (make_demo.py).
"""
import argparse
import itertools
import json
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

import config
from backtest import breakdown, metrics, run_backtest
from data import build_canonical, candle_snapshots, load_canonical, load_live_events, training_snapshots
from model import WinProbModel, calibration, evaluate

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)


def load(data_dir):
    matches, events, candles = load_canonical(data_dir)
    kal_ids = candles.match_id.unique()
    kal_dates = matches.set_index("match_id").loc[kal_ids, "date"].sort_values()
    kalshi_start = kal_dates.min()
    oos_start = kal_dates.iloc[int(len(kal_dates) * (1 - config.SPLIT["oos_frac"]))]
    print(f"Kalshi matches: {len(kal_ids)} | first {kalshi_start.date()} | "
          f"OOS from {oos_start.date()} (most recent {config.SPLIT['oos_frac']:.0%})")
    return matches, events, candles, kalshi_start, oos_start


def fit_model(matches, events, kalshi_start, fset, kind, C, val=True):
    feats = config.FEATURE_SETS[fset]
    snaps = training_snapshots(matches, events, kalshi_start, config.TRAIN_MINUTES)
    vstart = pd.Timestamp(config.SPLIT["val_start"])
    tr, va = snaps[snaps.date < vstart], snaps[snaps.date >= vstart]
    if val and len(va) and len(tr):
        m = WinProbModel(feats, kind, C).fit(tr)
        score, by = evaluate(m, va)
        print(f"\n[{fset} | {kind} C={C}] validation ({va.match_id.nunique()} matches): "
              f"logloss {score['logloss']:.4f}  brier {score['brier']:.4f}")
        print(by.T.to_string())
    # final model for trading: all pre-Kalshi matches
    return WinProbModel(feats, kind, C).fit(snaps), snaps


_CANDLE_CACHE = {}


def kalshi_frame(candles, matches, events, data_dir=None):
    if "c" not in _CANDLE_CACHE:
        print("computing match state for every candle...")
        _CANDLE_CACHE["c"] = candle_snapshots(candles, matches, events, load_live_events(data_dir))
    return _CANDLE_CACHE["c"]


def predict_outcome(model, c):
    P = model.predict(c).to_numpy()
    k = c.outcome.map({"home": 0, "draw": 1, "away": 2}).to_numpy()
    return pd.Series(P[np.arange(len(c)), k], index=c.index)


def bt_params(a):
    p = dict(config.BACKTEST)
    for k in p:
        v = getattr(a, k, None)
        if v is not None:
            p[k] = v
    p["block_prematch"] = config.ODDS == "closing"
    return p


def report(c, prob, params, title):
    tr = run_backtest(c, prob, **params)
    tr2 = run_backtest(c, prob, **{**params, "cost_mult": params["cost_mult"] * 2})
    m1, m2 = metrics(tr, title), metrics(tr2, title + " (costs x2)")
    print(pd.DataFrame([m1, m2]).set_index("label").T.to_string())
    if not tr.empty:
        print("\nby phase:\n", breakdown(tr, "phase").to_string())
        print("\nby league:\n", breakdown(tr, "league").to_string())
        print("\nbuy YES vs buy NO:\n", breakdown(tr.assign(dir=tr.side.map({1: "buy YES", -1: "buy NO"})), "dir").to_string())
    return tr, m1, m2


def log_trial(row):
    os.makedirs(config.PATHS["results"], exist_ok=True)
    f = f"{config.PATHS['results']}/trials.csv"
    pd.DataFrame([row]).to_csv(f, mode="a", header=not os.path.exists(f), index=False)
    n = len(pd.read_csv(f))
    print(f"\n[trial #{n} logged to {f} -- report this count in the note]")


def save_model(model, a, kstart):
    """models/<set>_<kind>_C<C>.pkl (load with pickle; needs model.py) + readable weights/info."""
    os.makedirs("models", exist_ok=True)
    name = f"models/{a.set}_{a.kind}_C{a.C}"
    with open(f"{name}.pkl", "wb") as fh:
        pickle.dump(model, fh)
    w = model.coefficients()
    w.loc["intercept"] = model.pipe[-1].intercept_.round(3)
    w.to_csv(f"{name}_weights.csv")
    with open(f"{name}_info.json", "w") as fh:
        json.dump(dict(saved=datetime.now().isoformat(timespec="seconds"), feature_set=a.set,
                       kind=a.kind, C=a.C, features=model.features, n_train_rows=model.n_train,
                       trained_on_matches_before=str(kstart.date())), fh, indent=1)
    print(f"\nsaved model -> {name}.pkl (+ _weights.csv, _info.json)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "fit", "backtest", "grid", "oos"])
    ap.add_argument("--data", default=config.PATHS["canonical"])
    ap.add_argument("--set", default="baseline")
    ap.add_argument("--kind", default=config.MODEL["kind"])
    ap.add_argument("--C", type=float, default=config.MODEL["C"])
    for k, v in config.BACKTEST.items():
        if isinstance(v, bool):
            continue
        ap.add_argument(f"--{k}", type=type(v), default=None)
    a = ap.parse_args()

    if a.cmd == "build":
        return build_canonical()

    matches, events, candles, kstart, oos_start = load(a.data)

    if a.cmd == "fit":
        model, snaps = fit_model(matches, events, kstart, a.set, a.kind, a.C)
        print("\ncoefficients (standardised features):\n", model.coefficients().to_string())
        vs = snaps[snaps.date >= pd.Timestamp(config.SPLIT["val_start"])]
        if len(vs):
            print("\ncalibration on validation:\n", calibration(model, vs).to_string())
        save_model(model, a, kstart)
        return

    c = kalshi_frame(candles, matches, events, a.data)
    is_c, oos_c = c[c.date < oos_start], c[c.date >= oos_start]

    if a.cmd == "backtest":
        model, _ = fit_model(matches, events, kstart, a.set, a.kind, a.C, val=False)
        params = bt_params(a)
        tr, m1, m2 = report(is_c, predict_outcome(model, is_c), params, f"IN-SAMPLE {a.set}")
        log_trial(dict(time=datetime.now().isoformat(timespec="seconds"), sample="IS",
                       feature_set=a.set, kind=a.kind, C=a.C, **{k: v for k, v in params.items()},
                       **{f"net_{k}": v for k, v in m1.items() if k != "label"},
                       x2_pnl_total=m2.get("pnl_total")))
        os.makedirs(config.PATHS["results"], exist_ok=True)
        tr.to_csv(f"{config.PATHS['results']}/trades_IS_{a.set}.csv", index=False)

    elif a.cmd == "grid":
        keys = list(config.GRID)
        combos = list(itertools.product(*config.GRID.values()))
        print(f"grid: {len(combos)} combinations (each one is a trial you must disclose)")
        rows, models = [], {}
        for combo in combos:
            g = dict(zip(keys, combo))
            mk = (g["feature_set"], g["kind"], g["C"])
            if mk not in models:
                models[mk] = predict_outcome(
                    fit_model(matches, events, kstart, *mk, val=False)[0], is_c)
            params = {**bt_params(a), **{k: v for k, v in g.items() if k in config.BACKTEST}}
            m1 = metrics(run_backtest(is_c, models[mk], **params))
            m2 = metrics(run_backtest(is_c, models[mk], **{**params, "cost_mult": params["cost_mult"] * 2}))
            row = dict(time=datetime.now().isoformat(timespec="seconds"), sample="IS", **g,
                       **{f"net_{k}": v for k, v in m1.items() if k != "label"},
                       x2_pnl_total=m2.get("pnl_total"))
            rows.append(row)
            log_trial(row)
        out = pd.DataFrame(rows)
        print(out[keys + ["net_trades", "net_pnl_total", "net_pnl_per_trade", "net_t_stat", "x2_pnl_total"]].to_string())
        print("\nPick a configuration for a reason you can defend (stable across neighbours), "
              "not just the top row.")

    elif a.cmd == "oos":
        f = f"{config.PATHS['results']}/oos_runs.log"
        if os.path.exists(f):
            print("!!! OOS has already been evaluated:\n" + open(f).read() +
                  "Running it again means it is no longer out-of-sample. Disclose every run.")
        model, _ = fit_model(matches, events, kstart, a.set, a.kind, a.C, val=False)
        params = bt_params(a)
        tr, m1, m2 = report(oos_c, predict_outcome(model, oos_c), params, f"OUT-OF-SAMPLE {a.set}")
        os.makedirs(config.PATHS["results"], exist_ok=True)
        with open(f, "a") as fh:
            fh.write(json.dumps(dict(time=datetime.now().isoformat(timespec="seconds"),
                                     set=a.set, kind=a.kind, C=a.C, params=params,
                                     result=m1), default=str) + "\n")
        tr.to_csv(f"{config.PATHS['results']}/trades_OOS_{a.set}.csv", index=False)


if __name__ == "__main__":
    main()
