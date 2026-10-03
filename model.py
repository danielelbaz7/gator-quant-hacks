"""
P(home win), P(draw), P(away win) from a snapshot of the match state.

kind="logit": multinomial logistic regression. It is linear in the features
              (log-odds = w . x), so it is the "linear regression" for
              probabilities and always outputs valid probabilities.
kind="ols":   plain linear regression on 0/1 outcomes (linear probability model),
              clipped and renormalised. Kept for comparison; usually worse.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import build_X

CLASSES = ["H", "D", "A"]


class WinProbModel:
    def __init__(self, features, kind="logit", C=1.0):
        self.features, self.kind, self.C = list(features), kind, C

    def fit(self, snaps):
        X = build_X(snaps, self.features)
        ok = X.notna().all(axis=1) & snaps["result"].isin(CLASSES)
        if not ok.any():
            empty = X.columns[X.isna().all()].tolist()
            raise ValueError(f"no usable training rows: these features are entirely missing {empty}. "
                             "Their source data isn't in matches/events (e.g. xg needs xg_shots.parquet).")
        dropped = (~ok).mean()
        if dropped > 0.05:
            print(f"  [model] dropped {dropped:.0%} of training rows with missing features "
                  f"(check which inputs are missing for this feature set)")
        X, y = X[ok], snaps.loc[ok, "result"].map({c: i for i, c in enumerate(CLASSES)})
        order = np.lexsort([snaps.loc[ok, "minute"].to_numpy(), snaps.loc[ok, "match_id"].astype(str).to_numpy()]) \
            if {"match_id", "minute"} <= set(snaps.columns) else np.arange(len(X))
        X, y = X.iloc[order], y.iloc[order]        # fixed row order: fit doesn't depend on table order
        if self.kind == "logit":
            self.pipe = make_pipeline(StandardScaler(),
                                      LogisticRegression(C=self.C, max_iter=10000, tol=1e-8))
            self.pipe.fit(X, y)
        elif self.kind == "ols":
            self.pipe = make_pipeline(StandardScaler(), LinearRegression())
            self.pipe.fit(X, np.eye(3)[y])
        else:
            raise ValueError(self.kind)
        self.n_train = len(X)
        return self

    def predict(self, df):
        X = build_X(df, self.features)
        ok = X.notna().all(axis=1).to_numpy()
        P = np.full((len(df), 3), np.nan)
        if ok.any():
            if self.kind == "logit":
                P[ok] = self.pipe.predict_proba(X[ok])
            else:
                raw = np.clip(self.pipe.predict(X[ok]), 1e-3, None)
                P[ok] = raw / raw.sum(axis=1, keepdims=True)
        return pd.DataFrame(P, index=df.index, columns=["pH", "pD", "pA"])

    def coefficients(self):
        est = self.pipe[-1]
        coef = est.coef_ if self.kind == "logit" else est.coef_
        return pd.DataFrame(coef, index=CLASSES, columns=self.features).T.round(3)


def evaluate(model, snaps):
    """Log loss and Brier score, overall and by minute bucket."""
    P = model.predict(snaps)
    ok = P.notna().all(axis=1) & snaps["result"].isin(CLASSES)
    P, s = P[ok].to_numpy(), snaps[ok]
    y = s["result"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    ll = -np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1))
    br = ((P - np.eye(3)[y]) ** 2).sum(axis=1)
    out = pd.DataFrame({"minute": s["minute"].to_numpy(), "logloss": ll, "brier": br})
    out["bucket"] = pd.cut(out.minute, [-1, 0, 30, 60, 75, 90, 200],
                           labels=["pre", "1-30", "31-60", "61-75", "76-90", "90+"])
    by = out.groupby("bucket", observed=True)[["logloss", "brier"]].mean().round(4)
    return dict(logloss=ll.mean(), brier=br.mean(), n=len(y)), by


def calibration(model, snaps, bins=10):
    """Predicted vs realised frequency for each outcome; the core table for the note."""
    P = model.predict(snaps)
    rows = []
    for k, c in enumerate(CLASSES):
        p = P.iloc[:, k]
        hit = (snaps["result"] == c).astype(float)
        ok = p.notna()
        b = pd.cut(p[ok], np.linspace(0, 1, bins + 1))
        g = pd.DataFrame({"p": p[ok], "hit": hit[ok], "bin": b}).groupby("bin", observed=True)
        t = g.agg(pred=("p", "mean"), actual=("hit", "mean"), n=("p", "size")).reset_index()
        t.insert(0, "outcome", c)
        rows.append(t)
    return pd.concat(rows, ignore_index=True).round(3)
