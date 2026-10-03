# Hypothesis (committed before any backtest on real Kalshi data)

Kalshi soccer game-result markets (home / draw / away, one YES contract each) misprice
in-play outcomes relative to a fair-value model built from match state plus independent
pre-match information.

**Counterparty:** hurried or sentiment-driven traders reacting to goals and red cards.

**Fails if:** costs eat the edge, or the market reprices faster than our data can know events
(we assume a 2-minute event lag).

## Fixed before testing
- Model: multinomial logistic regression, `baseline` features, C=1.0
  (`models/baseline_logit_C1.0.pkl`), trained only on matches before the first Kalshi match
  (2025-05-11). The model never sees Kalshi prices.
- Trade rule: buy YES if `p - ask - fee > threshold`, buy NO if `bid - p - fee > threshold`;
  fill at next minute's quote; hold to settlement; one trade per market.
  Starting values: threshold 0.04, max spread 0.06, minutes 0-85, no halftime, no pre-match.
- Fee: 0.07 * P * (1 - P) per contract; every result also shown at 2x costs.
- Split: in-sample = Kalshi matches 2025-05-11 to 2026-04-03 (1,394);
  out-of-sample = 2026-04-04 to 2026-05-24 (364), evaluated once at the end.
- Every backtest variant is logged to `results/trials.csv` and the count is reported.
