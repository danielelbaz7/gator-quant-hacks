# In-play win-probability model vs Kalshi soccer prices

A model estimates P(home win / draw / away win) at every minute of a match from
match state and pre-match information. **It never sees Kalshi prices.** The backtest
compares those probabilities to Kalshi bid/ask quotes and trades when the gap
exceeds fees plus a threshold.

## Files
| file | what it does |
|---|---|
| `config.py` | everything you tune: feature sets, model, trading rules, grid, time alignment |
| `features.py` | feature registry; add a feature with `@feature("name")` |
| `model.py` | multinomial logistic regression (or OLS), validation log loss, calibration table |
| `data.py` | raw sources -> canonical tables; name matching; clock -> match minute; match state |
| `backtest.py` | trading rule, fees, next-bar fills, metrics |
| `run.py` | command line |
| `make_demo.py` | synthetic data so you can run everything before the real data is ready |

## Workflow
```
pip install pandas numpy scikit-learn pyarrow duckdb scipy

python make_demo.py                                   # optional: try it on fake data
python run.py fit --data data/demo --set baseline

python run.py build                                   # real data -> data/canonical/
python run.py fit      --set baseline                 # validation + calibration + coefficients
python run.py backtest --set baseline --threshold 0.05
python run.py grid                                    # sweep config.GRID
python run.py oos      --set baseline --threshold 0.05   # once, at the end
```
`build` needs, in this folder: `transfermarkt-datasets.duckdb`,
`football_data_games.parquet` (soccerdata MatchHistory), `big5_markets.parquet`,
`big5_candles.parquet`, and optionally `xg_shots.parquet` and `team_aliases.csv`.

## Design rules built in
- **No Kalshi data in the model.** Trained only on matches before the first Kalshi match.
- **Validation split** by date (`SPLIT["val_start"]`); compare feature sets on validation log loss.
- **Event lag**: the model only "knows" a goal/red `TIME["event_lag"]` minutes after it happens.
- **Next-bar fills**: signal on bar t, fill at bar t+1 quotes.
- **Closing odds can't be used before kickoff**: pre-match trades are blocked when `ODDS="closing"`.
- **Costs**: Kalshi fee approximated as `0.07 * P * (1-P)` per contract (verify!), every report also shown at 2x costs.
- **OOS lock**: most recent 20% of Kalshi matches. `oos` warns if it has been run before and logs every run.
- **Trial log**: every backtest/grid row is appended to `results/trials.csv`. Report the count.

## Status
Tested end to end on synthetic data (fit, backtest, grid, oos) and the Kalshi linking on a
mock file. **The Transfermarkt and Football-Data adapters were written from documentation
and have not been run on the real files.** Check the printed diagnostics after `build`:
goal-timeline reconciliation rate, odds match rate, Kalshi match rate, one-winner check,
and the kickoff comparison.
