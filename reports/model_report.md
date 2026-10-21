# Model Report — Parallel-Run Reconciler

All data is **synthetic** with injected causes/labels. Every number below was produced by the run that generated this file (`python -m src.run_pipeline`). These results describe the synthetic benchmark only; they do not predict real-world performance.

## Problem A — difference classifier

- Deals generated: 6,000; differing deals in train (months 1–8): 725; test (months 9–12): 345 (split by period — no future leakage).
- Gradient boosting used 66 of 400 configured trees (early stopping, validation fraction 0.15 within the training months).
- Feature importance method: SHAP (TreeExplainer).

| Model | Accuracy | Macro F1 |
|---|---|---|
| rules baseline | 0.988 | 0.988 |
| gradient boosting | 0.980 | 0.979 |
| cascade (rules first, ML second) | 0.980 | 0.979 |

Per-class precision / recall (test differences):

| Cause | Rules P | Rules R | GBM P | GBM R | Support |
|---|---|---|---|---|---|
| rounding | 0.992 | 0.983 | 0.983 | 0.983 | 120 |
| rule | 0.980 | 1.000 | 0.960 | 1.000 | 96 |
| timing | 1.000 | 0.969 | 1.000 | 0.969 | 64 |
| data | 0.985 | 1.000 | 0.984 | 0.954 | 65 |

- Auto-classified at confidence >= 0.9: 97.4% of test differences, accuracy among them 98.8% (pure ML view).
- Top-2 accuracy (true cause in model's top two): 98.3%.
- Dual-cause rows in test: 25; both injected causes surfaced in the top two for 48.0% of them.

## Problem B — mapping-error detector

- Accounts: 160 (20 injected mis-mappings); test split 48 accounts with 6 positives (stratified by label).
- ROC AUC 0.988; PR AUC 0.911; precision@10 0.60, precision@20 0.30 on the ranked review list.
- Simple anomaly rules (category mismatch OR mapping collision OR sign flip) on the same test accounts: precision 0.50, recall 0.83. The flag list is a blunt instrument next to the ranked list: compare its precision with precision@10 above.

## Problem C — feed-health monitor

- 24 months scored; mean health 91.5; months flagged as breaks (score < 75): 2025-07, 2026-01, 2026-07 — these are exactly the three injected breaks (row drop, null spike, late spike).

## Where rules beat ML, and where ML wins

- Rules are exact for `rounding` and for missing records by construction, and every rule decision is auditable in one line.
- ML earns its place on the residual differences: it weighs overlapping signals (e.g. a data change that also looks like a rule change) and returns calibrated top-two candidates, which is what makes dual causes visible instead of forcing one label.
- The cascade is the production shape: rules clear the certain items in bulk, ML ranks the rest, humans decide the low-confidence tail.

## Limitations

- Synthetic data with clean, injected causes; real differences are messier and causes stack in ways this generator only approximates.
- 2% reviewer label noise is simulated on single-cause rows; real labels will be noisier and sometimes contested.
- Problem B is evaluated on one snapshot month with a stratified account split; a real migration would validate per entity and per period.
- Problem C thresholds (75/90) are set by hand, not learned.
- No real pilot has been run; no claim here transfers to production data.
