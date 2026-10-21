# Parallel-Run Reconciler

![Python](https://img.shields.io/badge/python-3.12-blue)
![Tests](https://img.shields.io/badge/tests-32%20passing-brightgreen)
![Data](https://img.shields.io/badge/data-fully%20synthetic-orange)
![License](https://img.shields.io/badge/license-MIT-green)

**Classify every difference between an old and a new system's results during a migration parallel run — by likely cause, ranked for investigation, with a plain-language reason for each one.**

Built as a portfolio project for a finance systems & FP&A role: Qobra parallel commission runs, Business Central → NetSuite migration tie-outs (chart-of-accounts mapping, opening balances, UAT), and the feed monitoring that keeps both honest. **All data is synthetic. Nothing here reflects real company data, and the results below describe the synthetic benchmark only — they do not predict real-world performance.**

## The problem

During a system cutover, the same deals are calculated in the old and the new system, and *every* difference must be explained before go-live. Done by hand, that is days of spreadsheet archaeology. This tool takes deal-level old/new results and, for each difference, suggests a cause:

| Cause | Meaning |
|---|---|
| `rounding` | Cent-level difference between the two calculations |
| `rule` | The new plan logic pays a different rate |
| `timing` | Deal crosses a period boundary; commission lands in the next period |
| `data` | Deal amount differs — or the deal is missing from the new system entirely |

A person still reviews every item. The tool ranks and explains; it never posts a change.

## Approach — rules first, ML second

1. **Generate** synthetic deals, ledger accounts and feed statistics with injected causes (`src/generate_data.py`, `src/generate_ledger.py`, `src/feed_health.py`). ~5% of differing deals carry **two overlapping causes** — the worst case, kept visible via top-two predictions.
2. **Rules baseline** (`src/rules.py`): ordered checks a reviewer could apply by hand — exact match, missing record, period boundary, rounding tolerance (configurable), amount change.
3. **Gradient boosting** (`src/models.py`): shallow trees (depth 2) with early stopping on difference features, SHAP explanations. Deliberately small — small tabular data, and trust/traceability matter more than squeezing the benchmark.
4. **Cascade** (`src/review.py`): the production shape. Rules clear the certain items in bulk; a deal missing from the new system is flagged by rule as a *data* cause (a feed gap, not an error); ML classifies the rest; anything under the confidence threshold goes to the top of the review list as `needs_review`, with the model's top-two candidate causes and their probabilities.
5. **Split by period**: train on months 1–8, test on months 9–12 — no future leakage.

## Results (synthetic data, seed 42 — regenerate with `python -m src.run_pipeline`)

Test set: **345 differing deals** (months 9–12); training: 725 differing deals (months 1–8), out of 6,000 deals.

| Model | Accuracy | Macro F1 |
|---|---|---|
| Rules baseline | **0.988** | **0.988** |
| Gradient boosting (66 of 400 trees after early stopping) | 0.980 | 0.979 |
| Cascade (rules first, ML second) | 0.980 | 0.979 |

Per-class precision / recall on test differences:

| Cause | Rules P | Rules R | GBM P | GBM R |
|---|---|---|---|---|
| rounding | 0.992 | 0.983 | 0.983 | 0.983 |
| rule | 0.980 | 1.000 | 0.960 | 1.000 |
| timing | 1.000 | 0.969 | 1.000 | 0.969 |
| data | 0.985 | 1.000 | 0.984 | 0.954 |

**Read this honestly:** on this clean synthetic benchmark, the auditable rules *beat* the model on headline accuracy — the injected causes are cleanly separable, and the rules encode exactly those boundaries. That is the expected result, and it is why the design is rules-first. What ML measurably adds:

- **Confidence-ranked triage.** At a 0.90 threshold, ML auto-classifies 97.4% of differences at 98.8% accuracy; the cascade leaves only **6 of 345** items for a human decision (133 cleared by rules, 206 auto-classified, 6 `needs_review` — ranked first in the review list).
- **Top-two candidates.** The true cause is in the model's top two for 98.3% of differences. For the 25 dual-cause test deals, both injected causes appear in the top two in 48.0% of cases — the rest is the honest hard core: a cent-level rounding error stacked on a rate change is barely visible in the features, and the review list shows the second candidate with its probability instead of hiding it.
- **SHAP explanations** per feature (`reports/shap_importance.png`; top signals: `new_over_old`, `amount_ratio`, `is_cent_level`).

Test-period gap by predicted cause (waterfall, `reports/waterfall.png`): timing −$45,197 · rule −$15,999 · data −$13,277 · rounding +$0.11 · **total −$74,473**.

### Problem B — mapping-error detector

160 synthetic accounts across 4 entities, 20 injected mis-mappings (swaps — half between similar accounts in the same category — and collisions). A gradient-boosting detector over anomaly features (balance ratio, z-score vs the account's own history, category match, name similarity, collision count) produces a **ranked review list**: on the held-out 48 accounts (6 positives), ROC AUC **0.988**, PR AUC **0.911**, and **all 6 mis-mappings appear in the top 10** (precision@10 = 0.60, the maximum possible with 6 positives; recall@10 = 1.00). A naive flag list (category mismatch OR collision OR sign flip) gets precision 0.50 / recall 0.83 on the same accounts. Migration renames (~30% of accounts) are simulated so name similarity can't act as a perfect oracle.

### Problem C — feed-health monitor

24 synthetic months scored 0–100 from row-count completeness, null rate and late arrivals (mean 91.5). The monitor flags exactly the three injected breaks — **2025-07** (row drop), **2026-01** (null spike), **2026-07** (late-arrival spike) — and no others. Thresholds (warn 90, break 75) are set by hand, not learned.

Full detail: [`reports/model_report.md`](reports/model_report.md) (regenerated every run) and `reports/metrics.json`.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.run_pipeline     # generates data, trains, writes reports/ + review lists
pytest -q                      # 32 tests
streamlit run app.py           # the review sheet: ranked list + plain-language reasons
```

Notebooks (executed, with outputs): `notebooks/01_eda.ipynb`, `02_features.ipynb`, `03_model.ipynb`.

## Project structure

```
app.py                     Streamlit review sheet (ranked investigation list)
notebooks/                 01_eda, 02_features, 03_model — executed
src/
  generate_data.py         Problem A: synthetic deals with injected causes (+ dual causes)
  generate_ledger.py       Problem B: synthetic chart of accounts + mappings
  feed_health.py           Problem C: monthly feed-health scoring
  features.py              difference features (A) + anomaly features (B)
  rules.py                 rules-first baseline (configurable rounding tolerance)
  models.py                gradient boosting with early stopping; SHAP importance
  review.py                cascade, plain-language reasons, ranked review lists
  evaluate.py              per-class metrics, confidence coverage, top-2 / dual-cause, binary ranking metrics
  run_pipeline.py          end-to-end run: data -> models -> charts -> reports
tests/                     32 pytest tests (generation, rules, features, models, review, feed health, pipeline, app)
data/synthetic/            generated CSVs (committed: fully synthetic)
reports/                   metrics.json, model_report.md, review lists, charts
.github/workflows/ci.yml   CI: pipeline + tests on push/PR (Python 3.11/3.12)
```

## Limitations

- Synthetic data with clean, injected causes; real differences are messier and stack in ways this generator only approximates (the dual-cause slice is a first pass at that).
- 2% reviewer label noise is simulated on single-cause rows; real labels will be noisier and sometimes contested.
- Problem B is evaluated on one snapshot month with a stratified account split; a real migration would validate per entity and per period. Real charts contain near-duplicate accounts that would degrade the detector.
- Problem C thresholds are hand-set heuristics, not learned or calibrated.
- CI is configured (`.github/workflows/ci.yml`) but has not been executed on GitHub yet — the suite passes locally (32/32) on Python 3.12.
- No real pilot has been run. The honest next step is a small, approved sample of reviewer-labeled real differences; the pipeline is built so that pilot only has to swap the generators for loaders.

## Author

Bhupesh Chandra Dimri — Financial Analyst (FP&A / finance systems). Project brief, code and results are my own; all data synthetic.
