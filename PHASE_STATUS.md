# Phase Status

> For the full implementation walkthrough — build process, per-phase reasoning,
> and annotated file structure — see
> [docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md).

The single source of truth for what actually exists. A phase moves to DONE
only when its code is written, its tests pass, and its acceptance criteria
are met — not when it has been designed.

| Phase | Component | Status |
|-------|-----------|--------|
| 0 | Repo skeleton, config, logging, operational store | **DONE** |
| 1 | Market data ingestion + data quality | **DONE** |
| 2 | Leakage-safe feature engineering | **DONE** |
| 3 | Purged / embargoed validation | **DONE** |
| 4 | Base ML models | **DONE** |
| 5 | Stacking + probability calibration | **DONE** |
| 6 | Decision engine | **DONE** |
| 7 | Event-based backtester | **DONE** |
| 8 | Monitoring + drift + retraining | **DONE** |
| 9 | Daily orchestration + lineage | **DONE** |
| 10 | Experiment tracking + model registry | **DONE** |
| 11 | Dashboard + reporting | **DONE** |
| 12 | Production hardening + CI | NOT STARTED |
| 13 | Final documentation + release | NOT STARTED |

## Phase 0 — what was built

- `config/config.yaml`, `config/universe.yaml` — every tunable in one place,
  content-hashed so each run records the exact settings that produced it
- `src/trademind/config.py` — typed loader with fail-fast validation
- `src/trademind/logging_setup.py` — run-id-tagged structured logging
- `src/trademind/storage/` — SQLite operational store: predictions, outcomes,
  runs, model registry, data-quality issues
- `tests/` — 25 tests, all passing

Two storage engines, deliberately:

- **SQLite** for the operational store. Transactional, row-level, needs
  uniqueness constraints for idempotency. Standard library, zero setup.
- **Parquet + DuckDB** for the analytical lake (market data, features).
  Columnar bulk scan, added in Phase 1.

## Phase 1 — what was built

- `src/trademind/ingestion/base.py` — three explicitly named price series, provider interface
- `corporate_actions.py` — as-traded reconstruction, total-return series, split position math
- `calendar.py` — NSE calendar with an honest weekday fallback
- `data_validator.py` — 15 checks across structure, prices, volume, jumps, corporate actions, staleness
- `yfinance_provider.py` — retrying provider with column normalisation
- `market_data.py` — fetch -> enrich -> validate -> persist, ERROR blocks the write
- `storage/lake.py` — Hive-partitioned Parquet lake
- 59 new tests; 84 total, all passing

### The three price series

`close` is deliberately absent from the schema so no module can grab "the"
close price and get the wrong one.

| Series | Meaning | Used by |
|--------|---------|---------|
| `*_raw` | as-traded that session | backtester execution **only** |
| `*_split` | split-adjusted, no dividends | charting, volume normalisation |
| `adj_close` | total return | features and labels **only** |

`adj_close` is retroactively revised by every new dividend, which is exactly why
filling an order at it would be look-ahead bias.

### yfinance does not serve as-traded prices

Its OHLC is already split-adjusted. As-traded prices are *reconstructed* by
reversing the split adjustment. The reconstruction is exact when the split
history is complete and wrong in proportion to any split the provider missed —
which is what the `ABNORMAL_JUMP` check exists to catch. Disclose this in the
final report.

## Phase 2 — what was built

`src/trademind/features/`: `technical.py`, `returns.py`, `volatility.py`,
`volume.py`, `regime.py`, `labels.py`, `pipeline.py`.
45 features. 26 new tests; 110 total, all passing.

### The label had to change

The roadmap specified `target_return_1d[t] = adj_close[t+1]/adj_close[t] - 1`
*and* execution at the next session. Those contradict: if the fill happens
during t+1, the close-to-close move from t is already partly gone. Training on
it teaches the model to forecast a return the strategy cannot capture.

The contract now is:

```
session t              session t+1          session t+2
close: features        open: ENTRY          open: EXIT
       decision
tradeable_return_1d[t] = adj_open[t+2] / adj_open[t+1] - 1
```

`tradeable_return_1d` is the training target. `research_return_1d` (the
roadmap's close-to-close version) is kept as a diagnostic. On synthetic
random-walk data the two correlate at **0.001** — they are effectively
different targets, which is the measure of how much apparent skill would have
been untradeable.

Cost: labels need data through t+2, so the two newest rows are always
unlabelled. They are retained, not dropped — the newest row is what the live
job predicts on.

### The leakage test

`tests/test_leakage.py` mutates all prices after a cut point and asserts every
feature at or before it is bit-identical. Runs across all 45 features at three
cut points. `test_the_test_can_actually_fail` plants a known-leaky feature and
confirms the detector fires — without it the suite would prove nothing.

### Expanding, never full-sample

`volatility_20.rank(pct=True)` ranks each day against years that had not
happened yet. No `shift(-1)`, no future column, looks entirely normal. Same for
z-scores against `series.mean()`, min-max scaling, and `pd.qcut`. All replaced
with expanding-window equivalents. Cross-sectional ranks group by date, which
is safe because those observations share a timestamp.

## Phase 3 — what was built

`src/trademind/validation/`: `embargo.py`, `purged_split.py`,
`dataset_validator.py`, `walk_forward.py`. 33 new tests; 143 total.

### Purge must be horizon + 1

`config.purge_days >= target_horizon_days` was too lenient and has been
tightened to `>= horizon + 1`. The extra session is the execution offset: a row
at *t* is labelled with `open[t+1]..open[t+1+h]`, so it encodes information
through *t+1+h*. Sizing the purge to the horizon alone leaves exactly one
leaking row at every fold boundary — enough to inflate scores, invisible in the
metrics.

`test_unpurged_split_leaves_leaking_rows_and_purge_removes_them` counts
contaminated rows directly rather than comparing scores. The structural count
is exact; a score comparison across a few folds would be indistinguishable from
noise.

### Splits are keyed on dates, not row positions

`sklearn.TimeSeriesSplit` is unusable here for two independent reasons. It
splits positionally, and this is a panel — consecutive rows are different
symbols on the same day, so a positional cut puts RELIANCE and TCS on the same
date on opposite sides of the boundary. And it leaves no gap at all.

### Purge and embargo are different mechanisms

Purge removes training rows whose *label* reaches forward into validation.
Embargo removes training rows immediately *after* validation, where rolling
features still share most of their window with data the model was scored on.
Walk-forward needs both, because each fold's validation becomes the next fold's
training data.

### The final-test lock is structural, not a promise

`split_development()` is the only supported path to fittable data and cannot
return a locked row. `FinalTestLock` fingerprints the locked window, refuses a
second unlock in the same session, and detects tampering. It is not a security
boundary — it makes the wrong thing require deliberate effort and leave a trace.

The development split also trims rows just before the boundary, because their
labels reach across it. That boundary is crossed once and never re-checked,
which makes it more dangerous than a fold boundary, not less.

## Phase 4 — what was built

`src/trademind/models/`: `base.py`, `direction.py`, `return_model.py`,
`metrics.py`, `oof.py`, `registry.py`. 45 new tests; 188 total.

### Measured result on random-walk data

The null hypothesis, run through the full pipeline. Truth is AUC 0.50.

| model | ROC-AUC | fold std | accuracy | majority | **lift** | BSS |
|---|---|---|---|---|---|---|
| direction_hgb | 0.5157 | 0.0137 | 0.5167 | 0.5240 | **-0.0073** | -0.0024 |
| direction_logistic | 0.5086 | 0.0091 | 0.5078 | 0.5240 | **-0.0161** | -0.0052 |
| baseline_majority | 0.4982 | 0.0000 | 0.5240 | 0.5240 | 0.0000 | -0.0009 |

Both models are **less accurate than a constant predictor**, and both have
negative Brier skill. Return-side IC is 0.027 with a fold standard deviation of
0.055 — indistinguishable from zero. R² is negative.

This is the correct outcome on synthetic random walks and is the reference
point real data must be read against.

### Preprocessing is fitted inside the fold

`StandardScaler().fit(X)` before splitting leaks, and the purged splitter cannot
help — the scaler already read the validation window and the locked final test.
Every model is an sklearn `Pipeline` so `fit()` scopes preprocessing to training
rows. `test_scaler_fitted_outside_the_fold_leaks` demonstrates it numerically.

### Baselines are mandatory

52% of sessions close up, so "52% accuracy" is what a constant achieves. Every
classification report carries `majority_accuracy`, `always_up_accuracy`, and
`skill_vs_majority`. `auc_stderr` accompanies every AUC: at 600 observations it
is ~0.024, so 0.53 and 0.55 are not different numbers.

`flag_suspicious()` runs automatically and warns above AUC 0.58, escalating past
0.65 with a pointer to the leakage suite. Stated in advance so a good number
cannot be quietly reinterpreted as skill later.

### HistGradientBoosting has no native feature importances

Discovered by a failing test. scikit-learn omits `feature_importances_` on HGB
deliberately — the split-count heuristic is biased toward high-cardinality
features. `BaseModel.permutation_importance()` is the fallback; measure it on
validation data, and read correlated features as families since each shares
credit with its twins.

## Phase 5 — what was built

`src/trademind/ensemble/`: `stacking.py`, `calibrator.py`,
`ensemble_metrics.py`, `pipeline.py`. 34 new tests; 222 total.

### Calibration fitted in-sample is circular

Fitting a calibrator on OOF predictions and then measuring calibration on those
same rows produces a near-zero ECE for **any** model. Isotonic regression maps
any monotone input onto the observed frequencies of the rows it was fitted on.
The number describes the calibrator's flexibility, not future performance.

`calibrate_out_of_fold()` walks forward in blocks: block *k* is transformed by a
calibrator fitted on blocks `0..k-1`. Block 0 is dropped — nothing precedes it.
`test_in_sample_calibration_is_circular` shows both numbers side by side.

Two distinct artifacts, routinely collapsed into one:
`calibrate_out_of_fold()` **measures** how well calibration generalises;
`fit_production_calibrator()` is the artifact that **ships**.

### Measured result on random-walk data

| stage | ROC-AUC |
|---|---|
| base logistic | 0.5097 |
| base hgb | 0.5175 |
| **stacked meta-model** | **0.5069** |

Stacking does not beat the best base model. Calibration moves ECE 0.0434 →
0.0405 while making Brier and MCE *worse*. Learned coefficients are all under
0.04 — no base model carries enough signal to earn weight.

The Brier decomposition diagnoses it: reliability 0.00263, resolution 0.00111.
Resolution below reliability means the forecast carries less information than it
has calibration error, and post-hoc calibration cannot manufacture information.
The pipeline says this itself, automatically.

### ECE needs two companions

Bin choice: equal-width bins concentrate an unsharp model into one or two bins
and let near-empty tails swing the number. Quantile bins are the default;
uniform is reported alongside so the choice is visible.

`sharpness` sits next to ECE always. A constant base-rate forecast has ECE near
zero and no value; sharpness distinguishes the two cases.

## Phase 6 — what was built

`src/trademind/decision/`: `schema.py`, `threshold_optimizer.py`,
`signal_generator.py`, `position_sizer.py`, `risk_engine.py`,
`decision_engine.py`, `evaluator.py`. 45 new tests; 267 total.

### The strategy cannot clear its own costs at a 1-day horizon

```
commission 3 + spread 5 + slippage 5 = 13 bps per side
round trip                           = 26 bps = 0.0026
```

A one-day holding period makes every signal a full round trip — no
amortisation. Expected edge from going long the top decile is approximately
`IC x sigma_daily x 1.755`:

| IC | expected edge | needed |
|---|---|---|
| 0.02 | 5.3 bps | 26 bps |
| 0.03 | 7.9 bps | 26 bps |
| 0.05 | 13.2 bps | 26 bps |
| 0.10 | 26.3 bps | 26 bps |

**Breakeven IC is 0.099.** Measured IC in Phase 4 was 0.027 +/- 0.055.

This is structural, not a tuning problem. No threshold makes it work, because
the search is choosing among trades that all lose money on costs. The levers
are a longer holding period (costs amortise over more days of edge), lower
costs, or accepting the system as decision-support rather than a profitable
strategy. **This project takes the third position explicitly.**

`cost_feasibility()` computes and prints this before any optimisation runs.

### The roadmap's min_expected_return was below the cost floor

It specified `0.0005` — 5 bps against a 26 bps round trip. Every trade clearing
that gate loses 21 bps on costs alone. The floor is now *derived*
(`round_trip x margin`), never tuned: it is a fact about the broker, not a free
parameter.

The second gate matters as much as the first. A 0.72 probability on a 3 bps
move is a confident prediction of a losing trade — confidence about direction
says nothing about magnitude.

### Errors preserve the position, never liquidate

Every rejection routes through `preserve_position()`. Liquidating on a data
outage turns an outage into a realised loss plus a round-trip cost; holding
costs nothing but the missed opportunity. Risk gates run *before* signal
generation, so a rejected input never produces a tradeable number.

HOLD on an open position returns a NaN target weight, not 0.0. Returning zero
would liquidate everything on every quiet day.

### Bug found by a failing test

Volatility sizing used `target_volatility / vol`, conflating a portfolio-level
target with a position weight. For a 0.25-vol symbol that yields 0.8, which the
weight cap flattens to the maximum — silently disabling the scaling for every
symbol. Now `max_weight x (reference_volatility / vol)`, with a test asserting
`weight x volatility` stays constant below the cap.

Kelly sizing is deliberately absent: it assumes the edge estimate is correct,
and sizes most aggressively where a noisy edge is most wrong.

## Phase 7 — what was built

`src/trademind/backtesting/`: `config.py`, `costs` (in config), `execution.py`,
`portfolio.py`, `engine.py`, `benchmarks.py`, `report.py`. 42 new tests;
309 total.

### Reconciliation that can actually fail

The usual check is a tautology:

```python
equity = cash + sum(shares * price)
assert equity == cash + sum(shares * price)   # proves nothing
```

Equity is computed twice here, from disjoint state:

- **Path A (balance sheet)** — `cash + Σ(shares × price)`, driven by the cash
  balance
- **Path B (flows)** — `initial + realised + unrealised + dividends − costs`,
  driven by accumulators that never read `cash`

Three tests plant bugs that path A alone cannot see — an uncharged commission,
an unrecorded dividend, phantom shares — and confirm reconciliation catches
each. It runs every session, so an error surfaces with its date attached.

Measured max reconciliation error across a 750-session run: **0.00000000**.

### Session ordering encodes the timing contract

1. Corporate actions → 2. Execute yesterday's orders at *today's open* →
3. Mark to market and reconcile → 4. Decide from today's data, for tomorrow.

Step 4 last is what makes same-session fills mechanically impossible rather
than merely forbidden. `execute_orders` also raises if `decision_date >=
execution_date`.

### Measured result on a weak synthetic signal

| | total return | CAGR | Sharpe | max DD | cost drag | turnover |
|---|---|---|---|---|---|---|
| TradeMind | +16.5% | +5.3% | 0.99 ± 0.58 | −6.7% | **7.9%** | **20.4x** |
| buy & hold | +47.4% | +13.9% | 1.56 ± 0.58 | −9.3% | 0.1% | — |

Costs consumed 7.9% of capital over three years — the strategy needed +24.4%
gross to deliver +16.5% net. Buy-and-hold beat it on return, Sharpe, and cost,
while trading twice.

This is Phase 6's feasibility arithmetic showing up as an equity curve.

### Reporting states its own limits

Sharpe is printed with its standard error (±0.58 over three years), and a
sample under three years triggers an explicit warning that the ratio is
indicative only. Turnover is annualised and reported next to it, because a
Sharpe at 20x turnover is far more sensitive to cost assumptions than the same
Sharpe at 2x. Costs are itemised into commission / spread / slippage so the
question "which assumption is doing the work?" has an answer.

Benchmarks pay identical costs. A cost-free benchmark against a cost-charged
strategy is the easiest way to make a losing strategy look competitive.

## Phase 8 — what was built

`src/trademind/monitoring/`: `feature_drift.py`, `performance_monitor.py`,
`calibration_monitor.py`, `prediction_monitor.py`, `retraining.py`,
`alerts.py`, `model_monitor.py`, `pipeline.py`. 45 new tests; 354 total.

### Naive drift monitoring would fire on 90% of days

45 features, KS test at p<0.05, checked daily:

```
expected false alarms per day        2.2
P(at least one false alarm today)    0.90
expected false alarms per month      47
```

A monitor that cries wolf nine days in ten trains the team to dismiss it — and
then to dismiss the real one. Three responses: **PSI** (effect size) as the
primary signal rather than p-values; **persistence** (3 consecutive breaches)
before reporting; **Benjamini–Hochberg FDR** on the p-values that remain.
Bonferroni would demand p<0.0011 across 45 correlated features and detect
nothing.

### Performance monitoring cannot detect this model's degradation

| window | observations | AUC std err | detectable drop |
|---|---|---|---|
| 7d | 105 | 0.195 | 0.39 |
| 30d | 450 | 0.094 | 0.19 |
| 90d | 1350 | 0.054 | 0.11 |

The model's entire edge is ~0.02 above chance. **Even a 90-day window cannot
detect a 0.11 drop — the model could lose its whole signal and the monitor
would see noise.**

Governance consequence: an unchanged metric is *not* evidence of health, and
the policy says so explicitly rather than treating stability as a green light.
Every metric carries its minimum detectable effect, and `degradation_detected()`
refuses to declare a drop smaller than the MDE.

### Universe size determines whether monitoring is possible

A 90-day calibration window on 3 symbols yields 195 observations — below the
300 needed for a 10-bin ECE. 15 symbols clears it. Found by a failing test;
now documented by one.

### The four governance rules, each with a named test

1. **Drift alone never retrains.** Markets change distribution constantly; a
   model can be healthy on shifted inputs.
2. **A data-quality ERROR blocks retraining regardless of severity elsewhere.**
   When performance collapses *and* ingestion is erroring, the data is the
   likelier cause — and retraining on bad data bakes the corruption in.
3. **Insufficient new observations means wait.** A model fitted on a handful of
   rows differs from its predecessor by noise.
4. **A failed candidate has no path to production.** `PromotionDecision`
   raises `PromotionBlocked` at construction if `promote=True` with a failed
   validation. No override flag exists.

Promotion checks four axes, not one: AUC, calibration regression, prediction
variance (a well-calibrated constant is not an improvement), and evaluation
sample size.

### Diagnosis, not just detection

- performance down + drift → likely regime change; retraining should help
- performance down, no drift → relationship change; **retraining may not help,
  investigate the feature set**
- drift, performance stable → the world moved and the model is coping

## Phase 9 — what was built

`src/trademind/orchestration/`: `base.py`, `ingestion_job.py`,
`outcome_job.py`, `prediction_job.py`, `monitoring_job.py`, `scheduler.py`,
`daily_pipeline.py`. 29 new tests; 383 total.

### The backfill contamination trap

The obvious way to populate prediction history:

```python
model = registry.load(production_version)   # trained through last month
for day in historical_dates:                # going back two years
    store.save(predict(model, features_on(day)))
```

Every row is a "prediction" made by a model trained on data from *after* the
date it predicts. Monitoring then reports excellent historical performance and
the Phase 8 baseline is fiction. Nothing raises — the dates are real, the
features are real, the outcomes are real, only the counterfactual is impossible.

Guard, enforced on every write:

```
training_end < prediction_date
```

`assert_temporally_valid()` raises otherwise. An honest backfill must therefore
walk forward, replaying out-of-fold predictions from the purged splitter.

### Job ordering is load-bearing

- **Outcomes before predictions.** Yesterday's outcome is known today; resolving
  first means monitoring sees the freshest evidence and the pending queue does
  not grow by one day every day.
- **Monitoring after predictions**, so today's features are in the current
  window when drift is computed.
- **`retrain` mode omits prediction entirely** — predicting with a model about
  to be replaced attributes rows to a version that may not survive.

### Failure recovery

Jobs return `JobResult` rather than raising, so one failure cannot leave a run
stuck at RUNNING. PARTIAL is not fatal — one delisted ticker must not stop the
system — but a FAILED critical job halts the pipeline. `reap_stale_runs()` marks
abandoned RUNNING rows FAILED on startup, since a killed process otherwise
leaves a row that makes "is anything executing?" unanswerable and lets the
scheduler stack overlapping runs.

### Phase 8's operational gap is closed

Real production feature distributions now reach the drift monitor. The reference
window is the model's **training** distribution, not all history — comparing
against all history would flag every genuine long-run trend as drift.

### No scheduler daemon

cron and systemd are already supervised, already log, and already survive a
reboot. The only part worth doing in Python is `should_run_today()`, because
cron cannot know the NSE holiday calendar. Default run time is 18:00 IST —
running before the 15:30 close would compute features from a partial bar, where
the close is wrong and nothing downstream can tell.

## Phase 10 — what was built

`src/trademind/experiments/`: `run.py`, `tracker.py`, `registry.py`,
`comparison.py`, `artifacts.py`. Two new tables (`experiments`,
`registry_transitions`). 37 new tests; 420 total.

### An experiment tracker is a machine for overfitting the validation set

Picking the best of N experiments by validation metric inflates that metric.
With AUC stderr 0.024 and a true AUC of 0.52:

| runs | E[best observed] | inflation |
|---|---|---|
| 1 | 0.5200 | +0.0000 |
| 10 | 0.5520 | +0.0320 |
| 50 | 0.5695 | +0.0495 |
| 100 | 0.5759 | +0.0559 |

**Fifty experiments on a worthless model produce a best observed AUC of 0.57**
— inside the range Phase 4's `flag_suspicious` warns about. The winning run
looks no different from a genuine discovery.

Not a reason to run fewer experiments; a reason to record how many were run.
`n_prior_experiments` is stamped on every logged run automatically, and the
leaderboard carries a `selection_adjusted` column. The only clean answer
remains the locked final test, used once.

### Comparison refuses to declare a winner the data cannot support

`is_distinguishable()` uses the standard error of a *difference* — sqrt(2)
larger than the standard error of either estimate, a factor that is easy to omit
and makes gaps look significant. When several experiments tie, the renderer says
so and advises the simpler model.

### Registry lifecycle is enforced, not documented

`CANDIDATE → VALIDATING → VALIDATED → STAGING → PRODUCTION → ARCHIVED`, with
`FAILED` terminal. Illegal transitions raise. A candidate cannot jump to
production, and there is no path out of FAILED — Phase 8's rule now has two
independent guards, deliberately, because it is the one most likely to be worked
around under pressure.

One PRODUCTION model per task: promoting archives the incumbent in the same
operation. Two live models mean ambiguous prediction lineage and a monitoring
layer silently averaging both.

Every transition appends to `registry_transitions` with timestamp, actor, and
reason. Append-only.

### Reproducibility is checked, not assumed

`reproducibility_gaps()` lists what is missing: no commit, dirty working tree,
no config hash, no dataset version, no seed. A dirty tree is a gap because the
recorded commit then points at code that never ran.

Artifacts are CSV and JSON, not pickle — an artifact you cannot open in five
years without the exact library version is a liability.

## Phase 11 — what was built

`src/trademind/reporting/` (`formatting.py`, `panels.py`, `loaders.py`) plus a
thin Streamlit shell in `dashboard/`. 37 new tests; 457 total.

### The reasoning is separated from the rendering

Every judgement — what counts as a concern, how a number should be read, what
order things appear in — lives in `panels.py` as pure functions and is
unit-tested. Streamlit only draws.

Presentation logic buried in Streamlit callbacks cannot be tested, and
untestable presentation logic is exactly where inconvenient findings quietly
stop being displayed.

### Panel order is a design decision

`build_overview` returns: **cost feasibility → model skill → accounting
integrity → data quality → performance.** The equity curve is last, and a test
enforces it. A dashboard opening with a rising curve invites the viewer to stop
reading.

Rendered with this project's real measured numbers, the overview opens:

```
✗ Cost feasibility: Measured IC 0.027 is below the 0.099 breakeven. At a
  1-session holding period the strategy cannot clear its own costs at any
  threshold.

! Model skill: AUC 0.5157 is within sampling error of chance (± 0.0137).
  No demonstrated discrimination.
  Lift vs majority  -0.0073
    Negative means a constant predictor does better.
```

### Nothing is displayed without what qualifies it

`Metric` carries stderr and baseline; `within_noise` uses the difference
standard error (√2 larger). Tests assert that an AUC within noise of chance
renders as CONCERN not skill, that negative lift renders as BAD, and that an
AUC above 0.58 is flagged as suspected leakage rather than a result.

### Rejections are shown, not filtered

A `BELOW_COST_FLOOR` rejection means the model was confident and the trade was
still not worth making — the system working. A page listing only BUYs would
make that invisible.

### Graceful degradation

Every loader returns empty rather than raising, so the dashboard works before
anything has been computed and shows "not yet available" with the command to
run. A dashboard that crashes during setup is useless exactly when it would
help most.

## Phase 12 — what was built

`.github/workflows/{tests,lint,build}.yml`, `Dockerfile`,
`docker-compose.yml`, `scripts/check_forbidden_patterns.py`,
`src/trademind/provenance.py`, `tests/test_protected_invariants.py`.
40 new tests; 497 total.

### Every protected invariant is paired with a mutation test

All eight roadmap invariants have a test asserting the property **and** a test
that deliberately breaks it and confirms the guard fires. An invariant test
that cannot fail proves nothing, and a suite of always-green tests is worse
than none — it manufactures confidence.

| # | Invariant | Mutation |
|---|---|---|
| 1 | Final test unreachable for fitting | Tamper a locked row → fingerprint catches it |
| 2 | No future info in features | Plant `rank(pct=True)` → detector fires |
| 3 | No same-day execution | Model predicting its own training window → refused |
| 4 | Split preserves wealth | Break the transform → reconciliation raises |
| 5 | Equity paths agree | Uncharged fee / phantom shares / unrecorded dividend |
| 6 | HOLD carries position | Three positive assertions across layers |
| 7 | SELL exits, never shorts | Sell unheld shares → refused |
| 8 | Failed candidate not promotable | Force it → raises; registry blocks independently |

Two meta-tests assert the suite stays complete, so an invariant cannot be
quietly dropped.

### An honest limitation, now asserted

`test_invariant_4c_reconciliation_cannot_detect_a_missing_split`: if a split
occurs and the provider never reports it, both equity paths use the same
unadjusted share count and still agree. Reconciliation is blind to it by
construction. Detection lives upstream in the Phase 1 `ABNORMAL_JUMP` check.
Documented as a test rather than a comment, so it stays true.

### The static gate parses instead of grepping

`scripts/check_forbidden_patterns.py` walks the AST for `rank(pct=True)`
outside a groupby, `bfill`, and `rolling(center=True)`.

The grep version false-positived on every docstring that *warns* about these
patterns — and this codebase documents each trap beside its guard. A gate that
fires on its own documentation is a gate someone disables. Verified against
planted violations of all three rules, and against a grouped rank it must
permit.

One bug found by its own test: the checker crashed on any path outside the
repo root.

### CI is layered by speed

`protected` runs first and alone — seconds, and it gates everything. The full
matrix follows. A `determinism` job runs the protected suite twice and diffs
the output, catching tests that pass only through iteration order.

`PYTHONHASHSEED=42` is set in the workflow env and the Dockerfile, because it
is read at interpreter startup and cannot be set from inside a running process.

The Docker image is pinned to `python:3.12.7-slim-bookworm`, not `3.12`. A
backtest that changes after a base-image refresh is not reproducible.

## Phase 13 — what was built

`README.md`, `reports/final/TradeMind_AI_Final_Report.md` (17 sections),
`LICENSE`, `scripts/final_evaluation.py`. 5 new tests; 499 total.

### The final test remains locked, and the Results section is empty

**No final-test result exists, so none is reported.**

The methodology is complete and the evaluation is one command. It has not been
run because no real data has been ingested — the sandbox this was built in has
no network access.

Writing a plausible number there would reproduce exactly the failure this
project was built to prevent. The original planning document recorded a "locked
final-test result" of +0.48% return at Sharpe 1.45 and a −0.31% drawdown,
described as "immutable benchmark results", for code that had never been
written. Those numbers had acquired the grammar of findings without ever having
been measured.

`test_no_fabricated_final_test_result_exists` enforces it: if no evidence file
is present, the report must still say `LOCKED. Not evaluated.`, and the
discarded figures may appear only within three lines of a disclaimer.

### The unlock is single-use and guarded

`scripts/final_evaluation.py` refuses to run without `--confirm`, refuses to
run twice (an evidence file blocks it), refuses to run from a dirty working
tree, and refuses to run before thresholds are derived and locked. Both guards
verified.

### A prediction recorded before the unlock

Given a development IC of 0.027 against a breakeven of 0.099, the expected
outcome is that the strategy underperforms buy-and-hold and fails to clear
costs. Recording this in advance is itself a control — it makes a surprisingly
good result something to investigate rather than celebrate.

## Open decisions

- Thresholds in `config/config.yaml` are `null` on purpose. They get fit on
  development/OOF data in Phase 6, then locked before the final test is
  touched even once.
- Sentiment/news modelling is deferred to Future Work. The original design had
  it as a hand-weighted ensemble input; the learned meta-model in Phase 5
  replaces that, and adding an NLP pipeline before the numeric path works
  would be premature.

## Known limitations to disclose in the final report

- **Survivorship bias.** The universe is today's large-caps, so it implicitly
  excludes names that were liquid in 2015 and have since delisted or shrunk.
  Backtest results are optimistic by an unmeasured amount.
- **Free-tier data quality.** yfinance split and dividend records are
  imperfect. Phase 1 reconciles what it can and logs the rest.
- **Single market, daily bars.** No intraday, no shorting, no leverage in V1.
- **Reconstructed execution prices.** As-traded prices are derived from a free
  provider's split history rather than sourced directly. A missed split
  corrupts the reconstruction for all prior dates.
- **Approximate calendar without `pandas-market-calendars`.** The weekday
  fallback flags Indian festival holidays as missing sessions, so those
  findings are downgraded to INFO. Install the package before trusting them.

## Discarded

The Phase 1–13 roadmap document listed Phases 1–8 as COMPLETE and recorded a
"locked final-test result" of +0.48% return at Sharpe 1.45. No such code or
backtest existed. Those numbers were placeholders and are not reproducible
results. They must not appear in the README, the final report, or any
description of this project.
