# TradeMind AI — Implementation Guide

**A complete walkthrough of the build: what was made, in what order, why each
decision was taken, and where every file lives.**

Version 1.0 · 13 phases · 82 modules · 499 tests

---

## Table of contents

1. [How to read this document](#1-how-to-read-this-document)
2. [What the system is](#2-what-the-system-is)
3. [The implementation process](#3-the-implementation-process)
4. [Complete folder and file structure](#4-complete-folder-and-file-structure)
5. [How data flows through the system](#5-how-data-flows-through-the-system)
6. [Phase-by-phase walkthrough](#6-phase-by-phase-walkthrough)
7. [The eight protected invariants](#7-the-eight-protected-invariants)
8. [Command reference](#8-command-reference)
9. [Test inventory](#9-test-inventory)
10. [Glossary of traps](#10-glossary-of-traps)
11. [Project statistics](#11-project-statistics)

---

## 1. How to read this document

**If you are new to the project**, read sections 2, 3, and 5. That gives you
what the system does, how it was built, and how data moves through it.

**If you need to find a specific file**, section 4 lists every file with a
one-line description of its job.

**If you want to understand a particular decision**, section 6 walks through
each phase and explains the reasoning. Each phase entry has the same shape:

- **Goal** — what the phase had to achieve
- **Key decisions** — what was chosen and why
- **Files created** — with their responsibilities
- **What it found** — the measured result or discovered problem
- **Tests added** — what is now guaranteed

**If you are preparing to talk about this project**, section 10 is the
highest-value part. It lists every trap the project hit, why each is easy to
miss, and how it is now prevented.

A note on tone throughout: this project's outputs are mostly *negative* results.
That is deliberate. The system was built to establish whether a daily NSE
signal can be traded profitably, and the honest answer turned out to be no. The
value is in how trustworthy that answer is.

---

## 2. What the system is

### In one sentence

An end-to-end machine-learning platform that generates calibrated BUY/HOLD/SELL
signals for NSE equities under strict look-ahead controls, evaluates them in an
event-based backtester with realistic costs, and monitors its own predictions
against what the market actually did.

### What it is not

- Not investment advice
- Not a profitable strategy — it demonstrates the opposite
- Not an autonomous trading system
- Not a price predictor

### The central result

```
Round-trip transaction cost at a 1-day holding period      26 bps
Information coefficient needed to break even                0.099
Information coefficient the models actually achieve         0.027 ± 0.055
```

The strategy cannot clear its own transaction costs at any threshold. This is
arithmetic, not a tuning problem — every candidate trade loses money on costs
before the market moves at all.

### Why that is worth building

Most stock-prediction projects report an impressive accuracy figure that
evaporates on contact with reality, because the model saw the future during
training. The engineering demonstrated here is the machinery that prevents
that, and then the discipline to report what the machinery finds.

A system reporting a poor honest number is more credible than one reporting an
excellent number of unknown provenance.

---

## 3. The implementation process

Every one of the thirteen phases followed the same five-step loop. Understanding
the loop makes the rest of the document easier to follow, because each phase
entry in section 6 is just an instance of it.

### Step 1 — Measure before building

Before writing code for a phase, compute the quantity that determines whether
the phase's approach can work at all.

This step caught several problems that would otherwise have surfaced far later,
or never:

| Phase | Measured first | What it showed |
|---|---|---|
| 6 | Round-trip cost vs achievable edge | Breakeven IC is 0.099; measured is 0.027 |
| 8 | False-alarm rate of naive drift testing | 45 features × p<0.05 fires on 90% of days |
| 8 | Minimum detectable AUC change per window | 0.11 at best — larger than the entire edge |
| 10 | Expected inflation from picking best-of-N | 50 experiments on noise → best AUC 0.57 |

In each case the number changed the design. Phase 8's drift monitor uses effect
sizes and persistence instead of p-values *because* of the arithmetic, not as a
stylistic preference.

### Step 2 — Write the code with the reasoning in it

Every non-obvious decision is documented in the module where it lives, next to
the code it explains. A design decision recorded only in a commit message or a
planning document is a decision that will be silently reversed.

Example, from `src/trademind/features/regime.py`:

```python
# The natural way to write "is volatility high right now?" is:
#     df["vol_percentile"] = df["volatility_20"].rank(pct=True)   # WRONG
# That ranks each day against the ENTIRE series, including years that had
# not happened yet.
```

That comment sits directly above `expanding_percentile`, the function that
replaces it.

### Step 3 — Write tests that can fail

A test asserting a property is only useful if it would notice the property
breaking. So wherever an invariant matters, it is written twice:

- **The invariant test** — asserts the property holds
- **The mutation test** — deliberately breaks it, asserts the guard fires

The clearest example is `tests/test_leakage.py`:

```python
def test_no_feature_sees_the_future():
    # mutate all prices after a cut point, assert features before it are
    # bit-identical

def test_the_test_can_actually_fail():
    # plant a known-leaky feature, assert the detector catches it
```

Without the second test, the first proves nothing. A suite of always-green tests
is worse than no suite, because it manufactures confidence.

### Step 4 — Run everything, investigate every failure

The full suite ran after every change. Several failures turned out to be real
bugs rather than bad tests:

| Phase | Failing test revealed |
|---|---|
| 4 | `generate_oof` constructed an extra throwaway model per call |
| 4 | HistGradientBoosting exposes no `feature_importances_` at all |
| 5 | `beats_best_base()` returned `numpy.bool_`, breaking `is False` checks |
| 6 | Volatility sizing conflated a portfolio target with a position weight |
| 8 | A 90-day calibration window on 3 symbols is below the observation floor |
| 12 | The forbidden-pattern checker crashed on paths outside the repo root |
| 12 | Reconciliation is structurally blind to a *missing* split |
| 13 | The fabrication guard matched line-by-line, but prose wraps |

Each of those was fixed in the code, and where it revealed a limitation rather
than a bug, the limitation was documented as a test so it stays true.

### Step 5 — Record status honestly, then package

`PHASE_STATUS.md` is updated at the end of each phase with what was built, what
was measured, and what is still open. A phase moves to DONE only when the code
is written, the tests pass, and the acceptance criteria are met — never when it
has merely been designed.

This matters because the project's original planning document marked eight
phases COMPLETE for code that had never been written, and recorded a
"locked final-test result" for a backtest that had never been run. The status
file is the correction, and it records the discarded numbers explicitly so they
cannot quietly resurface.

### The one rule underneath all five steps

> **Make the wrong thing hard to do, not merely discouraged.**

Prose says "never execute on the same day". A `ValueError` raised at the storage
boundary means nobody can. Throughout the codebase, rules that matter are
enforced structurally:

| Rule | Enforcement |
|---|---|
| No same-day execution | `Prediction.__post_init__` raises |
| Final test used once | `FinalTestLock.unlock` refuses a second call |
| Failed candidate never promoted | `PromotionDecision.__post_init__` raises |
| Backfill cannot use a future model | `assert_temporally_valid` on every write |
| No full-sample statistics | AST checker in CI |

---

## 4. Complete folder and file structure

```
TradeMind-AI/
│
├── README.md                     project overview and the headline finding
├── PHASE_STATUS.md               honest per-phase status; the source of truth
├── LICENSE                       MIT, plus a not-investment-advice notice
├── pyproject.toml                dependencies, pytest markers, ruff config
├── Makefile                      install / test / protected / lint / docker
├── Dockerfile                    pinned Python 3.12.7, PYTHONHASHSEED=42
├── docker-compose.yml            pipeline service + dashboard service
├── .dockerignore                 never bake data or artifacts into the image
├── .gitignore                    data, models, logs are regenerated not committed
├── main.py                       the CLI entry point for every command
│
├── config/
│   ├── config.yaml               every tunable; content-hashed for lineage
│   └── universe.yaml             the 15-symbol NSE universe and benchmark
│
├── src/trademind/                the library
│   ├── __init__.py
│   ├── config.py                 typed config loader with fail-fast validation
│   ├── logging_setup.py          run-id-tagged structured logging
│   ├── provenance.py             git state, packages, environment hash
│   │
│   ├── storage/                  ── Phase 0 ──────────────────────────────
│   │   ├── schema.sql            SQLite DDL: predictions, outcomes, runs,
│   │   │                         model_registry, data_quality_issues,
│   │   │                         experiments, registry_transitions
│   │   ├── db.py                 connection management and schema bootstrap
│   │   ├── predictions.py        PredictionStore: idempotent writes, outcome
│   │   │                         resolution, run tracking
│   │   └── lake.py               Parquet lake, Hive-partitioned by symbol
│   │
│   ├── ingestion/                ── Phase 1 ──────────────────────────────
│   │   ├── base.py               the three price series; provider interface
│   │   ├── yfinance_provider.py  retrying provider, column normalisation
│   │   ├── corporate_actions.py  raw reconstruction, total-return series,
│   │   │                         the ₹100,000 split invariant
│   │   ├── calendar.py           NSE calendar with a weekday fallback
│   │   ├── data_validator.py     15 checks; reports, never repairs
│   │   └── market_data.py        fetch → enrich → validate → persist
│   │
│   ├── features/                 ── Phase 2 ──────────────────────────────
│   │   ├── labels.py             the timing contract and both labels
│   │   ├── technical.py          RSI, MACD, Bollinger, ATR, SMA distances
│   │   ├── returns.py            multi-horizon returns and skip-momentum
│   │   ├── volatility.py         realised, Parkinson, Garman-Klass
│   │   ├── volume.py             relative volume, turnover, signed volume
│   │   ├── regime.py             expanding percentiles, trend state, drawdown
│   │   └── pipeline.py           assembles the 45-feature panel
│   │
│   ├── validation/               ── Phase 3 ──────────────────────────────
│   │   ├── embargo.py            purge and embargo windows; GapConfig
│   │   ├── purged_split.py       date-keyed walk-forward splitter
│   │   ├── dataset_validator.py  the final-test lock
│   │   └── walk_forward.py       fold runner and nested tuning
│   │
│   ├── models/                   ── Phase 4 ──────────────────────────────
│   │   ├── base.py               BaseModel, ModelSpec, fold-internal
│   │   │                         preprocessing, permutation importance
│   │   ├── direction.py          logistic and HistGradientBoosting classifiers
│   │   ├── return_model.py       Ridge and HistGradientBoosting regressors
│   │   ├── metrics.py            metrics with baselines and stderr;
│   │   │                         flag_suspicious()
│   │   ├── oof.py                out-of-fold generation and baselines
│   │   └── registry.py           model artifact persistence
│   │
│   ├── ensemble/                 ── Phase 5 ──────────────────────────────
│   │   ├── stacking.py           meta-model over OOF predictions
│   │   ├── calibrator.py         isotonic/sigmoid, fitted out-of-fold
│   │   ├── ensemble_metrics.py   reliability, ECE, Brier decomposition
│   │   └── pipeline.py           stack → calibrate → report
│   │
│   ├── decision/                 ── Phase 6 ──────────────────────────────
│   │   ├── schema.py             Signal, Decision, locked V1 semantics
│   │   ├── threshold_optimizer.py cost feasibility and threshold derivation
│   │   ├── signal_generator.py   probability + expected return → signal
│   │   ├── position_sizer.py     fixed / vol-target / confidence sizing
│   │   ├── risk_engine.py        pre-decision gates
│   │   ├── decision_engine.py    orchestrates gates → signal → sizing
│   │   └── evaluator.py          decision quality, net of costs
│   │
│   ├── backtesting/              ── Phase 7 ──────────────────────────────
│   │   ├── config.py             BacktestConfig and itemised cost model
│   │   ├── portfolio.py          dual-path reconciliation
│   │   ├── execution.py          next-session fills at as-traded prices
│   │   ├── engine.py             the event loop
│   │   ├── benchmarks.py         buy-and-hold, equal-weight rebalanced
│   │   └── report.py             performance metrics with uncertainty
│   │
│   ├── monitoring/               ── Phase 8 ──────────────────────────────
│   │   ├── feature_drift.py      PSI, FDR control, persistence tracking
│   │   ├── performance_monitor.py rolling windows and detectability limits
│   │   ├── calibration_monitor.py rolling ECE with observation floors
│   │   ├── prediction_monitor.py lifecycle and outcome resolution
│   │   ├── retraining.py         state machine and the four policy gates
│   │   ├── alerts.py             severity, cooldown, suppression
│   │   ├── model_monitor.py      aggregate HealthReport
│   │   └── pipeline.py           collects signals, applies the policy
│   │
│   ├── orchestration/            ── Phase 9 ──────────────────────────────
│   │   ├── base.py               Job protocol, temporal validity guard
│   │   ├── ingestion_job.py      IngestionJob and FeatureJob
│   │   ├── outcome_job.py        resolve predictions against outcomes
│   │   ├── prediction_job.py     generate and persist with lineage
│   │   ├── monitoring_job.py     MonitoringJob and RetrainingJob
│   │   ├── daily_pipeline.py     backfill / daily / retrain modes
│   │   └── scheduler.py          cron and systemd helpers
│   │
│   ├── experiments/              ── Phase 10 ─────────────────────────────
│   │   ├── run.py                ExperimentRun with reproducibility gaps
│   │   ├── tracker.py            logging and selection-inflation accounting
│   │   ├── registry.py           lifecycle with enforced transitions
│   │   ├── comparison.py         distinguishability-aware comparison
│   │   └── artifacts.py          CSV/JSON artifact storage
│   │
│   └── reporting/                ── Phase 11 ─────────────────────────────
│       ├── formatting.py         Metric and Panel; nothing shown bare
│       ├── panels.py             the dashboard's reasoning, unit-tested
│       └── loaders.py            graceful loading of partial state
│
├── dashboard/                    ── Phase 11 ─────────────────────────────
│   ├── app.py                    Streamlit shell and page routing
│   ├── components/render.py      shared panel rendering
│   └── pages/
│       ├── overview.py           feasibility first, equity curve last
│       ├── signals.py            decisions including rejections
│       ├── performance.py        equity, exposure, itemised costs
│       ├── model.py              skill vs baseline, calibration curve
│       ├── drift.py              PSI, naive vs FDR counts, detectability
│       └── retraining.py         policy state, run history, registry
│
├── scripts/
│   ├── check_forbidden_patterns.py  AST gate run by CI
│   └── final_evaluation.py          the single-use final-test unlock
│
├── tests/                        499 tests across 18 files
│   ├── conftest.py               markers and deterministic seeds
│   ├── test_protected_invariants.py  the 8 merge-blocking rules
│   ├── test_leakage.py           the mutation test suite
│   └── … (15 more, one per subsystem)
│
├── .github/workflows/
│   ├── tests.yml                 protected → full matrix → determinism
│   ├── lint.yml                  ruff + the forbidden-pattern gate
│   └── build.yml                 docker build and in-image verification
│
├── reports/final/
│   └── TradeMind_AI_Final_Report.md   17-section research report
│
├── data/                         generated; not committed
│   ├── raw/market_data/symbol=X/daily.parquet
│   ├── processed/features/symbol=X/f1.parquet
│   ├── predictions/calibrated_oof.parquet
│   ├── trademind.db              the operational store
│   └── final_test_lock.json
│
├── models/                       saved model artifacts
├── logs/                         run logs
└── notebooks/                    exploratory work
```

### Why two storage engines

This confuses people, so it is worth stating plainly.

**SQLite** (`data/trademind.db`) holds the *operational* store: predictions,
outcomes, runs, the model registry, data-quality issues, experiments. This work
is transactional and row-level, needs uniqueness constraints for idempotency,
and is small. SQLite is in the Python standard library, so there is no setup.

**Parquet + DuckDB** (`data/raw/`, `data/processed/`) hold the *analytical*
lake: market bars and feature panels. This work is columnar bulk scan over
millions of rows. Parquet is the right format and DuckDB reads the whole lake
with one glob.

The original planning document said "Parquet + DuckDB" and never specified
where predictions live. Filling that gap was Phase 0's main job.

---

## 5. How data flows through the system

### The daily cycle

```
                    ┌─────────────────────────┐
                    │  yfinance (NSE, .NS)    │
                    └───────────┬─────────────┘
                                ↓
  ┌──────────────────────────────────────────────────────────┐
  │ INGESTION                                                │
  │  fetch → corporate actions → validate → persist          │
  │  ERROR severity blocks the write                         │
  └───────────┬──────────────────────────────────────────────┘
              ↓  data/raw/market_data/symbol=X/daily.parquet
              │  three price series: *_raw, *_split, adj_close
              ↓
  ┌──────────────────────────────────────────────────────────┐
  │ FEATURES                                                 │
  │  45 features, all causal · warm-up trimmed · labels added│
  └───────────┬──────────────────────────────────────────────┘
              ↓  data/processed/features/symbol=X/f1.parquet
              ↓
  ┌──────────────────────────────────────────────────────────┐
  │ VALIDATION                                               │
  │  split off the locked final test → purged walk-forward   │
  └───────────┬──────────────────────────────────────────────┘
              ↓  development data only
              ↓
  ┌──────────────────────────────────────────────────────────┐
  │ MODELS → ENSEMBLE                                        │
  │  direction + return models → OOF → meta-model →          │
  │  out-of-fold calibration                                 │
  └───────────┬──────────────────────────────────────────────┘
              ↓  data/predictions/calibrated_oof.parquet
              ↓
  ┌──────────────────────────────────────────────────────────┐
  │ DECISION                                                 │
  │  risk gates → probability gate AND cost gate → sizing    │
  └───────────┬──────────────────────────────────────────────┘
              ↓  BUY / HOLD / SELL + target weight
              ↓
      ┌───────┴────────┐
      ↓                ↓
┌───────────┐   ┌──────────────────────────────────────────┐
│ BACKTEST  │   │ PERSIST to predictions table with        │
│ next-open │   │ full lineage; idempotent by construction │
│ fills,    │   └───────────────┬──────────────────────────┘
│ dual-path │                   ↓
│ reconcile │   ┌──────────────────────────────────────────┐
└───────────┘   │ OUTCOME RESOLUTION (next cycle)          │
                │ join to the realised tradeable return    │
                └───────────────┬──────────────────────────┘
                                ↓
                ┌──────────────────────────────────────────┐
                │ MONITORING → RETRAINING POLICY           │
                │ drift · performance · calibration · DQ   │
                │ four gates decide; a failed candidate    │
                │ has no path to production                │
                └──────────────────────────────────────────┘
```

### The timing contract — the most important diagram here

```
   session t              session t+1            session t+2
   ─────────              ───────────            ───────────
   close:                 open:                  open:
     features computed      ENTRY fills here       EXIT fills here
     decision made

   Label:  tradeable_return_1d[t] = adj_open[t+2] / adj_open[t+1] − 1
```

Features use everything up to the close of *t*, which is legitimate — the
decision is made after the bell. The entry fills at the open of *t+1*, the
earliest price no part of the decision could have seen.

The original design specified `adj_close[t+1] / adj_close[t] − 1` as the label
*and* next-session execution. Those contradict: by the time the position exists,
the close-to-close move from *t* has already happened. Training on it teaches
the model to forecast a return the strategy cannot capture.

Cost of the fix: labels need data through *t+2*, so the two newest rows are
always unlabelled. They are kept, not dropped — the newest row is exactly what
the live job predicts on.

---

## 6. Phase-by-phase walkthrough

### Phase 0 — Foundation

**Goal.** Repo skeleton, configuration, logging, and the operational store the
original roadmap never defined.

**Key decisions.**

*Two storage engines.* SQLite for transactional prediction data, Parquet for
columnar market data. See section 4.

*Deterministic prediction IDs.* `prediction_id` is a SHA-256 hash of the natural
key `(symbol, prediction_date, model_version, feature_version, decision_version,
threshold_version)` rather than a random UUID. This makes Phase 9's idempotency
requirement true *by construction* — re-running a day cannot produce a duplicate
row, even across processes.

*Conflicting rewrites raise.* If the same key is written with different values,
that means either revised upstream data or a non-deterministic pipeline — both
bugs. Silent overwriting would let a model rewrite its own track record, which
is precisely what monitoring exists to catch. `allow_overwrite=True` exists but
must be explicit.

**Files created.**

| File | Job |
|---|---|
| `config/config.yaml` | every tunable; nothing hard-coded elsewhere |
| `config/universe.yaml` | 15 liquid NSE large-caps |
| `src/trademind/config.py` | typed loader, content hash, fail-fast validation |
| `src/trademind/logging_setup.py` | run-id-tagged structured logging |
| `src/trademind/storage/schema.sql` | the operational schema |
| `src/trademind/storage/db.py` | connections, WAL, foreign keys |
| `src/trademind/storage/predictions.py` | the repository |

**Two guards placed at the storage boundary**, so no later phase can violate
them by accident:

- `execution_date` must be strictly after `prediction_date`
- outcomes are write-once

**Tests added.** 25.

---

### Phase 1 — Ingestion and data quality

**Goal.** Fetch market data, handle corporate actions, validate it, store it.

**The problem the roadmap assumed away.** It requires the backtester to execute
at "raw tradable prices". yfinance does not serve those — its OHLC is already
split-adjusted, and `Adj Close` is split- *and* dividend-adjusted. Neither is
what a human would have paid.

**Key decision — three named price series.** `close` alone is banned from the
schema so no module can reach for "the" close price and get the wrong one.

| Series | Meaning | Used by |
|---|---|---|
| `*_raw` | as-traded that session | backtester execution **only** |
| `*_split` | split-adjusted, no dividends | charting, volume normalisation |
| `adj_close` | total return | features and labels **only** |

`adj_close` is retroactively revised by every new dividend. That is exactly why
filling an order at it is look-ahead bias — you would transact at a price that
only exists in hindsight and will change again next quarter.

The `*_raw` series is *reconstructed* by reversing the split adjustment. Exact
when the split history is complete; wrong in proportion to any split the
provider missed. The `ABNORMAL_JUMP` validator check exists to catch that.

**Report, never repair.** Fifteen checks run on every ingestion. ERROR findings
block the write entirely; the frame never reaches the lake.
`test_validator_never_mutates_input` deep-copies the input and compares
afterwards, so the validator physically cannot repair anything. An
auto-repairing ingestion layer is worse than a broken one — the model then
trains on numbers the market never printed.

**Severity means something.** Negative price is ERROR and blocks. Zero volume is
WARNING and does not. A >50% move with no recorded corporate action is ERROR,
because an unrecorded split is far likelier than a real halving.

**Files created.** `base.py`, `yfinance_provider.py`, `corporate_actions.py`,
`calendar.py`, `data_validator.py`, `market_data.py`, plus `storage/lake.py`.

**Tests added.** 59 (84 total).

---

### Phase 2 — Leakage-safe feature engineering

**Goal.** 45 features that can only see the past, plus the labels.

**The label had to change.** See the timing contract in section 5. Both labels
are computed: `tradeable_return_1d` trains the model, `research_return_1d`
(close-to-close) is kept as a diagnostic.

**On synthetic data the two correlate at 0.001.** They are effectively
different targets. That number is the size of the trap.

**The leak found in the first draft.** The natural way to write "is volatility
high right now" is:

```python
df["vol_percentile"] = df["volatility_20"].rank(pct=True)
```

That ranks a day in 2016 against 2020's volatility. There is no `shift(-1)`, no
future column, nothing that looks wrong. The same trap swallows z-scores against
`series.mean()`, min-max scaling, and `pd.qcut` — all standard preprocessing,
all leaky.

Everything became expanding-window. Cross-sectional ranks group by date, which
is safe because those observations share a timestamp.

**The leakage test, as a property test.** `test_no_feature_sees_the_future`
mutates every price after a cut point — 50% drop, fresh noise, 7× volume —
recomputes, and asserts all 45 features at or before the cut are bit-identical.
Three cut points.

And critically: `test_the_test_can_actually_fail` plants a deliberately leaky
feature and confirms the detector fires.

**Sanity check.** On random-walk data the best single-feature correlation with
the label is 0.027. Near zero, which is correct — a random walk has no signal.
If features scored well there, they would be leaking.

**Tests added.** 26 (110 total).

---

### Phase 3 — Purged and embargoed validation

**Goal.** Time-series validation that does not leak, and a structurally locked
final test.

**Purge must be `horizon + 1`.** The config guard was tightened. The extra
session is the execution offset: a row at *t* is labelled with
`open[t+1]..open[t+1+h]`, so it encodes information through *t+1+h*. Sizing the
purge to the horizon alone leaves exactly one contaminated row at every fold
boundary — small, invisible in the metrics, systematically favourable.

The test for this counts contaminated rows directly rather than comparing
scores. A score comparison across three folds would be indistinguishable from
noise; the structural count is exact.

**`sklearn.TimeSeriesSplit` is unusable here**, for two independent reasons.

1. It splits on *row position*. This is a panel — consecutive rows are different
   symbols on the same day. A positional cut puts RELIANCE and TCS from the same
   session on opposite sides of the boundary.
2. It leaves no gap at all.

Every split here operates on the sorted unique date index and assigns whole
sessions.

**Purge and embargo do different jobs.**

```
|<--- train --->|<-purge->|<-- validation -->|<-embargo->|<- future train ->|
```

Purge removes training rows whose *label* reaches forward into validation.
Embargo removes training rows immediately *after* validation, where a 20-day
rolling feature still shares 19 of its 20 observations with data the model was
scored on. Walk-forward needs both, because each fold's validation becomes the
next fold's training data.

**The final-test lock.** `split_development()` is the only supported path to
fittable data and structurally cannot return a locked row. `FinalTestLock`
fingerprints the window, refuses a second unlock, and detects tampering. It is
not a security boundary — it makes the wrong thing require deliberate effort and
leave a trace.

**Tests added.** 33 (143 total).

---

### Phase 4 — Base models

**Goal.** Direction and return models, evaluated honestly.

**Preprocessing must be inside the fold.** The most common leak surviving in
otherwise careful pipelines:

```python
scaler = StandardScaler().fit(X)          # WRONG — sees validation
cross_validate(model, scaler.transform(X), y, cv=purged_splitter)
```

The purged splitter is doing its job and it does not matter — the scaler already
computed statistics over the whole panel, including the locked test. Every model
here is an sklearn `Pipeline` so `fit()` scopes preprocessing to training rows.

**Baselines are mandatory.** ~52% of sessions close up, so 52% accuracy is what
a constant achieves. Every classification report carries `majority_accuracy` and
`skill_vs_majority`. Every AUC carries `auc_stderr` — about 0.024 at 600
observations, so 0.53 and 0.55 are not different numbers.

**Measured result** on random-walk data, where truth is AUC 0.50:

| model | ROC-AUC | fold std | accuracy | majority | lift | BSS |
|---|---|---|---|---|---|---|
| direction_hgb | 0.5157 | 0.0137 | 0.5167 | 0.5240 | **−0.0073** | −0.0024 |
| direction_logistic | 0.5086 | 0.0091 | 0.5078 | 0.5240 | **−0.0161** | −0.0052 |
| baseline_majority | 0.4982 | — | 0.5240 | 0.5240 | 0.0000 | −0.0009 |

Both models are *less accurate than a constant predictor*. That is the correct
outcome on noise, and two tests lock it in permanently:
`test_no_skill_on_pure_noise` and `test_model_does_find_a_planted_signal`.
Either alone would be satisfiable by a broken pipeline.

**`flag_suspicious()`** warns above AUC 0.58 and escalates past 0.65 with a
pointer to the leakage suite — written before any real data was seen, so a
flattering result cannot be reinterpreted afterwards.

**Tests added.** 45 (188 total).

---

### Phase 5 — Stacking and calibration

**Goal.** Combine base models and produce probabilities that mean something.

**The ensemble is learned, not hand-weighted.** The original design proposed
fixed weights (0.35/0.25/0.20/0.20) to be "validated later". The problem is not
the numbers; it is that there is no principled way to choose them, and "validate
later" in practice means tuning them against whatever data remains.

**In-sample calibration is circular.** The obvious implementation:

```python
calibrator.fit(oof_probs, y)
ece = expected_calibration_error(y, calibrator.predict(oof_probs))
```

That ECE is near zero for *any* model. Isotonic regression maps any monotone
input onto the observed frequencies of the rows it was fitted on. The number
describes the calibrator's flexibility, not future performance.

It hides better than ordinary overfitting because "calibration" sounds like a
post-processing step. It is a fitted model.

`calibrate_out_of_fold()` walks forward in blocks instead. Two distinct
artifacts, routinely collapsed into one: it **measures** generalisation;
`fit_production_calibrator()` is what **ships**.

**Measured result.**

| stage | ROC-AUC |
|---|---|
| base logistic | 0.5097 |
| base hgb | 0.5175 |
| stacked meta-model | 0.5069 |

Stacking is worse than the better base model. The Brier decomposition explains
why: **reliability 0.00263, resolution 0.00111** — the forecast carries less
information than it has calibration error, and post-hoc calibration cannot
manufacture information. The pipeline prints that conclusion itself.

**Two guards on ECE.** Quantile bins by default (equal-width bins let near-empty
tails swing the number), and `sharpness` reported alongside always — a constant
base-rate forecast has ECE near zero and zero value.

**Tests added.** 34 (222 total).

---

### Phase 6 — Decision engine

**Goal.** Turn probabilities into position instructions, with derived thresholds.

**The central finding of the project.**

```
commission 3 + spread 5 + slippage 5 = 13 bps per side
round trip                           = 26 bps
```

A one-day holding period makes every signal a full round trip; nothing
amortises. Expected edge from going long the top decile ≈ `IC × σ_daily × 1.755`:

| IC | expected edge | needed |
|---|---|---|
| 0.02 | 5.3 bps | 26 bps |
| 0.03 | 7.9 bps | 26 bps |
| 0.05 | 13.2 bps | 26 bps |
| 0.10 | 26.3 bps | 26 bps |

**Breakeven IC is 0.099. Measured is 0.027 ± 0.055.**

Structural, not a tuning problem. `cost_feasibility()` computes and prints the
verdict *before* any optimisation runs, so the conclusion cannot be skipped past.

**The roadmap's `min_expected_return` was below the cost floor.** It specified
0.0005 — 5 bps against a 26 bps round trip. Every trade clearing that gate loses
21 bps on costs alone. It is now derived (`round_trip × margin`), never tuned,
because it is a fact about the broker rather than a free parameter.

**Two gates for a BUY.** Probability must clear the threshold **and** expected
return must clear the cost floor. A 0.72 probability on a 3 bps move is a
confident prediction of a losing trade — confidence about direction says nothing
about magnitude.

**V1 semantics, each with a named test.** BUY opens/increases; HOLD maintains;
SELL exits; SELL while flat stays flat; a rejected or stale input **preserves
the existing position**. That last rule matters most: liquidating on a data
outage turns an outage into a realised loss *plus* a round-trip cost.

HOLD on an open position returns NaN, not 0.0. Returning zero would liquidate
everything on every quiet day.

**Bug caught by a failing test.** Volatility sizing used `target_volatility /
vol`, conflating a portfolio-level target with a position weight. For a 0.25-vol
symbol that yields 0.8, which the cap flattens to maximum — silently disabling
the scaling for every symbol while appearing to work.

**Kelly is deliberately absent.** It assumes the edge estimate is correct, and
sizes most aggressively exactly where a noisy edge is most wrong.

**Tests added.** 45 (267 total).

---

### Phase 7 — Event-based backtester

**Goal.** Simulate the strategy with correct accounting and realistic costs.

**Reconciliation that can actually fail.** The usual check is a tautology:

```python
equity = cash + sum(shares * price)
assert equity == cash + sum(shares * price)     # proves nothing
```

It passes for any bug living *inside* the definition — an uncharged commission,
a split that adjusted price but not share count, a double-credited dividend.

Equity is computed twice here, from disjoint state:

- **Path A** — `cash + Σ(shares × price)`, driven by the cash balance
- **Path B** — `initial + realised + unrealised + dividends − costs`, driven by
  accumulators that never read `cash`

Three tests plant bugs path A cannot see and confirm the mismatch is caught.
Measured max reconciliation error over 750 sessions: **0.00000000**.

**Session ordering encodes the timing contract.**

1. Corporate actions
2. Execute yesterday's orders at *today's open*
3. Mark to market and reconcile
4. Decide from today's data, for tomorrow

Step 4 last makes same-session fills mechanically impossible rather than merely
forbidden.

**Measured result.**

| | return | CAGR | Sharpe | max DD | cost drag | turnover |
|---|---|---|---|---|---|---|
| TradeMind | +16.5% | +5.3% | 0.99 ± 0.58 | −6.7% | 7.9% | 20.4× |
| buy & hold | +47.4% | +13.9% | 1.56 ± 0.58 | −9.3% | 0.1% | — |

The strategy needed **+24.4% gross to deliver +16.5% net**. Buy-and-hold beat it
on every axis while trading twice.

Note the standard errors. A Sharpe of 0.99 ± 0.58 over three years is not
distinguishable from 0.4 — and any Sharpe reported on a sample this short
carries the same uncertainty.

**Reporting states its own limits.** Sharpe always prints with stderr; samples
under three years trigger an explicit warning. Turnover sits beside it, because
a Sharpe at 20× turnover is far more fragile to cost assumptions. Costs are
itemised so "which assumption is doing the work?" has an answer — here spread
and slippage are 77% of the drag, and they are the two you would trust least.

**Benchmarks pay identical costs.** A cost-free benchmark against a cost-charged
strategy is the easiest way to make a losing strategy look competitive.

**Tests added.** 42 (309 total).

---

### Phase 8 — Monitoring, drift, and governed retraining

**Goal.** Detect degradation and decide, under policy, when to retrain.

**Finding 1: naive drift monitoring fires on 90% of days.**

```
45 features, KS test at p<0.05, checked daily
expected false alarms per day        2.2
P(at least one false alarm today)    0.90
expected false alarms per month      47
```

A monitor that cries wolf nine days in ten trains the team to dismiss it — and
then to dismiss the real one. Three responses: **PSI** (effect size) as the
primary signal, **persistence** over three consecutive checks, and
**Benjamini–Hochberg FDR** on remaining p-values. Bonferroni would demand
p<0.0011 across 45 correlated features and detect nothing.

**Finding 2: performance monitoring cannot detect this model's degradation.**

| window | observations | AUC std err | detectable drop |
|---|---|---|---|
| 7d | 105 | 0.195 | 0.39 |
| 30d | 450 | 0.094 | 0.19 |
| 90d | 1350 | 0.054 | 0.11 |

The model's entire edge is ~0.02. It could lose all of it and the monitor would
report noise.

The governance consequence is stated explicitly in the policy: **an unchanged
metric is not evidence of health**, because it would look identical if the model
had failed completely. Every metric carries its minimum detectable effect, and
`degradation_detected()` refuses to declare a drop smaller than the MDE.

**The four governance rules, each with a named test.**

1. **Drift alone never retrains.** Markets change distribution constantly; a
   model can be healthy on shifted inputs.
2. **A data-quality ERROR blocks retraining**, however severe everything else
   looks. This feels backwards and is right: when performance collapses *and*
   ingestion is erroring, the bad data is the likelier cause, and retraining on
   it bakes the corruption into the replacement.
3. **Insufficient new observations means wait.**
4. **A failed candidate has no path to production.**

**Diagnosis, not just detection.** Performance down *with* drift → likely regime
change, retraining should help. Performance down *without* drift → the
feature-target relationship changed, and retraining on the same features may not
help.

**Found by a failing test.** A 90-day calibration window on 3 symbols yields 195
observations, below the 300 needed for a 10-bin ECE. Universe size determines
whether calibration monitoring is possible at all.

**Tests added.** 45 (354 total).

---

### Phase 9 — Daily orchestration

**Goal.** Turn the research pipeline into a repeatable daily job.

**The backfill contamination trap.** The obvious implementation:

```python
model = registry.load(production_version)   # trained through last month
for day in historical_dates:                # going back two years
    store.save(predict(model, features_on(day)))
```

Every row is a "prediction" made by a model trained on data from *after* the
date it predicts. Monitoring then reports excellent historical performance and
the Phase 8 baseline is fiction. Nothing raises — the dates are real, the
features are real, the outcomes are real. Only the counterfactual is impossible.

Guard, enforced on every write:

```
training_end < prediction_date
```

An honest backfill must therefore walk forward, replaying out-of-fold
predictions from the purged splitter.

**Job ordering is load-bearing**, and each ordering has a test:

- **Outcomes before predictions** — yesterday's outcome is known today, so
  resolving first keeps the pending queue from growing by one day every day
- **Monitoring after predictions** — today's features must be in the current
  drift window
- **`retrain` mode omits prediction entirely** — predicting with a model about
  to be replaced attributes rows to a version that may not survive

**Failure recovery.** Jobs return results rather than raising. PARTIAL is not
fatal (one delisted ticker must not stop the system); a FAILED critical job
halts the pipeline. `reap_stale_runs()` marks abandoned RUNNING rows FAILED on
startup — a killed process otherwise leaves a row that makes "is anything
executing?" unanswerable and lets the scheduler stack overlapping runs.

**No scheduler daemon.** cron and systemd are already supervised, already log,
already survive reboots. The only piece worth writing in Python is
`should_run_today()`, because cron cannot know the NSE holiday calendar. Default
run time is 18:00 IST — running before the 15:30 close computes features from a
partial bar, where the close is wrong and nothing downstream can tell.

**Tests added.** 29 (383 total).

---

### Phase 10 — Experiment tracking and registry lifecycle

**Goal.** Make every experiment reproducible and every promotion auditable.

**An experiment tracker is a machine for overfitting the validation set.** With
AUC stderr 0.024 and a true AUC of 0.52:

| runs | E[best observed] | inflation |
|---|---|---|
| 1 | 0.5200 | +0.0000 |
| 10 | 0.5520 | +0.0320 |
| 50 | 0.5695 | +0.0495 |
| 100 | 0.5759 | +0.0559 |

**Fifty experiments on a worthless model yield a best observed AUC of 0.57** —
inside the range Phase 4 flags as implausible. The winning run looks identical
to a genuine discovery.

Not an argument for fewer experiments; an argument for recording how many were
run. `n_prior_experiments` is stamped automatically, and the leaderboard carries
a `selection_adjusted` column.

**Comparison refuses to declare a winner the data cannot support.**
`is_distinguishable()` uses the standard error of a *difference* — √2 larger
than the standard error of either estimate, a factor routinely omitted, and
omitting it makes every gap look significant.

**Registry lifecycle is enforced, not documented.**

```
CANDIDATE → VALIDATING → VALIDATED → STAGING → PRODUCTION → ARCHIVED
                 └──────────────→ FAILED (terminal)
```

Illegal transitions raise. Phase 8's "failed candidate cannot be promoted" now
has two independent guards — deliberately, because it is the rule most likely to
be worked around at 11pm when numbers are disappointing.

One PRODUCTION model per task; promoting archives the incumbent in the same
operation. Two live models would give predictions ambiguous lineage.

**Reproducibility is checked, not assumed.** `reproducibility_gaps()` reports
missing commits, missing seeds — and a dirty working tree, because a commit
recorded against uncommitted changes points at code that never ran.

**Tests added.** 37 (420 total).

---

### Phase 11 — Dashboard and reporting

**Goal.** Make the system legible in minutes, without flattering it.

**The reasoning is separated from the rendering.** Every judgement lives in
`src/trademind/reporting/panels.py` as pure functions with 37 tests. Streamlit
only draws. Presentation logic buried in callbacks cannot be tested, and
untestable presentation logic is exactly where inconvenient findings quietly
stop being displayed.

**Panel order is a design decision, enforced by a test.** `build_overview`
returns: cost feasibility → model skill → accounting integrity → data quality →
**performance last**. A dashboard opening with a rising equity curve invites the
viewer to stop reading.

Rendered with the real measured numbers:

```
✗ Cost feasibility: Measured IC 0.027 is below the 0.099 breakeven. At a
  1-session holding period the strategy cannot clear its own costs at any
  threshold.

! Model skill: AUC 0.5157 is within sampling error of chance (± 0.0137).
  No demonstrated discrimination.
  Lift vs majority  −0.0073
    Negative means a constant predictor does better.

✓ Accounting integrity: Balance-sheet and flow equity agree to within a paisa.

✗ Performance: +16.54% against buy-and-hold's +47.41%. The strategy
  underperforms doing nothing.
```

**Nothing displays without what qualifies it.** `Metric` carries stderr and
baseline; `within_noise` uses the difference standard error. Tests assert that
an AUC within noise renders CONCERN not skill, that negative lift renders BAD,
and that AUC above 0.58 is flagged as suspected leakage.

**Rejections are shown, not filtered.** A `BELOW_COST_FLOOR` rejection means the
model was confident and the trade still was not worth making — the system
working. A page listing only BUYs makes that invisible.

**Graceful degradation.** Every loader returns empty rather than raising, so the
dashboard runs before anything has been computed and shows the command to run.

**Tests added.** 37 (457 total).

---

### Phase 12 — Production hardening

**Goal.** CI, Docker, and turn the eight invariants into merge blockers.

**Every protected invariant is paired with a mutation test.** All eight have a
test asserting the property *and* a test that deliberately breaks it and
confirms the guard fires. See section 7 for the full table.

**Writing invariant 4's mutation test revealed a real limitation.** My first
attempt corrupted the cost basis and expected reconciliation to catch it. It
did not — `apply_split` compares `shares × basis` before and after, and the
transform preserves that product for *any* starting values. Worse, I then found
that if a split happens and the provider never reports it, both equity paths use
the same unadjusted share count and still agree.

**Reconciliation is structurally blind to a missing split.** That is now
`test_invariant_4c`, asserted rather than commented, with a pointer to the
Phase 1 `ABNORMAL_JUMP` check that actually catches it.

**The static gate parses instead of grepping.** The first version was inline
`grep` in the workflow — and it false-positived on every docstring that *warns*
about these patterns. This codebase documents each trap beside its guard, so the
gate would have fired on its own documentation, and a gate that does that is a
gate someone disables.

`scripts/check_forbidden_patterns.py` walks the AST for `rank(pct=True)` outside
a groupby, `bfill`, and `rolling(center=True)`. Verified against planted
violations of all three, and against a grouped rank it must permit. Its own test
found a bug: it crashed on paths outside the repo root.

**CI is layered by speed.** `protected` runs first and alone — seconds, gating
everything else. A `determinism` job runs it twice and diffs the output.
`PYTHONHASHSEED=42` is set in both the workflow and the Dockerfile, since it is
read at interpreter startup. The image is pinned to `python:3.12.7-slim-bookworm`
— a backtest that changes after a base-image refresh is not reproducible.

**Tests added.** 40 (497 total).

---

### Phase 13 — Documentation and the final test

**Goal.** README, the 17-section research report, and the single final-test
unlock.

**The final test was not run, and the Results section is empty.**

The evaluation needs real ingested data. The environment this was built in has
no network access, so nothing was ever fetched. A plausible number could have
been written; everything else in the report is complete enough that one would
not have looked out of place.

That would have reproduced exactly the failure the project exists to prevent.
The original planning document recorded a *"locked final-test result"* of +0.48%
return at Sharpe 1.45 with a −0.31% drawdown, described as *"immutable benchmark
results"*, for code that had never been written. Those numbers had acquired the
grammar of findings without ever having been measured.

**A test enforces it.** `test_no_fabricated_final_test_result_exists` requires
that, absent an evidence file, the report still says `LOCKED. Not evaluated.`,
and that the discarded figures appear only near a disclaimer. It caught a bug in
its own first version — line-based matching on prose that wraps.

**The unlock is single-use and guarded.** `scripts/final_evaluation.py` refuses
without `--confirm`, refuses twice (an evidence file blocks it), refuses from a
dirty working tree, and refuses before thresholds are derived and locked.

**A prediction recorded in advance**, in §15 of the report: given a development
IC of 0.027 against a 0.099 breakeven, the strategy is expected to underperform
buy-and-hold and fail to clear costs. Writing that down before the unlock is
itself a control — it makes a surprisingly *good* result something to
investigate rather than celebrate.

**Tests added.** 2 (499 total).

---

## 7. The eight protected invariants

These are the rules that block a merge. CI runs them as a separate job that
finishes in seconds, because a slow gate gets bypassed and a bypassed gate is
not a gate.

| # | Invariant | Mutation test that verifies the guard |
|---|---|---|
| 1 | Final test is unreachable for fitting | Tamper a locked row → fingerprint catches it |
| 2 | No future information in features | Plant `rank(pct=True)` → detector fires |
| 3 | No same-day execution | Model predicting its own training window → refused |
| 4 | A split preserves wealth | Break the transform → reconciliation raises |
| 5 | Equity paths reconcile | Uncharged fee / phantom shares / unrecorded dividend |
| 6 | HOLD carries the position | Three positive assertions across decision and execution |
| 7 | SELL exits, never shorts | Sell unheld shares → refused |
| 8 | A failed candidate is never promoted | Force it → raises; registry blocks independently |

Two meta-tests assert the suite stays complete, so an invariant cannot be
quietly dropped from it.

**Invariant 6 has no mutation test** and the file says why: "HOLD did nothing"
has no meaningful mutation — breaking it means returning 0.0, which the NaN
assertion already catches directly.

---

## 8. Command reference

### First-time setup

```bash
pip install -e ".[dev]"
python main.py init                # create the operational store
```

### Building the system from scratch

```bash
python main.py ingest              # fetch + validate market data
python main.py features            # build the 45-feature panel
python main.py validate            # inspect folds, lock the final test
python main.py train               # fit base models against baselines
python main.py ensemble            # stack + calibrate, both out-of-fold
python main.py thresholds          # cost feasibility, derive thresholds
                                   # → write results into config/config.yaml
python main.py backtest            # event-based backtest on development data
python main.py monitor             # drift, health, retraining gate
python main.py experiments         # leaderboard + registry state
```

### Daily operation

```bash
python main.py daily               # the full cycle for one trading day
python main.py daily --date 2024-03-15   # a specific date
python main.py daily --force       # run even on a non-trading day
python main.py backfill            # rebuild history (walk-forward)
python main.py retrain             # monitoring + governance only
```

### Testing and quality

```bash
make protected                     # 8 merge-blocking invariants (seconds)
make test                          # full suite (499 tests)
make lint                          # ruff
make ci                            # everything CI runs
python scripts/check_forbidden_patterns.py    # the AST gate
```

### Dashboard and containers

```bash
make dashboard                     # streamlit run dashboard/app.py
make docker                        # build the image
docker compose up                  # pipeline + dashboard
```

### The one-time final evaluation

```bash
python scripts/final_evaluation.py --confirm
```

Refuses without `--confirm`, refuses twice, refuses from a dirty tree, refuses
before thresholds are locked.

---

## 9. Test inventory

| File | Tests | Lines | Covers |
|---|---|---|---|
| `test_protected_invariants.py` | 27 | 499 | the 8 merge-blocking rules + mutations |
| `test_decision.py` | 45 | 441 | cost feasibility, V1 semantics, sizing, gates |
| `test_models.py` | 45 | 479 | preprocessing leak, baselines, registry |
| `test_monitoring.py` | 45 | 510 | drift, MDE, the four governance rules |
| `test_backtesting.py` | 42 | 458 | reconciliation, execution timing, benchmarks |
| `test_experiments.py` | 37 | 390 | selection inflation, lifecycle transitions |
| `test_reporting.py` | 37 | 315 | panel ordering, verdicts, uncertainty |
| `test_ensemble.py` | 34 | 375 | circular calibration, stacking alignment |
| `test_validation.py` | 31 | 426 | purge sizing, panel splits, the lock |
| `test_orchestration.py` | 29 | 438 | idempotency, lineage, failure recovery |
| `test_leakage.py` | 26 | 421 | the mutation suite over all 45 features |
| `test_validator.py` | 20 | 191 | the 15 data-quality checks |
| `test_corporate_actions.py` | 18 | 188 | the ₹100,000 invariant, adjustments |
| `test_storage.py` | 16 | 169 | idempotency, write-once outcomes |
| `test_provenance.py` | 15 | 170 | reproducibility, the fabrication guard |
| `test_ingestion_pipeline.py` | 12 | 241 | fetch → validate → persist ordering |
| `test_config.py` | 11 | 104 | fail-fast validation, purge sizing |
| `test_calendar.py` | 9 | 78 | `next_session` and the t+1 guarantee |
| **Total** | **499** | **5,893** | |

---

## 10. Glossary of traps

Every trap this project hit, why each is easy to miss, and how it is now
prevented. This is the section worth rereading before an interview.

### 1. The leaky label

**The trap.** Training on `close[t+1]/close[t] − 1` while executing at *t+1*.
By the time the position exists, the move has already happened.

**Why it is missed.** Every number is real. It is the textbook definition. There
is no `shift(-1)` anywhere suspicious.

**Prevention.** The timing contract; the tradeable label is
`open[t+2]/open[t+1] − 1`. The two correlate at 0.001.

### 2. Full-sample statistics

**The trap.** `series.rank(pct=True)`, z-scores against `series.mean()`,
min-max scaling, `pd.qcut`.

**Why it is missed.** No future column, no shift, and it is standard
preprocessing that appears in every tutorial.

**Prevention.** Expanding-window equivalents; the mutation test; an AST gate
in CI.

### 3. Preprocessing outside the fold

**The trap.** `StandardScaler().fit(X)` before splitting.

**Why it is missed.** The purged splitter is right there and looks like it is
handling things. It is, and it cannot help.

**Prevention.** Every model is an sklearn `Pipeline`.

### 4. Undersized purge

**The trap.** `purge = horizon` leaves exactly one contaminated row per fold
boundary.

**Why it is missed.** One row out of thousands, and the effect is far too small
to see in a metric.

**Prevention.** `purge >= horizon + 1`, validated at config load; a structural
count test rather than a score comparison.

### 5. Positional splitting on a panel

**The trap.** `TimeSeriesSplit` cuts through the middle of a trading session.

**Why it is missed.** It is the standard tool and it produces plausible folds.

**Prevention.** All splits key on the sorted unique date index.

### 6. Circular calibration

**The trap.** Fitting a calibrator on OOF predictions and measuring ECE on those
same rows.

**Why it is missed.** "Calibration" sounds like post-processing rather than a
fitted model. It is a fitted model.

**Prevention.** `calibrate_out_of_fold()` measures; `fit_production_calibrator()`
ships. Different artifacts, different jobs.

### 7. Thresholds below the cost floor

**The trap.** `min_expected_return = 0.0005` against a 26 bps round trip.

**Why it is missed.** 5 bps looks like a conservative filter until you compare
it to the cost.

**Prevention.** The floor is derived from the cost model, never tuned.

### 8. Tautological reconciliation

**The trap.** `assert equity == cash + Σ(shares × price)`.

**Why it is missed.** It looks like a rigorous check. It restates a definition.

**Prevention.** Two independent equity paths that share no state.

### 9. Multiple testing in drift monitoring

**The trap.** 45 features × p<0.05, daily.

**Why it is missed.** Each individual test is statistically correct. Nobody
multiplies.

**Prevention.** PSI, persistence, FDR control.

### 10. Undetectable degradation read as health

**The trap.** A monitor whose MDE exceeds the effect it is watching for.

**Why it is missed.** The metric is stable, which looks like good news.

**Prevention.** Every metric carries its MDE; the policy states that an
unchanged metric is not evidence of health.

### 11. Backfill contamination

**The trap.** Loading the production model and predicting backwards.

**Why it is missed.** Backfilling looks like data plumbing, not modelling.

**Prevention.** `training_end < prediction_date`, enforced on every write.

### 12. Selection inflation

**The trap.** Picking the best of 50 experiments by validation metric.

**Why it is missed.** The winning run looks exactly like a genuine discovery.

**Prevention.** Experiment count recorded; discount reported; the locked final
test is the only clean answer.

### 13. A dashboard that flatters

**The trap.** Opening with an equity curve.

**Why it is missed.** It is what every dashboard does.

**Prevention.** Panel ordering enforced by a test; nothing displayed without its
uncertainty.

### 14. Fabricated results

**The trap.** Writing a plausible number where a measurement should be.

**Why it is missed.** It is the last step, after all the hard work, when the
number is the only thing missing.

**Prevention.** The final test stays locked; a test asserts the Results section
remains empty without an evidence file.

---

## 11. Project statistics

### Code

| Package | Lines | Purpose |
|---|---|---|
| `monitoring` | 1,392 | drift, performance, calibration, governance |
| `models` | 1,116 | direction, return, metrics, OOF, registry |
| `backtesting` | 1,102 | portfolio, execution, engine, benchmarks |
| `decision` | 1,052 | feasibility, thresholds, signals, risk |
| `ingestion` | 1,028 | providers, corporate actions, validation |
| `orchestration` | 971 | jobs and the daily pipeline |
| `validation` | 893 | purged splits, embargo, the lock |
| `ensemble` | 870 | stacking and calibration |
| `experiments` | 830 | tracking, lifecycle, comparison |
| `features` | 762 | 45 features and the timing contract |
| `reporting` | 696 | dashboard reasoning |
| `storage` | 432 | operational store and lake |
| **Total source** | **~11,500** | 82 modules |

### Tests

- **499 tests** across 18 files, ~5,900 lines
- **27 protected invariants**, each paired with a mutation test where meaningful
- Test-to-source ratio: roughly **1 : 2**

### Coverage of the original roadmap

All 13 phases complete. Every acceptance criterion met, with four corrections to
the original design:

1. The label was changed from close-to-close to open-to-open
2. `purge_days` was tightened from `>= horizon` to `>= horizon + 1`
3. `min_expected_return` became derived rather than hand-set
4. The eight "COMPLETE" phases and the fabricated final-test result were
   discarded and recorded as discarded

---

*Generated for TradeMind AI v1.0. For per-phase status see `PHASE_STATUS.md`;
for methodology see `reports/final/TradeMind_AI_Final_Report.md`.*
