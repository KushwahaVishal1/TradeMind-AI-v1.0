# TradeMind AI

**Probabilistic market signal and decision-support platform for NSE equities.**

An end-to-end time-series ML system that generates calibrated BUY/HOLD/SELL
signals under strict look-ahead controls, evaluates them in an event-based
backtester with realistic transaction costs, and monitors its own predictions
against realised outcomes.

> **This is a research and decision-support system.** It is not investment
> advice, not a profitable strategy, and not an autonomous trading system. Its
> value is the controls: leakage prevention, temporal validation, accounting
> correctness, and governance.

---

## The headline finding

Most stock-prediction projects report an impressive number. This one reports an
honest one, and the honest one is negative:

```
Round-trip cost at a 1-day holding period       26 bps
Information coefficient required to break even   0.099
Information coefficient actually measured        0.027 ± 0.055
```

**The strategy cannot clear its own transaction costs at any threshold.** That
is arithmetic, not a tuning problem — every candidate threshold is choosing
among trades that lose money on costs before the market moves.

The system says this itself, on the first panel of the dashboard, before it
shows an equity curve.

---

## What was built, and what it found

| Phase | Component | Finding |
|---|---|---|
| 0 | Config, logging, operational store | Deterministic `prediction_id` makes idempotency structural |
| 1 | Ingestion, data quality, corporate actions | yfinance serves no as-traded prices; three named price series |
| 2 | Leakage-safe features | The roadmap's label was untradeable; its correlation with the tradeable one is 0.001 |
| 3 | Purged/embargoed validation | Purge must be `horizon + 1`; the +1 is the execution offset |
| 4 | Base models | Both models score **below** a constant predictor (lift −0.007, −0.016) |
| 5 | Stacking + calibration | Stacking does not beat the best base model; in-sample calibration is circular |
| 6 | Decision engine | Breakeven IC is 0.099 against a measured 0.027 |
| 7 | Event-based backtester | Costs consumed 7.9% of capital; buy-and-hold won on every axis |
| 8 | Monitoring + governance | Naive drift monitoring fires on 90% of days; degradation is undetectable at this signal strength |
| 9 | Daily orchestration | Naive backfill produces predictions no model could have made |
| 10 | Experiment tracking | 50 experiments on noise produce a best AUC of 0.57 |
| 11 | Dashboard | Panel order puts feasibility first, equity curve last |
| 12 | CI + hardening | Every invariant paired with a mutation test that breaks it |
| 13 | Documentation | Final test still locked — see below |

**Documentation:**

- [docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md) — full walkthrough
  of all 13 phases, the build process, and every file's job
- [PHASE_STATUS.md](PHASE_STATUS.md) — honest per-phase status
- [reports/final/TradeMind_AI_Final_Report.md](reports/final/TradeMind_AI_Final_Report.md)
  — the 17-section research report

---

## The controls, and what each prevents

| Risk | Control | Enforced by |
|---|---|---|
| Leaky features | Features at *t* use only data at or before *t* | Mutation test: perturb the future, assert the past is bit-identical |
| Leaky preprocessing | Scalers and imputers fitted inside each fold | Every model is an sklearn `Pipeline` |
| Full-sample statistics | Expanding windows, never `rank(pct=True)` | AST checker in CI |
| Leaky validation | Purged, embargoed walk-forward on dates, not row positions | `assert_fold_is_clean` on every fold |
| Untradeable labels | Label is `open[t+2]/open[t+1] − 1` | Timing contract in `features/labels.py` |
| Test-set optimisation | Final window locked, fingerprinted, single-use | `FinalTestLock`, `scripts/final_evaluation.py` |
| Circular calibration | Calibrator fitted on earlier blocks, applied to later | `calibrate_out_of_fold` |
| Same-day execution | Fills at the next session's open, at as-traded prices | Refused at the storage *and* execution boundaries |
| Accounting error | Equity computed twice from disjoint state | Dual-path reconciliation, every session |
| Corporate actions | Splits move share count and cost basis, not wealth | ₹100,000 invariant test |
| Selection inflation | Experiment count recorded, discount reported | `selection_inflation()` |
| Backfill contamination | `training_end < prediction_date` | `assert_temporally_valid` on every write |
| Reflexive retraining | Governed policy with four gates | Drift alone never retrains |
| Silent promotion | Failed candidates have no path to production | Two independent guards |

---

## Quick start

```bash
pip install -e ".[dev]"

python main.py init          # create the operational store
python main.py ingest        # fetch and validate market data
python main.py features      # build the leakage-safe feature panel
python main.py validate      # inspect folds, lock the final-test window
python main.py train         # fit base models against baselines
python main.py ensemble      # stack and calibrate, both out-of-fold
python main.py thresholds    # cost feasibility, then derive thresholds
python main.py backtest      # event-based backtest on development data
python main.py monitor       # drift, health, retraining gate
```

Then daily:

```bash
python main.py daily         # ingest → features → outcomes → predict → monitor
```
Dashboard: `make dashboard`. Docker: `docker compose up`.
```bash
python -m streamlit run dashboard/app.py         # Terminal Dashboard
docker compose up --build -d dashboard   # start dashboard at http://localhost:8501
```

```bash
docker compose down                      # stop all services
```
**Expect warnings on the first ingest run.** Read them. That output is the
data-quality section of your report, and "here are the 23 anomalies I found in
free-tier data" is a stronger claim than silence.

---

## Testing

```bash
# Windows PowerShell Commands

# 1. Run protected invariants (8 merge-blocking tests)
 $env:PYTHONHASHSEED="42"; pytest -m protected -v

# 2. Run full test suite (499 tests)
 $env:PYTHONHASHSEED="42"; pytest -q

# 3. Run complete local CI pipeline (ruff linting + protected tests + full suite)
ruff check src tests dashboard; $env:PYTHONHASHSEED="42"; pytest -m protected; pytest -q
```

```bash
# Docker execution:
# Run the 8 merge-blocking invariant tests in Docker
docker compose run --rm trademind pytest -m protected -v

# Run full test suite in Docker
docker compose run --rm trademind pytest -q
```

```bash
#For Linux / macOS / Git Bash:
make protected    # the 8 merge-blocking invariants — seconds
make test         # full suite — 499 tests
make ci           # lint + protected + full, as CI runs it
```

Every protected invariant has a **mutation test** that breaks the property and
confirms the guard fires. An invariant test that cannot fail proves nothing.

---

## Layout

```
config/                  every tunable, content-hashed for lineage
src/trademind/
  config.py              typed loader with fail-fast validation
  provenance.py          reproducibility capture
  storage/               SQLite operational store + Parquet lake
  ingestion/             providers, corporate actions, calendar, validation
  features/              leakage-safe features and the timing contract
  validation/            purged splits, embargo, the final-test lock
  models/                direction and return models, metrics, baselines
  ensemble/              OOF stacking and out-of-fold calibration
  decision/              cost feasibility, thresholds, signals, risk
  backtesting/           execution, portfolio, benchmarks, reporting
  monitoring/            drift, performance, calibration, retraining policy
  orchestration/         daily/backfill/retrain jobs and the pipeline
  experiments/           tracking, registry lifecycle, comparison
  reporting/             dashboard reasoning (tested separately from rendering)
dashboard/               thin Streamlit shell
scripts/                 forbidden-pattern gate, final evaluation
tests/                   499 tests, 27 of them protected invariants
```

82 modules, ~11,500 lines of source, ~5,900 lines of tests.

---

## Known limitations

Stated here rather than buried, because they bound every number the system
produces.

- **Survivorship bias.** The universe is today's liquid large-caps, so names
  that were liquid in 2015 and have since delisted are silently excluded. Every
  backtest figure is optimistic by an unmeasured amount.
- **Reconstructed execution prices.** yfinance serves split-adjusted OHLC, not
  as-traded prices. The as-traded series is reconstructed by reversing the split
  adjustment, which is exact only when the split history is complete.
- **Approximate calendar without `pandas-market-calendars`.** The weekday
  fallback flags Indian festival holidays as missing sessions.
- **Single market, daily bars, long only.** No intraday, no shorting, no
  leverage.
- **Monitoring cannot detect this model's degradation.** Even a 90-day window
  can only see AUC changes of 0.11; the model's entire edge is 0.02.
- **Reconciliation cannot detect a missing split.** Both equity paths share the
  share count. Detection lives upstream, in the ingestion validator.
- **No live trading.** Nothing here has traded real money.

---

## Final test status

**ORIGINAL LOCK INVALIDATED. Not evaluated.**

The Phase 3 window was never successfully unlocked and no result exists. Its
fingerprint no longer matches the regenerated historical features, and the
exact original snapshot was not found in the repository or available archives.
The original fingerprint remains unchanged as integrity evidence.

A replacement future holdout is frozen in `data/final_test_plan_v2.json` for
2026-09-02 through 2027-01-29. Until it completes, continue running the daily
pipeline after market close without tuning against holdout outcomes.

To run it — once:

```bash
python scripts/final_evaluation.py --confirm
```

The script refuses to run twice, refuses to run from a dirty working tree, and
refuses to run before thresholds have been derived and locked on development
data.

Until it has been run against real ingested data, the Results section of the
final report stays empty. **Fabricating a number there is the exact failure this
project was built to prevent** — the original planning document for TradeMind AI
recorded a "locked final-test result" of +0.48% return at Sharpe 1.45, for code
that had never been written.

---

## License

MIT — see [LICENSE](LICENSE).
