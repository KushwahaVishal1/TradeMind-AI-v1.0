# TradeMind AI — Final Research Report

**Probabilistic Market Signal & Decision-Support Platform**

Status: methodology complete; final-test evaluation **not yet run**.
See §15.

---

## 1. Executive summary

TradeMind AI is an end-to-end machine-learning platform for generating daily
probabilistic trading signals on NSE equities. It covers ingestion, corporate
actions, leakage-safe feature engineering, purged time-series validation,
model stacking, probability calibration, cost-aware decisioning, event-based
backtesting, drift monitoring, and governed retraining.

The system's principal result is negative, and it is the most useful thing the
system produces:

> At a one-day holding period and a 26 bps round-trip cost, the information
> coefficient required to break even is **0.099**. The information coefficient
> the models achieve is **0.027 ± 0.055** — indistinguishable from zero. No
> threshold, hyperparameter, or ensemble weight can bridge that gap, because
> every candidate trade loses money on costs before the market moves.

Development-set backtesting confirms it: the strategy returned +16.5% over three
years against buy-and-hold's +47.4%, while paying 7.9% of capital in costs and
turning the portfolio over 20 times a year.

The engineering value of the project is the set of controls that make this
conclusion trustworthy. A system that reports a poor honest number is more
credible than one reporting an excellent number of unknown provenance.

---

## 2. Problem definition

**Task.** Predict, for each symbol in a liquid NSE universe, whether the
tradeable forward return over the next session will be positive, and by how
much; convert those predictions into position instructions.

**What makes it hard.** Daily equity direction is close to unpredictable. The
base rate of up-days is ~52%, so a constant predictor achieves 52% accuracy.
Realistic leak-free ROC-AUC sits around 0.51–0.54, and R² on the return
regression is negative — the mean beats any individual forecast in squared
error.

**What is therefore being demonstrated.** Not predictive skill, which is
largely unattainable at this horizon. The demonstrable properties are: absence
of look-ahead bias, temporally valid evaluation, correct portfolio accounting,
honest uncertainty reporting, and governance that cannot be silently bypassed.

**Scope.** Long-only, daily bars, single market, no leverage. Deferred:
shorting, intraday, multi-market, news/sentiment.

---

## 3. Dataset and data quality

**Source.** yfinance, NSE symbols with the `.NS` suffix.

**Universe.** 15 liquid large-caps. Deliberately small: every phase iterates
fast, and liquidity keeps the cost model defensible.

**Validation.** Fifteen checks run on every ingestion — chronology, duplicates,
OHLC bound violations, non-positive prices, negative and zero volume, abnormal
jumps, corporate-action reconciliation, stale-price runs, missing and unexpected
sessions.

**Report, never repair.** ERROR-severity findings block the write; the frame
never reaches the lake. `test_validator_never_mutates_input` deep-copies the
input and compares it afterwards, so the validator physically cannot repair
anything. An auto-repairing ingestion layer is worse than a broken one: the
model then trains on numbers the market never printed and nothing downstream can
tell.

Findings persist to `data_quality_issues` at WARNING and ERROR. The Phase 8
retraining policy reads that table.

---

## 4. Corporate actions

**The invariant.** 100 shares × ₹1,000 = 200 shares × ₹500. A split changes
share count and cost basis; it does not change wealth. Asserted by
`test_split_preserves_position_value` and, end to end, by
`test_invariant_4b_a_split_preserves_portfolio_equity`.

**yfinance serves no as-traded prices.** Its OHLC is already split-adjusted and
`Adj Close` is additionally dividend-adjusted. Neither is what a human would
have paid. The system therefore carries **three explicitly named series**, with
bare `close` banned from the schema so no module can grab "the" close price and
get the wrong one:

| Series | Meaning | Used by |
|---|---|---|
| `*_raw` | as-traded that session | backtester execution **only** |
| `*_split` | split-adjusted, no dividends | charting, volume normalisation |
| `adj_close` | total return | features and labels **only** |

`adj_close` is retroactively revised by every new dividend. That is precisely
why filling an order at it would be look-ahead bias — transacting at a price
that only exists in hindsight and will change again next quarter.

**Limitation.** The as-traded series is *reconstructed* by reversing the split
adjustment. It is exact when the split history is complete and wrong in
proportion to any split the provider missed. The `ABNORMAL_JUMP` check exists to
catch that case.

---

## 5. Feature engineering

45 features across five modules: technical indicators, returns and momentum,
volatility (including Parkinson and Garman-Klass range estimators), volume, and
regime.

**The causal rule.** Every feature at time *t* uses only rows `0..t`. No
`shift(-n)`, no `center=True`, no `bfill`, and — the subtlest — no full-sample
statistics.

**The trap that full-sample statistics set.** The natural way to write "is
volatility high right now" is:

```python
df["vol_percentile"] = df["volatility_20"].rank(pct=True)   # WRONG
```

That ranks a day in 2016 against 2020's volatility. There is no `shift(-1)`, no
future column, nothing that looks wrong. The same trap swallows z-scores against
`series.mean()`, min-max scaling, and `pd.qcut` — all standard preprocessing,
all leaky. Everything is replaced with expanding-window equivalents.
Cross-sectional ranks group by date, which is safe because those observations
share a timestamp.

A CI gate walks the AST for these constructs. It parses rather than greps,
because the grep version false-positived on the docstrings that warn about them.

---

## 6. Leakage prevention and the timing contract

The planning document specified two things that cannot both hold:

```
target_return_1d[t] = adj_close[t+1] / adj_close[t] - 1
"Signals at t execute at the next trading session."
```

If the decision is made after the close of *t* and the order fills during *t+1*,
the close-to-close move from *t* is already partly gone. Training on that label
teaches the model to forecast a return the strategy cannot capture.

The contract now used:

```
session t              session t+1          session t+2
close: features        open: ENTRY          open: EXIT
       decision made

tradeable_return_1d[t] = adj_open[t+2] / adj_open[t+1] - 1
```

Both labels are computed. `tradeable_return_1d` trains the model;
`research_return_1d` (the close-to-close version) is a diagnostic. **On
synthetic data the two correlate at 0.001** — they are effectively different
targets, and that number is the size of the trap.

**The leakage test.** `tests/test_leakage.py` mutates every price after a cut
point (50% drop, fresh noise, 7× volume), recomputes, and asserts all 45
features at or before the cut are bit-identical. Three cut points. Critically,
`test_the_test_can_actually_fail` plants a known-leaky feature and confirms the
detector fires — without it, a passing leakage suite proves nothing.

---

## 7. Validation methodology

**Purged, embargoed walk-forward, keyed on dates.**

`sklearn.TimeSeriesSplit` is unusable here for two independent reasons. It
splits on row position — but this is a panel, so consecutive rows are different
symbols on the same day, and a positional cut puts two stocks from the same
session on opposite sides of the boundary. And it leaves no gap at all.

**Purge and embargo are different mechanisms.** Purge removes training rows
whose *label* reaches forward into validation. Embargo removes training rows
immediately *after* validation, where a 20-day rolling feature still shares 19
of its 20 observations with data the model was scored on. Walk-forward needs
both, because each fold's validation becomes the next fold's training data.

**Purge must be `horizon + 1`.** The extra session is the execution offset: a
row at *t* is labelled with `open[t+1]..open[t+1+h]`, so it encodes information
through *t+1+h*. Sizing the purge to the horizon alone leaves exactly one
contaminated row at every fold boundary — small, invisible in the metrics, and
systematically favourable.

**The final-test lock.** `split_development()` is the only supported path to
fittable data and structurally cannot return a locked row. `FinalTestLock`
fingerprints the window, refuses a second unlock, and detects tampering. It is
not a security boundary; it makes the wrong thing require deliberate effort and
leave a trace.

---

## 8. Base models

Direction: L2 logistic regression and HistGradientBoostingClassifier.
Return: Ridge and HistGradientBoostingRegressor. All wrapped in sklearn
`Pipeline`s so preprocessing is fitted per fold.

**Preprocessing must be inside the fold.** `StandardScaler().fit(X)` before
splitting is the most common leak surviving in otherwise careful pipelines, and
the purged splitter is powerless against it — the scaler already read the
validation window and the locked test.

**Measured result on random-walk data**, where truth is AUC 0.50:

| model | ROC-AUC | fold std | accuracy | majority | lift | BSS |
|---|---|---|---|---|---|---|
| direction_hgb | 0.5157 | 0.0137 | 0.5167 | 0.5240 | **−0.0073** | −0.0024 |
| direction_logistic | 0.5086 | 0.0091 | 0.5078 | 0.5240 | **−0.0161** | −0.0052 |
| baseline_majority | 0.4982 | — | 0.5240 | 0.5240 | 0.0000 | −0.0009 |

Both models are **less accurate than a constant predictor**. This is the correct
outcome on noise, and it is the reference against which real data must be read.

**Baselines are structural.** Every classification report carries
`majority_accuracy` and `skill_vs_majority`. Every AUC carries `auc_stderr` —
about 0.024 at 600 observations, which means 0.53 and 0.55 are not different
numbers. `flag_suspicious()` warns above 0.58 and escalates past 0.65 with a
pointer to the leakage suite. Written before any real data was seen, so a
flattering result cannot be reinterpreted afterwards.

---

## 9. Stacking and calibration

**The ensemble is learned, not hand-weighted.** The planning document proposed
fixed weights (0.35 direction, 0.25 price, 0.20 trend, 0.20 sentiment) to be
"validated later". The problem is not the numbers; it is that there is no
principled way to choose them, and "validate later" in practice means tuning
them against whatever data remains.

A meta-model learns the weights under the same purged validation as everything
else, and its coefficients are inspectable — so the interpretability the fixed
weights were meant to provide is retained, but earned.

**In-sample calibration is circular.** Fitting a calibrator on OOF predictions
and measuring calibration on those same rows produces a near-zero ECE for *any*
model: isotonic regression maps any monotone input onto the observed frequencies
of the rows it was fitted on. `calibrate_out_of_fold()` walks forward in blocks
instead — block *k* transformed by a calibrator fitted on blocks `0..k-1`.

Two distinct artifacts, routinely collapsed into one:
`calibrate_out_of_fold()` **measures** generalisation;
`fit_production_calibrator()` is what **ships**.

**Measured result.**

| stage | ROC-AUC |
|---|---|
| base logistic | 0.5097 |
| base hgb | 0.5175 |
| stacked meta-model | 0.5069 |

Stacking is worse than the better base model. Calibration moves ECE 0.0434 →
0.0405 while making Brier and MCE worse. The Brier decomposition diagnoses it:
**reliability 0.00263, resolution 0.00111** — the forecast carries less
information than it has calibration error, and post-hoc calibration cannot
manufacture information.

---

## 10. Decision engine

**V1 semantics (locked).** BUY opens or increases a long; HOLD maintains
whatever is held; SELL exits or reduces; SELL while flat stays flat; a rejected
or stale input preserves the existing position.

The last rule matters most operationally. Liquidating on a data outage turns an
outage into a realised loss *plus* a round-trip cost; holding costs only the
missed opportunity. Risk gates run *before* signal generation, so a rejected
input never produces a tradeable number.

HOLD on an open position returns a NaN target weight, not 0.0. Returning zero
would liquidate everything on every quiet day.

**Two gates for a BUY.** Probability must clear the buy threshold **and**
expected return must clear the cost floor. The second makes the first
meaningful: a 0.72 probability on a 3 bps move is a confident prediction of a
losing trade. Confidence about direction says nothing about magnitude.

**The cost floor is derived, never tuned.** The planning document specified
`min_expected_return = 0.0005` — 5 bps against a 26 bps round trip. Every trade
clearing that gate loses 21 bps on costs alone. It is now
`round_trip × margin`, because it is a fact about the broker rather than a free
parameter.

**Thresholds are derived out-of-fold.** Searching thresholds on the same
predictions used to report performance is selection bias, identical in kind to
in-sample calibration.

---

## 11. Cost feasibility

The central analysis.

```
commission 3 + spread 5 + slippage 5 = 13 bps per side
round trip                           = 26 bps
```

A one-day holding period makes every signal a full round trip; nothing
amortises. Expected edge from going long the top decile is approximately
`IC × σ_daily × 1.755`:

| IC | expected edge | needed |
|---|---|---|
| 0.02 | 5.3 bps | 26 bps |
| 0.03 | 7.9 bps | 26 bps |
| 0.05 | 13.2 bps | 26 bps |
| 0.10 | 26.3 bps | 26 bps |

**Breakeven IC is 0.099. Measured IC is 0.027 ± 0.055.**

The available levers are a longer holding period (costs amortise over more days
of edge), lower costs, or accepting the system as decision support rather than
a strategy. This project takes the third position explicitly.
`cost_feasibility()` computes and prints the verdict before any optimisation
runs, so the conclusion cannot be skipped past.

---

## 12. Backtesting and portfolio accounting

**Event-based, next-session execution, as-traded prices.** Session order:
corporate actions → execute yesterday's orders at today's open → mark to market
and reconcile → decide from today's data, for tomorrow. Decisions last is what
makes same-session fills mechanically impossible.

**Reconciliation that can actually fail.** The usual check is a tautology.
Equity is computed twice here from disjoint state:

- **Path A** — `cash + Σ(shares × price)`, driven by the cash balance
- **Path B** — `initial + realised + unrealised + dividends − costs`, driven by
  accumulators that never read `cash`

Three tests plant bugs path A cannot see — an uncharged commission, an
unrecorded dividend, phantom shares — and confirm the mismatch is caught.
Measured max reconciliation error across a 750-session run: **0.00000000**.

**An honest limitation.** Reconciliation cannot detect a *missing* split: both
paths share the share count and still agree. Detection lives upstream, in the
`ABNORMAL_JUMP` check. Asserted as a test so it stays documented.

**Development-set result:**

| | return | CAGR | Sharpe | max DD | cost drag | turnover |
|---|---|---|---|---|---|---|
| TradeMind | +16.5% | +5.3% | 0.99 ± 0.58 | −6.7% | 7.9% | 20.4× |
| buy & hold | +47.4% | +13.9% | 1.56 ± 0.58 | −9.3% | 0.1% | — |

The strategy needed **+24.4% gross to deliver +16.5% net**. Buy-and-hold beat it
on return, Sharpe, drawdown, and cost while trading twice.

Note the standard errors: 0.99 ± 0.58 over three years. Any Sharpe reported on a
sample this short carries the same ±0.58 and is not distinguishable from 0.4.

Benchmarks pay identical costs. A cost-free benchmark against a cost-charged
strategy is the easiest way to make a losing strategy look competitive.

---

## 13. Risk management

Position sizing offers fixed, volatility-targeted, and confidence-scaled
methods. Volatility targeting scales by `reference_volatility / symbol_volatility`
so each position contributes similar risk.

**Kelly is deliberately absent.** It assumes the edge estimate is correct, and
sizes most aggressively exactly where a noisy edge is most wrong. Half-Kelly
does not fix a wrong sign.

Risk gates: staleness (bars older than 3 sessions), incomplete features,
non-finite probabilities, and a position cap that blocks new entries but never
force-closes — a limit breach must not become an unplanned liquidation.

---

## 14. Monitoring, drift, and retraining governance

**Naive drift monitoring would fire on 90% of days.** 45 features, KS at
p<0.05, checked daily: 2.2 expected false alarms per day, 47 per month. A
monitor that cries wolf nine days in ten trains the team to dismiss it.
Responses: PSI (effect size) as the primary signal, persistence over three
consecutive checks, and Benjamini–Hochberg FDR on the p-values that remain.

**Performance monitoring cannot detect this model's degradation.**

| window | observations | AUC std err | detectable drop |
|---|---|---|---|
| 7d | 105 | 0.195 | 0.39 |
| 30d | 450 | 0.094 | 0.19 |
| 90d | 1350 | 0.054 | 0.11 |

The model's entire edge is ~0.02. It could lose all of it and the monitor would
report noise. The governance consequence is stated explicitly in the policy: **an
unchanged metric is not evidence of health**, because it would look identical if
the model had failed completely.

**The four governance rules.** Drift alone never retrains. A data-quality ERROR
blocks retraining regardless of severity elsewhere — when performance collapses
*and* ingestion is erroring, the data is the likelier cause, and retraining on
it bakes the corruption in. Insufficient new observations means wait. A failed
candidate has no path to production, enforced at the promotion decision *and*
independently at the registry.

**Diagnosis, not just detection.** Performance down with drift → likely regime
change, retraining should help. Performance down *without* drift → the
feature-target relationship changed, and retraining on the same features may not
help.

---

## 15. Final test

**Status: ORIGINAL LOCK INVALIDATED. Not evaluated.**

The Phase 3 final-test window was never successfully unlocked. Its sealed
fingerprint (`52439e3babd08f4b`) does not match the regenerated historical
feature snapshot (`b9e7813b7984bdab`), and no archive containing the exact
original feature data was available. The original fingerprint remains
unchanged, no access was recorded, and no evaluation evidence file exists.

**No result is reported here because no result exists.** The original result is
unrecoverable without the exact snapshot. A replacement future holdout is
frozen for 2026-09-02 through 2027-01-29 at code commit `8a69605`. It must be
reported as v2, never as the original Phase 3 test, and must not be inspected
before the period completes. After that date, evaluation remains one command:

```bash
python scripts/final_evaluation.py --confirm
```

The script refuses to run twice, refuses to run from a dirty working tree, and
refuses to run before thresholds have been derived and locked on development
data. On success it writes the equity curve, trades, metrics, and an evidence
file recording that the unlock occurred.

**Why this section is empty rather than filled.** The original planning document
for TradeMind AI recorded a "locked final-test result" of +0.48% cumulative
return at Sharpe 1.45, with a maximum drawdown of −0.31%, described as
"immutable benchmark results". No such code existed. The numbers were
placeholders that had acquired the grammar of findings.

Writing a plausible number here would reproduce exactly the failure the previous
fourteen sections were built to prevent. The section stays empty until the
evaluation runs against real ingested data.

**Prediction, recorded in advance.** Given a development-set IC of 0.027 against
a breakeven of 0.099, the expected final-test outcome is that the strategy
underperforms buy-and-hold and fails to clear costs. Recording this before the
unlock is itself a control: it makes a surprisingly good result something to
investigate rather than celebrate.

---

## 16. Limitations

- **Survivorship bias.** Today's large-caps; delisted names silently excluded.
  Every backtest figure is optimistic by an unmeasured amount.
- **Reconstructed execution prices**, exact only if the provider's split history
  is complete.
- **Approximate trading calendar** without `pandas-market-calendars`.
- **Single market, daily bars, long only.** No shorting, intraday, or leverage.
- **Monitoring is blind to this model's degradation** (§14).
- **Reconciliation cannot detect a missing split** (§12).
- **Synthetic data for development.** Results in §8–§12 are measured on
  random-walk synthetic series, which establishes that the pipeline finds
  nothing in noise. Real-data figures require running the pipeline.
- **No live trading.** Nothing here has traded real money.

---

## 17. Future work

**Most likely to change the conclusion**, in order:

1. **Longer holding period.** The single highest-leverage change. A 20-session
   horizon amortises the round trip over 20 days of edge, dropping breakeven IC
   from 0.099 to about 0.022 — within reach of a real signal.
2. **Cross-sectional rather than directional.** Long-short ranking within the
   universe removes market beta and is where daily equity signals have
   historically been strongest.
3. **Wider universe.** 15 symbols limits cross-sectional features and makes
   several monitoring windows unreportable.

**Would improve the system without changing the conclusion:**

4. Point-in-time universe construction, to remove survivorship bias.
5. Paid data with genuine as-traded prices and audited corporate actions.
6. News and sentiment features — deliberately deferred, since adding an NLP
   pipeline before the numeric path works would be premature.
7. Intraday execution modelling, including participation-rate slippage.

---

## Appendix: reproducing this report

```bash
git clone <repository>
cd TradeMind-AI
pip install -e ".[dev]"

make ci                          # lint, protected invariants, full suite
python main.py init
python main.py ingest            # requires network
python main.py features
python main.py validate          # creates the final-test lock
python main.py train
python main.py ensemble
python main.py thresholds        # write the derived values into config
python main.py backtest

# Once, and only once:
python scripts/final_evaluation.py --confirm
```

Every run records provenance: git commit and dirty flag, config hash,
environment hash, package versions, and random seed. A run from a dirty working
tree is flagged as unreproducible, because the recorded commit then points at
code that never ran.
