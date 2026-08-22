"""Panel construction: what the dashboard says, separated from how it looks.

Every panel is a pure function from data to a ``Panel``, so the dashboard's
*reasoning* is unit-testable. Putting this logic inside Streamlit callbacks
would make it untestable, and untestable presentation logic is exactly where
inconvenient findings quietly stop being displayed.

## Ordering is a design decision

``build_overview`` returns panels in a fixed order, and the equity curve is not
first. The order is:

1. **Cost feasibility** — can the strategy clear its own costs at all?
2. **Model skill vs baseline** — does the model beat a constant predictor?
3. **Accounting integrity** — do the books reconcile?
4. **Data quality** — is the input trustworthy?
5. **Performance** — the equity curve, last.

A dashboard that opens with a rising equity curve invites the viewer to stop
reading. Opening with the feasibility verdict means the curve is read in the
right context, which for this system is: the strategy underperforms
buy-and-hold and the reason is arithmetic, not tuning.
"""

from __future__ import annotations

import math

import pandas as pd

from .formatting import Metric, Panel, Verdict, format_sample_caveat, pluralise


def build_cost_feasibility_panel(
    round_trip_cost: float | None,
    breakeven_ic: float | None,
    observed_ic: float | None,
    holding_days: int = 1,
) -> Panel:
    """The first thing anyone should see."""
    if round_trip_cost is None or breakeven_ic is None:
        return Panel(
            "Cost feasibility", Verdict.UNKNOWN,
            "Not yet computed. Run `main.py thresholds`.",
        )

    feasible = (
        observed_ic is not None
        and math.isfinite(observed_ic)
        and observed_ic >= breakeven_ic
    )

    if observed_ic is None or not math.isfinite(observed_ic):
        verdict, headline = Verdict.UNKNOWN, (
            f"Breakeven IC is {breakeven_ic:.3f}; no measured IC yet."
        )
    elif feasible:
        verdict, headline = Verdict.GOOD, (
            f"Measured IC {observed_ic:.3f} clears the {breakeven_ic:.3f} "
            "breakeven."
        )
    else:
        verdict, headline = Verdict.BAD, (
            f"Measured IC {observed_ic:.3f} is below the {breakeven_ic:.3f} "
            f"breakeven. At a {holding_days}-session holding period the "
            "strategy cannot clear its own costs at any threshold."
        )

    return Panel(
        "Cost feasibility", verdict, headline,
        metrics=(
            Metric("Round-trip cost", round_trip_cost, unit="bps"),
            Metric("Breakeven IC", breakeven_ic, decimals=3),
            Metric("Measured IC", observed_ic, decimals=3,
                   baseline=breakeven_ic, baseline_label="breakeven"),
            Metric("Holding period", float(holding_days), unit="x", decimals=0),
        ),
        caveats=() if feasible else (
            "No threshold fixes an infeasible strategy — the search is choosing "
            "among trades that all lose money on costs.",
            "The levers are a longer holding period, lower costs, or treating "
            "this as decision support rather than a strategy.",
        ),
    )


def build_skill_panel(
    auc: float | None,
    auc_stderr: float | None,
    majority_accuracy: float | None,
    accuracy: float | None,
) -> Panel:
    """Does the model beat a constant predictor? Usually the honest answer is no."""
    if auc is None or not math.isfinite(auc):
        return Panel("Model skill", Verdict.UNKNOWN, "No evaluated model yet.")

    lift = (
        accuracy - majority_accuracy
        if accuracy is not None and majority_accuracy is not None else None
    )
    auc_metric = Metric("ROC-AUC", auc, stderr=auc_stderr, baseline=0.5,
                        baseline_label="chance")

    if auc_metric.within_noise:
        verdict = Verdict.CONCERN
        headline = (
            f"AUC {auc:.4f} is within sampling error of chance "
            f"(± {auc_stderr:.4f}). No demonstrated discrimination."
        )
    elif lift is not None and lift < 0:
        verdict = Verdict.BAD
        headline = (
            f"AUC {auc:.4f}, but accuracy is {abs(lift):.4f} *below* the "
            "majority-class baseline."
        )
    elif auc > 0.58:
        verdict = Verdict.CONCERN
        headline = (
            f"AUC {auc:.4f} is above the plausible range for next-day equity "
            "direction. Suspect leakage before skill."
        )
    else:
        verdict = Verdict.GOOD
        headline = f"AUC {auc:.4f}, measurably above chance."

    return Panel(
        "Model skill", verdict, headline,
        metrics=(
            auc_metric,
            Metric("Accuracy", accuracy, decimals=4,
                   baseline=majority_accuracy, baseline_label="majority"),
            Metric("Lift vs majority", lift, decimals=4,
                   note="Negative means a constant predictor does better."
                   if lift is not None and lift < 0 else ""),
        ),
        caveats=(
            "52% of sessions close up, so 52% accuracy is what a constant "
            "achieves. Read accuracy against the majority baseline, never alone.",
        ),
    )


def build_accounting_panel(max_reconciliation_error: float | None) -> Panel:
    """Do the two independent equity calculations agree?"""
    if max_reconciliation_error is None:
        return Panel("Accounting integrity", Verdict.UNKNOWN, "No backtest run yet.")

    ok = max_reconciliation_error < 0.01
    return Panel(
        "Accounting integrity",
        Verdict.GOOD if ok else Verdict.BAD,
        (
            "Balance-sheet and flow equity agree to within a paisa across every "
            "session."
            if ok else
            f"Equity paths disagree by up to {max_reconciliation_error:,.4f}. "
            "The backtest numbers cannot be trusted."
        ),
        metrics=(
            Metric("Max reconciliation error", max_reconciliation_error,
                   decimals=8),
        ),
    )


def build_data_quality_panel(n_errors: int, n_warnings: int) -> Panel:
    """Ingestion health — and the retraining gate it controls."""
    if n_errors:
        return Panel(
            "Data quality", Verdict.BAD,
            f"{pluralise(n_errors, 'unresolved error')}. Retraining is blocked "
            "until these are fixed.",
            metrics=(
                Metric("Errors", float(n_errors), decimals=0),
                Metric("Warnings", float(n_warnings), decimals=0),
            ),
            caveats=(
                "A data-quality ERROR blocks retraining regardless of how severe "
                "other signals look: retraining on bad data bakes it in.",
            ),
        )
    if n_warnings:
        return Panel(
            "Data quality", Verdict.CONCERN,
            f"{pluralise(n_warnings, 'warning')} logged. Ingestion is not blocked.",
            metrics=(Metric("Warnings", float(n_warnings), decimals=0),),
        )
    return Panel("Data quality", Verdict.GOOD, "No errors or warnings.")


def build_performance_panel(
    metrics: dict[str, float] | None,
    benchmark: dict[str, float] | None = None,
) -> Panel:
    """The equity curve — deliberately last in the overview."""
    if not metrics:
        return Panel("Performance", Verdict.UNKNOWN, "No backtest run yet.")

    total = metrics.get("total_return")
    sharpe = metrics.get("sharpe")
    stderr = metrics.get("sharpe_stderr")
    cost_drag = metrics.get("cost_drag")
    bench_return = benchmark.get("total_return") if benchmark else None

    if bench_return is not None and total is not None and total < bench_return:
        verdict = Verdict.BAD
        headline = (
            f"{total:+.2%} against buy-and-hold's {bench_return:+.2%}. "
            "The strategy underperforms doing nothing."
        )
    elif cost_drag is not None and total is not None and cost_drag > abs(total):
        verdict = Verdict.CONCERN
        headline = (
            f"{total:+.2%} net, with {cost_drag:.2%} lost to costs — costs "
            "exceed the return."
        )
    else:
        verdict = Verdict.NEUTRAL
        headline = f"{total:+.2%} total return." if total is not None else "n/a"

    caveats = []
    sample = format_sample_caveat(int(metrics.get("n_sessions", 0)))
    if sample:
        caveats.append(sample)
    if metrics.get("annual_turnover", 0) > 10:
        caveats.append(
            f"{metrics['annual_turnover']:.0f}x annual turnover: the result is "
            "highly sensitive to the slippage and spread assumptions."
        )

    return Panel(
        "Performance", verdict, headline,
        metrics=(
            Metric("Total return", total, unit="%"),
            Metric("CAGR", metrics.get("cagr"), unit="%"),
            Metric("Sharpe", sharpe, stderr=stderr, decimals=2),
            Metric("Max drawdown", metrics.get("max_drawdown"), unit="%"),
            Metric("Cost drag", cost_drag, unit="%",
                   note="Gross return needed just to break even: "
                        f"{metrics.get('gross_return_required', float('nan')):+.2%}"
                   if cost_drag else ""),
            Metric("Annual turnover", metrics.get("annual_turnover"), unit="x"),
            Metric("Buy & hold", bench_return, unit="%"),
        ),
        caveats=tuple(caveats),
    )


def build_overview(state: dict) -> list[Panel]:
    """The overview page, in the order that keeps the numbers readable.

    Feasibility and skill come before the equity curve on purpose.
    """
    return [
        build_cost_feasibility_panel(
            state.get("round_trip_cost"), state.get("breakeven_ic"),
            state.get("observed_ic"), state.get("holding_days", 1),
        ),
        build_skill_panel(
            state.get("auc"), state.get("auc_stderr"),
            state.get("majority_accuracy"), state.get("accuracy"),
        ),
        build_accounting_panel(state.get("max_reconciliation_error")),
        build_data_quality_panel(
            state.get("n_dq_errors", 0), state.get("n_dq_warnings", 0),
        ),
        build_performance_panel(
            state.get("performance"), state.get("benchmark"),
        ),
    ]


def build_drift_panel(summary: dict[str, float] | None) -> Panel:
    """Drift, with the naive count shown beside the controlled one."""
    if not summary:
        return Panel("Feature drift", Verdict.UNKNOWN, "No drift computed yet.")

    n_sig = int(summary.get("n_significant", 0))
    naive = int(summary.get("n_ks_naive", 0))
    controlled = int(summary.get("n_ks_fdr", 0))

    verdict = Verdict.CONCERN if n_sig else Verdict.GOOD
    headline = (
        f"{pluralise(n_sig, 'feature')} with PSI above 0.25."
        if n_sig else "No feature shows significant distribution shift."
    )

    return Panel(
        "Feature drift", verdict, headline,
        metrics=(
            Metric("Significant (PSI > 0.25)", float(n_sig), decimals=0),
            Metric("Moderate (PSI > 0.10)",
                   summary.get("n_moderate"), decimals=0),
            Metric("Max PSI", summary.get("max_psi"), decimals=3),
            Metric("KS at p<0.05, naive", float(naive), decimals=0,
                   note="Uncorrected. With 45 features this fires on ~90% of "
                        "days from noise alone."),
            Metric("KS after FDR control", float(controlled), decimals=0),
        ),
        caveats=(
            "Drift alone never triggers retraining. A model can be healthy on "
            "shifted inputs; drift makes a performance problem interpretable.",
        ),
    )


def build_detectability_panel(windows: list) -> Panel:
    """What the monitoring layer can and cannot see."""
    if not windows:
        return Panel("Monitoring sensitivity", Verdict.UNKNOWN,
                     "No resolved predictions yet.")

    reportable = [w for w in windows if getattr(w, "reportable", False)]
    if not reportable:
        return Panel(
            "Monitoring sensitivity", Verdict.CONCERN,
            "No window has enough resolved predictions to report on.",
            metrics=tuple(
                Metric(f"{w.window_days}d observations",
                       float(w.n_observations), decimals=0)
                for w in windows
            ),
        )

    best = min(reportable, key=lambda w: w.minimum_detectable_effect)
    return Panel(
        "Monitoring sensitivity", Verdict.CONCERN,
        f"The most sensitive window can only detect AUC changes of "
        f"{best.minimum_detectable_effect:.3f} or larger.",
        metrics=tuple(
            Metric(f"{w.window_days}d detectable change",
                   w.minimum_detectable_effect if w.reportable else None,
                   decimals=3,
                   note=f"{w.n_observations:,} observations")
            for w in windows
        ),
        caveats=(
            "The model's entire edge is about 0.02. It could lose all of it and "
            "these windows would report noise.",
            "An unchanged performance metric is therefore not evidence of "
            "health.",
        ),
    )


def build_signals_table(decisions: pd.DataFrame) -> pd.DataFrame:
    """Today's decisions, rejections included.

    Showing only BUYs would hide the most informative rows: a rejection with
    reason ``BELOW_COST_FLOOR`` says the model was confident and the trade was
    still not worth making, which is the system working.
    """
    if decisions.empty:
        return pd.DataFrame()

    columns = [c for c in (
        "symbol", "signal", "calibrated_probability", "expected_return",
        "target_weight", "risk_bucket", "rejected", "reject_reason", "rationale",
    ) if c in decisions.columns]

    frame = decisions[columns].copy()
    if "calibrated_probability" in frame:
        frame = frame.sort_values("calibrated_probability", ascending=False)
    return frame.reset_index(drop=True)


def signal_counts(decisions: pd.DataFrame) -> dict[str, int]:
    if decisions.empty:
        return {}
    counts = decisions["signal"].value_counts().to_dict()
    if "rejected" in decisions.columns:
        counts["REJECTED"] = int(decisions["rejected"].sum())
    return {k: int(v) for k, v in counts.items()}
