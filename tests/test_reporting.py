"""Reporting tests.

The dashboard's *reasoning* is tested here; the Streamlit files only draw. The
tests that matter most assert that inconvenient findings are surfaced rather
than smoothed over — an infeasible strategy must render as BAD, an AUC within
noise of chance must not render as skill, and the overview must not open with
the equity curve.
"""

from __future__ import annotations

import pandas as pd

from trademind.reporting import (
    Metric,
    Verdict,
    build_accounting_panel,
    build_cost_feasibility_panel,
    build_data_quality_panel,
    build_detectability_panel,
    build_drift_panel,
    build_overview,
    build_performance_panel,
    build_signals_table,
    build_skill_panel,
    format_sample_caveat,
    signal_counts,
)

# =====================================================================
# Ordering — the honest findings come first
# =====================================================================


def test_overview_leads_with_cost_feasibility():
    """A dashboard that opens with a rising curve invites you to stop reading."""
    panels = build_overview(
        {"round_trip_cost": 0.0026, "breakeven_ic": 0.099, "observed_ic": 0.027}
    )
    assert panels[0].title == "Cost feasibility"


def test_overview_ends_with_performance():
    panels = build_overview({})
    assert panels[-1].title == "Performance"


def test_skill_is_shown_before_performance():
    titles = [p.title for p in build_overview({})]
    assert titles.index("Model skill") < titles.index("Performance")


def test_overview_survives_an_empty_state():
    """The dashboard must work before anything has been computed."""
    panels = build_overview({})
    assert len(panels) == 5
    assert all(p.verdict in (Verdict.UNKNOWN, Verdict.GOOD) for p in panels)


# =====================================================================
# Cost feasibility
# =====================================================================


def test_infeasible_strategy_renders_as_bad():
    panel = build_cost_feasibility_panel(0.0026, 0.099, 0.027, holding_days=1)

    assert panel.verdict is Verdict.BAD
    assert "cannot clear its own costs" in panel.headline
    assert any("No threshold fixes" in c for c in panel.caveats)


def test_feasible_strategy_renders_as_good():
    panel = build_cost_feasibility_panel(0.0026, 0.05, 0.12)
    assert panel.verdict is Verdict.GOOD
    assert not panel.caveats


def test_unmeasured_ic_renders_as_unknown():
    panel = build_cost_feasibility_panel(0.0026, 0.099, None)
    assert panel.verdict is Verdict.UNKNOWN


def test_uncomputed_feasibility_points_at_the_command():
    panel = build_cost_feasibility_panel(None, None, None)
    assert panel.verdict is Verdict.UNKNOWN
    assert "main.py thresholds" in panel.headline


# =====================================================================
# Model skill
# =====================================================================


def test_auc_within_noise_of_chance_is_not_skill():
    """0.515 ± 0.024 is not a result."""
    panel = build_skill_panel(
        auc=0.515, auc_stderr=0.024, majority_accuracy=0.524, accuracy=0.517
    )
    assert panel.verdict is Verdict.CONCERN
    assert "within sampling error of chance" in panel.headline


def test_negative_lift_renders_as_bad():
    """Worse than a constant predictor must not read as a mild result."""
    panel = build_skill_panel(
        auc=0.60, auc_stderr=0.01, majority_accuracy=0.524, accuracy=0.510
    )
    assert panel.verdict is Verdict.BAD
    assert "below" in panel.headline


def test_implausibly_high_auc_is_flagged_as_leakage():
    panel = build_skill_panel(auc=0.72, auc_stderr=0.01, majority_accuracy=0.52, accuracy=0.68)
    assert panel.verdict is Verdict.CONCERN
    assert "leakage" in panel.headline.lower()


def test_genuine_skill_renders_as_good():
    panel = build_skill_panel(
        auc=0.56, auc_stderr=0.010, majority_accuracy=0.52, accuracy=0.55
    )
    assert panel.verdict is Verdict.GOOD


def test_skill_panel_always_states_the_baseline_caveat():
    panel = build_skill_panel(0.56, 0.01, 0.52, 0.55)
    assert any("52%" in c for c in panel.caveats)


# =====================================================================
# Accounting and data quality
# =====================================================================


def test_clean_reconciliation_is_good():
    assert build_accounting_panel(0.0).verdict is Verdict.GOOD


def test_reconciliation_mismatch_is_bad():
    panel = build_accounting_panel(15.3)
    assert panel.verdict is Verdict.BAD
    assert "cannot be trusted" in panel.headline


def test_data_quality_errors_mention_the_retraining_block():
    panel = build_data_quality_panel(n_errors=2, n_warnings=5)
    assert panel.verdict is Verdict.BAD
    assert "Retraining is blocked" in panel.headline


def test_warnings_alone_do_not_block():
    panel = build_data_quality_panel(n_errors=0, n_warnings=7)
    assert panel.verdict is Verdict.CONCERN
    assert "not blocked" in panel.headline


def test_clean_ingestion_is_good():
    assert build_data_quality_panel(0, 0).verdict is Verdict.GOOD


# =====================================================================
# Performance
# =====================================================================


def test_underperforming_the_benchmark_renders_as_bad():
    panel = build_performance_panel(
        {
            "total_return": 0.165,
            "sharpe": 0.99,
            "sharpe_stderr": 0.58,
            "cost_drag": 0.079,
            "n_sessions": 750,
            "annual_turnover": 20.4,
        },
        benchmark={"total_return": 0.474},
    )
    assert panel.verdict is Verdict.BAD
    assert "underperforms doing nothing" in panel.headline


def test_costs_exceeding_return_is_flagged():
    panel = build_performance_panel(
        {"total_return": 0.02, "cost_drag": 0.08, "n_sessions": 750}
    )
    assert panel.verdict is Verdict.CONCERN
    assert "costs" in panel.headline


def test_high_turnover_earns_a_caveat():
    panel = build_performance_panel(
        {"total_return": 0.10, "n_sessions": 1500, "annual_turnover": 25.0}
    )
    assert any("slippage" in c for c in panel.caveats)


def test_short_sample_earns_a_caveat():
    panel = build_performance_panel({"total_return": 0.10, "n_sessions": 300})
    assert any("not reliably estimable" in c for c in panel.caveats)


def test_long_sample_needs_no_caveat():
    assert format_sample_caveat(252 * 6) == ""


# =====================================================================
# Drift and detectability
# =====================================================================


def test_drift_panel_shows_the_naive_count_beside_the_controlled_one():
    panel = build_drift_panel(
        {
            "n_significant": 0,
            "n_moderate": 3,
            "max_psi": 0.12,
            "n_ks_naive": 11,
            "n_ks_fdr": 1,
        }
    )
    labels = [m.label for m in panel.metrics]
    assert any("naive" in label for label in labels)
    assert any("FDR" in label for label in labels)


def test_drift_panel_states_that_drift_alone_does_not_retrain():
    panel = build_drift_panel({"n_significant": 5, "n_ks_naive": 20, "n_ks_fdr": 3})
    assert any("never triggers retraining" in c for c in panel.caveats)


class FakeWindow:
    def __init__(self, days, n, mde, reportable=True):
        self.window_days = days
        self.n_observations = n
        self.minimum_detectable_effect = mde
        self.reportable = reportable


def test_detectability_panel_states_the_blindness():
    panel = build_detectability_panel(
        [
            FakeWindow(7, 105, 0.39),
            FakeWindow(30, 450, 0.19),
            FakeWindow(90, 1350, 0.11),
        ]
    )
    assert panel.verdict is Verdict.CONCERN
    assert any("entire edge is about 0.02" in c for c in panel.caveats)
    assert any("not evidence of health" in c for c in panel.caveats)


def test_detectability_handles_thin_windows():
    panel = build_detectability_panel([FakeWindow(7, 20, 0.9, reportable=False)])
    assert panel.verdict is Verdict.CONCERN
    assert "enough resolved predictions" in panel.headline


# =====================================================================
# Signals
# =====================================================================


def test_signals_table_keeps_rejections():
    """A BELOW_COST_FLOOR rejection is the system working; hiding it hides that."""
    decisions = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "signal": ["BUY", "HOLD"],
            "calibrated_probability": [0.61, 0.72],
            "rejected": [False, True],
            "reject_reason": ["NONE", "BELOW_COST_FLOOR"],
        }
    )
    table = build_signals_table(decisions)

    assert len(table) == 2
    assert "BELOW_COST_FLOOR" in table["reject_reason"].to_list()


def test_signals_table_sorts_by_probability():
    decisions = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "signal": ["HOLD"] * 3,
            "calibrated_probability": [0.40, 0.80, 0.60],
        }
    )
    assert build_signals_table(decisions)["symbol"].to_list() == ["B", "C", "A"]


def test_signal_counts_include_rejections():
    decisions = pd.DataFrame(
        {
            "signal": ["BUY", "BUY", "HOLD", "SELL"],
            "rejected": [False, True, True, False],
        }
    )
    counts = signal_counts(decisions)
    assert counts["BUY"] == 2
    assert counts["REJECTED"] == 2


def test_empty_decisions_render_empty():
    assert build_signals_table(pd.DataFrame()).empty
    assert signal_counts(pd.DataFrame()) == {}


# =====================================================================
# Metric formatting
# =====================================================================


def test_metric_carries_its_stderr():
    m = Metric("AUC", 0.5157, stderr=0.0137)
    assert "±" in m.format_full()
    assert "0.0137" in m.format_full()


def test_metric_carries_its_baseline():
    m = Metric("AUC", 0.5157, baseline=0.5, baseline_label="chance")
    assert "chance" in m.format_full()


def test_gap_within_noise_is_detected():
    """The sqrt(2) factor on the difference standard error."""
    assert Metric("AUC", 0.515, stderr=0.024, baseline=0.5).within_noise
    assert not Metric("AUC", 0.62, stderr=0.010, baseline=0.5).within_noise


def test_missing_value_reads_as_not_available():
    m = Metric("AUC", None)
    assert not m.available
    assert m.format_value() == "n/a"


def test_bps_formatting():
    assert Metric("cost", 0.0026, unit="bps").format_value() == "+26.0 bps"


def test_panel_render_includes_caveats():
    panel = build_cost_feasibility_panel(0.0026, 0.099, 0.027)
    text = panel.render()
    assert "✗" in text
    assert "No threshold fixes" in text
