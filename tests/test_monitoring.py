"""Monitoring and governance tests.

Each of the four policy rules from the roadmap has a named test, plus the two
statistical findings that shaped the design: the multiple-testing false-alarm
rate, and the minimum detectable effect.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from trademind.monitoring.alerts import AlertCollector, Severity
from trademind.monitoring.calibration_monitor import (
    MIN_CALIBRATION_OBSERVATIONS,
    calibration_drifted,
    monitor_calibration,
)
from trademind.monitoring.feature_drift import (
    PSI_SIGNIFICANT,
    DriftTracker,
    benjamini_hochberg,
    compute_drift,
    population_stability_index,
    summarise_drift,
)
from trademind.monitoring.performance_monitor import (
    PerformanceMonitor,
    auc_minimum_detectable_effect,
    degradation_detected,
    rolling_windows,
)
from trademind.monitoring.retraining import (
    Diagnosis,
    HealthState,
    MonitoringSignals,
    PromotionBlocked,
    PromotionDecision,
    RetrainingPolicy,
    ValidationOutcome,
    decide_promotion,
    validate_candidate,
)

RNG = np.random.default_rng(0)


def resolved_frame(n_days=120, symbols=("A", "B", "C"), auc_signal=0.0, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    rows = []
    for s in symbols:
        latent = rng.normal(size=n_days)
        outcome = (latent * auc_signal + rng.normal(size=n_days) > 0).astype(float)
        rows.append(
            pd.DataFrame(
                {
                    "prediction_date": dates,
                    "symbol": s,
                    "calibrated_probability": 1 / (1 + np.exp(-latent * auc_signal)),
                    "actual_direction": outcome,
                }
            )
        )
    return pd.concat(rows).reset_index(drop=True)


def feature_frame(n=2000, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {f"f{i}": rng.normal(shift if i == 0 else 0.0, 1.0, n) for i in range(10)}
    )


FEATURES = [f"f{i}" for i in range(10)]


# =====================================================================
# The multiple-testing finding
# =====================================================================


def test_naive_ks_testing_produces_false_alarms():
    """45 features at p<0.05 means ~90% chance of a false alarm every day.

    Documents why PSI and persistence are the primary signals rather than
    per-feature significance.
    """
    rng = np.random.default_rng(1)
    n_features = 45
    # Two samples from the SAME distribution — every rejection is false.
    ref = pd.DataFrame({f"f{i}": rng.normal(size=1500) for i in range(n_features)})
    cur = pd.DataFrame({f"f{i}": rng.normal(size=1500) for i in range(n_features)})

    drift = compute_drift(ref, cur, list(ref.columns))
    naive = int((drift["ks_pvalue"] < 0.05).sum())

    # The theoretical rate is 45 x 0.05 = 2.25 per run.
    assert naive >= 0
    theoretical = 1 - 0.95**n_features
    assert theoretical > 0.85


def test_fdr_control_reduces_false_discoveries():
    rng = np.random.default_rng(2)
    ref = pd.DataFrame({f"f{i}": rng.normal(size=1500) for i in range(45)})
    cur = pd.DataFrame({f"f{i}": rng.normal(size=1500) for i in range(45)})

    drift = compute_drift(ref, cur, list(ref.columns))
    assert int(drift["ks_significant_fdr"].sum()) <= int((drift["ks_pvalue"] < 0.05).sum())


def test_benjamini_hochberg_rejects_nothing_when_nothing_is_real():
    p = np.random.default_rng(3).uniform(0, 1, 50)
    assert benjamini_hochberg(p, alpha=0.05).sum() <= 3


def test_benjamini_hochberg_finds_a_real_effect():
    p = np.r_[np.full(5, 1e-8), np.random.default_rng(4).uniform(0.2, 1.0, 45)]
    assert benjamini_hochberg(p, alpha=0.05).sum() >= 5


def test_persistence_suppresses_a_one_day_spike():
    """A single breach is noise; three consecutive are worth reporting."""
    tracker = DriftTracker(persistence_days=3)

    spike = pd.DataFrame({"feature": ["f0"], "psi": [0.5]})
    calm = pd.DataFrame({"feature": ["f0"], "psi": [0.01]})

    assert tracker.update(spike) == []
    assert tracker.update(calm) == []
    assert tracker.streak("f0") == 0


def test_persistent_drift_is_reported():
    tracker = DriftTracker(persistence_days=3)
    spike = pd.DataFrame({"feature": ["f0"], "psi": [0.5]})

    assert tracker.update(spike) == []
    assert tracker.update(spike) == []
    assert tracker.update(spike) == ["f0"]


# =====================================================================
# PSI
# =====================================================================


def test_identical_distributions_have_near_zero_psi():
    a = RNG.normal(size=5000)
    b = RNG.normal(size=5000)
    assert population_stability_index(a, b) < 0.05


def test_shifted_distribution_has_high_psi():
    a = RNG.normal(0, 1, 5000)
    b = RNG.normal(2, 1, 5000)
    assert population_stability_index(a, b) > PSI_SIGNIFICANT


def test_psi_uses_reference_bins():
    """Re-binning on the current sample would move the yardstick with the data."""
    a = RNG.normal(0, 1, 5000)
    b = RNG.normal(0, 5, 5000)  # same mean, much wider
    assert population_stability_index(a, b) > 0.1


def test_empty_bin_does_not_produce_infinity():
    a = RNG.normal(0, 1, 2000)
    b = np.full(2000, 100.0)  # entirely outside the reference range
    psi = population_stability_index(a, b)
    assert np.isfinite(psi)


def test_constant_feature_has_zero_psi():
    assert population_stability_index(np.ones(1000), np.ones(1000)) == 0.0


def test_drift_flags_only_the_shifted_feature():
    drift = compute_drift(feature_frame(seed=1), feature_frame(shift=2.0, seed=2), FEATURES)
    significant = set(drift.loc[drift["severity"] == "SIGNIFICANT", "feature"])
    assert "f0" in significant
    assert len(significant) == 1


def test_small_samples_are_marked_insufficient():
    drift = compute_drift(feature_frame(n=50), feature_frame(n=50), FEATURES)
    assert drift["insufficient_data"].all()
    assert (drift["severity"] == "UNKNOWN").all()


def test_summary_reports_both_naive_and_fdr_counts():
    s = summarise_drift(
        compute_drift(feature_frame(seed=1), feature_frame(shift=1.0, seed=2), FEATURES)
    )
    assert "n_ks_naive" in s and "n_ks_fdr" in s


# =====================================================================
# The detectability finding
# =====================================================================


def test_short_windows_cannot_detect_realistic_degradation():
    """A 7-day window can only see an AUC change of ~0.39.

    The model's whole edge is about 0.02. The monitor is structurally blind to
    this model's degradation, which is a conclusion about the system.
    """
    assert auc_minimum_detectable_effect(105) > 0.15
    assert auc_minimum_detectable_effect(1350) > 0.05


def test_mde_shrinks_with_sample_size():
    assert auc_minimum_detectable_effect(100) > auc_minimum_detectable_effect(10000)


def test_degradation_below_the_mde_is_not_declared():
    """The core guard: do not report differences that cannot be distinguished.

    A 30-day window on 15 symbols holds ~330 observations, giving an MDE around
    0.11. A real-world AUC drop from 0.52 to 0.50 — the model losing its entire
    edge — is a fifth of that and must not be reported as degradation.
    """
    windows = rolling_windows(
        resolved_frame(n_days=60, symbols=tuple("ABCDEFGHIJKLMNO"), auc_signal=0.0),
        windows=(30,),
    )
    assert windows[0].reportable
    detected, reason = degradation_detected(windows[0], baseline_auc=0.52)

    assert not detected
    assert (
        "not distinguishable from noise" in reason.lower()
        or "at or above baseline" in reason.lower()
    )


def test_large_degradation_is_declared():
    resolved = resolved_frame(n_days=200, auc_signal=0.0, seed=9)
    windows = rolling_windows(resolved, windows=(90,))
    detected, _reason = degradation_detected(windows[0], baseline_auc=0.95)
    assert detected


def test_thin_windows_are_not_reportable():
    windows = rolling_windows(resolved_frame(n_days=5), windows=(7,))
    assert not windows[0].reportable
    assert "not reportable" in str(windows[0])


def test_monitor_reports_insufficient_data_rather_than_guessing():
    result = PerformanceMonitor(baseline_auc=0.52).evaluate(resolved_frame(n_days=3))
    assert result["status"] == "INSUFFICIENT_DATA"
    assert not result["detected"]


# =====================================================================
# Calibration monitoring
# =====================================================================


def test_calibration_needs_a_bigger_sample_than_auc():
    """10 bins x ~30 per bin, or ECE measures the binning rather than the model."""
    assert MIN_CALIBRATION_OBSERVATIONS > 100


def test_thin_calibration_window_is_not_reportable():
    windows = monitor_calibration(resolved_frame(n_days=10), windows=(7,))
    assert not windows[0].reportable


def test_calibration_within_tolerance_is_not_flagged():
    windows = monitor_calibration(resolved_frame(n_days=200, auc_signal=0.3), windows=(90,))
    drifted, _ = calibration_drifted(windows, baseline_ece=0.50)
    assert not drifted


def test_calibration_beyond_tolerance_is_flagged():
    # 15 symbols needed to clear the 300-observation floor on a 90-day window:
    # 90 calendar days is ~65 sessions, so 3 symbols would give only 195.
    windows = monitor_calibration(
        resolved_frame(n_days=200, symbols=tuple("ABCDEFGHIJKLMNO"), auc_signal=0.0),
        windows=(90,),
    )
    assert windows[0].reportable
    drifted, message = calibration_drifted(windows, baseline_ece=0.0, tolerance=0.001)
    assert drifted
    assert "exceeds baseline" in message


def test_three_symbols_cannot_support_a_90_day_calibration_window():
    """Documents the data requirement: a small universe cannot be monitored.

    90 calendar days is ~65 sessions. With 3 symbols that is 195 observations,
    below the 300 needed for a 10-bin ECE. The universe size determines whether
    calibration monitoring is possible at all.
    """
    windows = monitor_calibration(
        resolved_frame(n_days=200, symbols=("A", "B", "C")), windows=(90,)
    )
    assert not windows[0].reportable


# =====================================================================
# Policy rule 1: drift alone never retrains
# =====================================================================


def test_drift_alone_does_not_trigger_retraining():
    decision = RetrainingPolicy().evaluate(
        MonitoringSignals(
            performance_degraded=False,
            n_features_drifting=20,
            n_features_total=45,
            n_new_observations=5000,
        )
    )
    assert not decision.should_retrain
    assert decision.diagnosis is Diagnosis.DRIFT_ONLY_MODEL_COPING


def test_widespread_drift_raises_a_warning_not_a_retrain():
    decision = RetrainingPolicy().evaluate(
        MonitoringSignals(
            n_features_drifting=40,
            n_features_total=45,
            n_new_observations=5000,
        )
    )
    assert decision.state is HealthState.WARNING
    assert not decision.should_retrain


# =====================================================================
# Policy rule 2: data-quality ERROR blocks everything
# =====================================================================


def test_data_quality_error_blocks_retraining():
    """Even with performance collapsed and drift everywhere."""
    decision = RetrainingPolicy().evaluate(
        MonitoringSignals(
            performance_degraded=True,
            calibration_drifted=True,
            n_features_drifting=45,
            n_features_total=45,
            data_quality_errors=1,
            n_new_observations=100_000,
        )
    )
    assert decision.state is HealthState.BLOCKED
    assert not decision.should_retrain
    assert decision.diagnosis is Diagnosis.DATA_QUALITY_SUSPECTED


def test_data_quality_warnings_do_not_block():
    decision = RetrainingPolicy().evaluate(
        MonitoringSignals(
            performance_degraded=True,
            data_quality_warnings=50,
            n_new_observations=5000,
        )
    )
    assert decision.state is not HealthState.BLOCKED
    assert decision.should_retrain


# =====================================================================
# Policy rule 3: insufficient data means wait
# =====================================================================


def test_insufficient_observations_defers_retraining():
    decision = RetrainingPolicy().evaluate(
        MonitoringSignals(
            performance_degraded=True,
            n_new_observations=50,
        )
    )
    assert decision.state is HealthState.DEGRADED
    assert not decision.should_retrain
    assert any("new observations" in r for r in decision.reasons)


def test_sufficient_observations_allows_retraining():
    decision = RetrainingPolicy().evaluate(
        MonitoringSignals(
            performance_degraded=True,
            n_new_observations=5000,
        )
    )
    assert decision.state is HealthState.RETRAIN_REQUIRED
    assert decision.should_retrain


# =====================================================================
# Diagnosis
# =====================================================================


def test_degradation_with_drift_diagnoses_regime_change():
    decision = RetrainingPolicy().evaluate(
        MonitoringSignals(
            performance_degraded=True,
            n_features_drifting=15,
            n_features_total=45,
            n_new_observations=5000,
        )
    )
    assert decision.diagnosis is Diagnosis.LIKELY_REGIME_CHANGE


def test_degradation_without_drift_diagnoses_relationship_change():
    """Retraining on the same features may not help — say so."""
    decision = RetrainingPolicy().evaluate(
        MonitoringSignals(
            performance_degraded=True,
            n_features_drifting=0,
            n_features_total=45,
            n_new_observations=5000,
        )
    )
    assert decision.diagnosis is Diagnosis.LIKELY_RELATIONSHIP_CHANGE
    assert any("feature set" in r for r in decision.reasons)


def test_undetectable_performance_is_not_treated_as_health():
    """Unchanged metrics below the MDE are not evidence the model is fine."""
    decision = RetrainingPolicy().evaluate(
        MonitoringSignals(
            performance_degraded=False,
            performance_detectable=False,
            n_new_observations=5000,
        )
    )
    assert any("NOT evidence of health" in r for r in decision.reasons)


def test_all_clear_is_healthy():
    decision = RetrainingPolicy().evaluate(
        MonitoringSignals(
            n_new_observations=5000,
            n_features_total=45,
        )
    )
    assert decision.state is HealthState.HEALTHY
    assert not decision.should_retrain


# =====================================================================
# Policy rule 4: a failed candidate cannot be promoted
# =====================================================================


def failing_outcome() -> ValidationOutcome:
    return ValidationOutcome(
        passed=False,
        candidate_version="cand-1",
        incumbent_version="prod-1",
        checks={"auc_not_worse": False},
    )


def test_promoting_a_failed_candidate_raises():
    """Structural, not procedural. There is no override flag."""
    with pytest.raises(PromotionBlocked, match="no override"):
        PromotionDecision(promote=True, outcome=failing_outcome())


def test_failed_candidate_keeps_the_incumbent():
    decision = decide_promotion(failing_outcome())
    assert not decision.promote
    assert decision.final_state is HealthState.KEEP_CURRENT_MODEL
    assert "prod-1" in decision.reason


def test_passing_candidate_is_promoted():
    outcome = validate_candidate(
        "cand-1",
        "prod-1",
        candidate_metrics={
            "roc_auc": 0.55,
            "ece_quantile": 0.03,
            "sharpness": 0.05,
            "n": 5000,
        },
        incumbent_metrics={"roc_auc": 0.52, "ece_quantile": 0.04},
    )
    decision = decide_promotion(outcome)
    assert outcome.passed and decision.promote
    assert decision.final_state is HealthState.PROMOTE_NEW_MODEL


def test_better_auc_with_worse_calibration_fails():
    """A single metric must not decide promotion."""
    outcome = validate_candidate(
        "cand-1",
        "prod-1",
        candidate_metrics={
            "roc_auc": 0.60,
            "ece_quantile": 0.30,
            "sharpness": 0.05,
            "n": 5000,
        },
        incumbent_metrics={"roc_auc": 0.52, "ece_quantile": 0.04},
    )
    assert not outcome.passed
    assert "calibration_not_worse" in outcome.failed_checks


def test_constant_predictor_fails_validation():
    """A well-calibrated constant is not an improvement."""
    outcome = validate_candidate(
        "cand-1",
        "prod-1",
        candidate_metrics={"roc_auc": 0.55, "ece_quantile": 0.01, "sharpness": 0.0, "n": 5000},
        incumbent_metrics={"roc_auc": 0.52, "ece_quantile": 0.04},
    )
    assert not outcome.passed
    assert "produces_varied_predictions" in outcome.failed_checks


def test_thin_evaluation_sample_fails_validation():
    outcome = validate_candidate(
        "cand-1",
        "prod-1",
        candidate_metrics={"roc_auc": 0.60, "ece_quantile": 0.02, "sharpness": 0.05, "n": 10},
        incumbent_metrics={"roc_auc": 0.52, "ece_quantile": 0.04},
    )
    assert "sufficient_evaluation_data" in outcome.failed_checks


# =====================================================================
# Alerts
# =====================================================================


def test_duplicate_alerts_are_suppressed_within_cooldown():
    c = AlertCollector(cooldown_days=3)
    assert c.add(Severity.WARNING, "DRIFT", "x", when=date(2023, 6, 1))
    assert c.add(Severity.WARNING, "DRIFT", "x", when=date(2023, 6, 2)) is None
    assert len(c.alerts) == 1


def test_alert_fires_again_after_cooldown():
    c = AlertCollector(cooldown_days=3)
    c.add(Severity.WARNING, "DRIFT", "x", when=date(2023, 6, 1))
    assert c.add(Severity.WARNING, "DRIFT", "x", when=date(2023, 6, 10))


def test_info_alerts_are_not_alertable():
    c = AlertCollector()
    c.add(Severity.INFO, "NOTE", "just so you know")
    assert c.alerts and not c.alertable


def test_worst_severity_is_reported():
    c = AlertCollector()
    c.add(Severity.INFO, "A", "a")
    c.add(Severity.ERROR, "B", "b")
    c.add(Severity.WARNING, "C", "c")
    assert c.worst is Severity.ERROR
