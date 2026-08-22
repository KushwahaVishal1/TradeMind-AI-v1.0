"""Monitoring pipeline: gather signals, apply the policy, emit a health report.

Ordering matters. Data-quality state is collected *first*, because a
data-quality ERROR short-circuits the retraining policy regardless of what
performance and drift say — and gathering signals in the order the policy
consumes them keeps the two from drifting apart.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from .alerts import AlertCollector, Severity
from .calibration_monitor import calibration_drifted, monitor_calibration
from .feature_drift import DriftTracker, compute_drift, summarise_drift
from .model_monitor import HealthReport
from .performance_monitor import PerformanceMonitor
from .retraining import MonitoringSignals, RetrainingPolicy

log = logging.getLogger(__name__)


def count_open_data_quality_errors(store, since: date | None = None) -> tuple[int, int]:
    """Unresolved ERROR and WARNING counts from ingestion."""
    sql = "SELECT severity, COUNT(*) AS n FROM data_quality_issues"
    params = ()
    if since:
        sql += " WHERE created_at >= ?"
        params = (since.isoformat(),)
    sql += " GROUP BY severity"

    rows = store.conn.execute(sql, params).fetchall()
    counts = {r["severity"]: r["n"] for r in rows}
    return counts.get("ERROR", 0), counts.get("WARNING", 0)


def run_monitoring(
    resolved: pd.DataFrame,
    reference_features: pd.DataFrame,
    current_features: pd.DataFrame,
    feature_columns: list[str],
    baseline_auc: float,
    baseline_ece: float,
    store=None,
    tracker: DriftTracker | None = None,
    policy: RetrainingPolicy | None = None,
    as_of: pd.Timestamp | None = None,
) -> HealthReport:
    """Collect every monitoring signal and run the governance policy."""
    alerts = AlertCollector()
    tracker = tracker or DriftTracker()
    policy = policy or RetrainingPolicy()
    stamp = (as_of or pd.Timestamp.utcnow()).isoformat()

    # 1. Data quality first — it gates everything downstream.
    dq_errors, dq_warnings = (
        count_open_data_quality_errors(store) if store is not None else (0, 0)
    )
    if dq_errors:
        alerts.add(
            Severity.ERROR,
            "DATA_QUALITY_ERRORS",
            f"{dq_errors} unresolved ingestion error(s); retraining is blocked.",
        )

    # 2. Performance.
    perf_monitor = PerformanceMonitor(baseline_auc=baseline_auc)
    performance = perf_monitor.evaluate(resolved, as_of)
    if performance["detected"]:
        alerts.add(Severity.WARNING, "PERFORMANCE_DEGRADED", performance["explanation"])
    elif performance["status"] == "INSUFFICIENT_DATA":
        alerts.add(Severity.INFO, "INSUFFICIENT_OUTCOMES", performance["explanation"])

    # 3. Calibration.
    calibration = monitor_calibration(resolved, as_of)
    cal_drifted, cal_message = calibration_drifted(calibration, baseline_ece)
    if cal_drifted:
        alerts.add(Severity.WARNING, "CALIBRATION_DRIFT", cal_message)

    # 4. Feature drift, with persistence.
    drift = compute_drift(reference_features, current_features, feature_columns)
    persistent = tracker.update(drift) if not drift.empty else []
    if persistent:
        alerts.add(
            Severity.WARNING,
            "PERSISTENT_FEATURE_DRIFT",
            f"{len(persistent)} feature(s) drifting for "
            f"{tracker.persistence_days}+ consecutive checks: {persistent[:5]}",
        )

    summary = summarise_drift(drift) if not drift.empty else {}

    # 5. Policy.
    judged = performance.get("judged_on")
    signals = MonitoringSignals(
        performance_degraded=performance["detected"],
        performance_detectable=bool(judged is not None and judged.reportable),
        calibration_drifted=cal_drifted,
        n_features_drifting=len(persistent),
        n_features_total=int(summary.get("n_features", len(feature_columns))),
        data_quality_errors=dq_errors,
        data_quality_warnings=dq_warnings,
        n_new_observations=len(resolved),
    )
    decision = policy.evaluate(signals)

    log.info("Health: %s (%s)", decision.state.value, decision.diagnosis.value)

    return HealthReport(
        as_of=stamp,
        signals=signals,
        decision=decision,
        alerts=alerts,
        performance=performance,
        calibration=calibration,
        drift=drift,
    )
