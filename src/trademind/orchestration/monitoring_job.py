"""Monitoring job: close the Phase 8 gap.

The roadmap noted one operational gap in Phase 8 -- real production feature
distributions were never wired into the drift monitor. This is where that
closes: the reference window is the model's training distribution, the current
window is what the live pipeline is actually seeing today.

Getting the reference right matters. Comparing today against *all* history
would flag every genuine long-run trend as drift. Comparing against the
training window answers the question that matters: does the data look like what
the model learned from?
"""

from __future__ import annotations

import pandas as pd

from ..monitoring import DriftTracker, RetrainingPolicy, run_monitoring
from .base import Job, JobResult, JobStatus, PipelineContext


class MonitoringJob(Job):
    name = "monitoring"

    def __init__(self, tracker: DriftTracker | None = None,
                 policy: RetrainingPolicy | None = None) -> None:
        self.tracker = tracker or DriftTracker()
        self.policy = policy or RetrainingPolicy()

    def execute(self, context: PipelineContext) -> JobResult:
        store = context.store
        panel = context.get("panel")
        if store is None or panel is None or panel.empty:
            return JobResult(self.name, JobStatus.SKIPPED, "nothing to monitor")

        feature_cols = context.get("feature_columns")
        if not feature_cols:
            from ..features import feature_columns as fc
            feature_cols = fc(panel)

        resolved = pd.DataFrame([dict(r) for r in store.resolved()])

        # Reference = the model's training window. Current = recent live data.
        training_end = context.get("training_end")
        dates = pd.to_datetime(panel["date"])
        if training_end:
            boundary = pd.Timestamp(training_end)
            reference = panel[dates <= boundary]
            current = panel[dates > boundary]
        else:
            boundary = dates.quantile(0.75)
            reference = panel[dates <= boundary]
            current = panel[dates > boundary]

        if current.empty:
            return JobResult(
                self.name, JobStatus.SKIPPED,
                "no data after the training window yet; nothing to compare",
            )

        report = run_monitoring(
            resolved=resolved,
            reference_features=reference,
            current_features=current,
            feature_columns=feature_cols,
            baseline_auc=context.get("baseline_auc", 0.52),
            baseline_ece=context.get("baseline_ece", 0.05),
            store=store,
            tracker=self.tracker,
            policy=self.policy,
        )
        context.put("health_report", report)

        return JobResult(
            self.name, JobStatus.SUCCESS,
            f"{report.state.value} ({report.decision.diagnosis.value})",
            records=len(resolved),
            details={"state": report.state.value,
                     "should_retrain": report.decision.should_retrain},
        )


class RetrainingJob(Job):
    """Act on the policy's verdict -- and only on the policy's verdict.

    Deliberately does nothing unless ``should_retrain`` is true. A retraining
    job that decides for itself when to fire would duplicate the policy logic
    and eventually disagree with it, which is how governed systems quietly
    become ungoverned.
    """

    name = "retraining"

    def __init__(self, retrain_fn=None) -> None:
        self.retrain_fn = retrain_fn

    def execute(self, context: PipelineContext) -> JobResult:
        report = context.get("health_report")
        if report is None:
            return JobResult(self.name, JobStatus.SKIPPED, "no health report")

        decision = report.decision
        if not decision.should_retrain:
            return JobResult(
                self.name, JobStatus.SKIPPED,
                f"policy says no ({decision.state.value})",
            )

        if self.retrain_fn is None:
            return JobResult(
                self.name, JobStatus.SKIPPED,
                "retraining required but no retrain function is configured",
                details={"diagnosis": decision.diagnosis.value},
            )

        outcome = self.retrain_fn(context)
        promoted = bool(getattr(outcome, "promote", False))
        return JobResult(
            self.name, JobStatus.SUCCESS,
            "candidate promoted" if promoted else "candidate rejected; "
            "incumbent retained",
            details={"promoted": promoted},
        )
