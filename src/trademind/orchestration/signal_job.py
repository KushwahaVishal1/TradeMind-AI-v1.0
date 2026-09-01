"""Generate current-session signals from the persisted production ensemble."""

from __future__ import annotations

import pandas as pd

from ..ensemble import load_production_ensemble
from .base import Job, JobResult, JobStatus, PipelineContext, assert_temporally_valid


class LiveSignalJob(Job):
    name = "signals"

    def execute(self, context: PipelineContext) -> JobResult:
        if context.mode != "daily":
            return JobResult(self.name, JobStatus.SKIPPED, "live inference is daily-only")

        panel = context.get("panel")
        if panel is None or panel.empty:
            return JobResult(self.name, JobStatus.SKIPPED, "no panel available")

        path = context.cfg.root / "models" / "production_ensemble.joblib"
        if not path.exists():
            return JobResult(
                self.name,
                JobStatus.SKIPPED,
                "no production ensemble; run `python main.py ensemble`",
            )

        bundle = load_production_ensemble(path)
        if bundle.feature_version != context.cfg.feature_version:
            return JobResult(
                self.name,
                JobStatus.FAILED,
                f"ensemble uses {bundle.feature_version}, config uses "
                f"{context.cfg.feature_version}",
            )

        assert_temporally_valid(context.as_of, bundle.training_end, bundle.version)
        dates = pd.to_datetime(panel["date"]).dt.date
        latest = panel.loc[dates == context.as_of].copy()
        if latest.empty:
            return JobResult(
                self.name,
                JobStatus.SKIPPED,
                f"no feature rows for {context.as_of}",
            )

        signals = bundle.predict(latest)
        context.put("signals", signals)
        context.put("model_version", bundle.version)
        context.put("training_start", bundle.training_start)
        context.put("training_end", bundle.training_end)
        return JobResult(self.name, JobStatus.SUCCESS, records=len(signals))
