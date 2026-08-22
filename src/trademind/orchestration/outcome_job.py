"""Outcome resolution: join stored predictions to what the market did.

Resolves against the *tradeable* label -- open[t+2]/open[t+1] - 1 -- the same
target the model was trained on. Resolving against a close-to-close return
would score the model on a target it never optimised for, and the resulting
"degradation" would be an artifact of the mismatch rather than a fact about
the model.
"""

from __future__ import annotations

import pandas as pd

from ..features.labels import TRADEABLE_LABEL
from .base import Job, JobResult, JobStatus, PipelineContext


class OutcomeJob(Job):
    name = "outcomes"

    def execute(self, context: PipelineContext) -> JobResult:
        store = context.store
        if store is None:
            return JobResult(self.name, JobStatus.SKIPPED, "no store")

        panel = context.get("panel")
        if panel is None or panel.empty:
            return JobResult(self.name, JobStatus.SKIPPED, "no panel available")

        pending = store.pending(context.as_of.isoformat())
        if not pending:
            return JobResult(self.name, JobStatus.SUCCESS, "nothing pending")

        labels = panel.dropna(subset=[TRADEABLE_LABEL])
        lookup = {
            (r.symbol, pd.Timestamp(r.date).date()): float(getattr(r, TRADEABLE_LABEL))
            for r in labels.itertuples()
        }

        resolved = 0
        for row in pending:
            key = (row["symbol"], pd.Timestamp(row["prediction_date"]).date())
            actual = lookup.get(key)
            if actual is None:
                continue
            store.resolve(row["prediction_id"], actual)
            resolved += 1

        still_pending = len(pending) - resolved
        message = f"{still_pending} still awaiting outcomes" if still_pending else ""
        return JobResult(self.name, JobStatus.SUCCESS, message, records=resolved)
