"""Job protocol, results, and the guard that makes backfill honest.

## The backfill contamination trap

Backfilling prediction history is the obvious way to populate the monitoring
tables, and the obvious implementation is wrong::

    model = registry.load(production_version)      # trained through last month
    for day in historical_dates:                   # going back two years
        store.save(predict(model, features_on(day)))

Every one of those rows is a "prediction" made by a model trained on data from
*after* the date it predicts. The stored history then shows a model that knew
the future, monitoring reports excellent performance, and the baseline that
Phase 8 compares against is fiction. Nothing raises. The dates are real, the
features are real, the outcomes are real — only the counterfactual is impossible.

It is the same class of error as a leaky feature, moved into the operational
layer, and it is easy to commit because backfilling *looks* like data plumbing
rather than modelling.

The guard is one invariant, enforced at write time for every prediction:

    training_end < prediction_date

A model may only predict dates strictly after the last day it was trained on.
``assert_temporally_valid`` raises otherwise, so an honest backfill has to walk
forward — retraining as it goes, or replaying out-of-fold predictions from the
purged splitter, which is what ``prediction_job`` does.

## Failure recovery

Every job returns a ``JobResult`` rather than raising, so one failing job does
not abort the pipeline and leave the run record stuck at RUNNING. The pipeline
decides what is fatal. A run that crashes hard is detected on the next start and
marked FAILED, so the run table never accumulates phantom RUNNING rows that
would make "is anything executing right now?" unanswerable.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum

log = logging.getLogger(__name__)

# A run still marked RUNNING after this long is assumed dead.
STALE_RUN_HOURS = 6


class JobStatus(StrEnum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    PARTIAL = "PARTIAL"


class TemporalValidityError(RuntimeError):
    """A model was asked to predict a date at or before its training window ends."""


def assert_temporally_valid(
    prediction_date: date,
    training_end: date | str | None,
    model_version: str = "",
) -> None:
    """A model may only predict dates strictly after its training window.

    Cheap enough to call on every prediction. Call it on every prediction.
    """
    if training_end is None:
        raise TemporalValidityError(
            f"Model {model_version} has no recorded training_end, so its "
            "predictions cannot be shown to be free of look-ahead. Refusing."
        )

    end = date.fromisoformat(training_end) if isinstance(training_end, str) else training_end
    if end >= prediction_date:
        raise TemporalValidityError(
            f"Model {model_version} was trained through {end} and cannot "
            f"predict {prediction_date}. A prediction made by a model that saw "
            "the future is not a prediction. For historical dates use "
            "walk-forward out-of-fold predictions."
        )


@dataclass
class JobResult:
    """Outcome of one job. Jobs return these; they do not raise."""

    name: str
    status: JobStatus
    message: str = ""
    records: int = 0
    duration_seconds: float = 0.0
    details: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in (JobStatus.SUCCESS, JobStatus.SKIPPED)

    def __str__(self) -> str:
        base = (
            f"{self.name:<18} {self.status.value:<8} "
            f"{self.records:>6,} records  {self.duration_seconds:>6.1f}s"
        )
        return f"{base}  {self.message}" if self.message else base


class Job(ABC):
    """One stage of the daily pipeline."""

    name: str = "job"

    @abstractmethod
    def execute(self, context: PipelineContext) -> JobResult:
        """Do the work. Must not raise; wrap failures in a FAILED result."""

    def run(self, context: PipelineContext) -> JobResult:
        """Execute with timing and a last-resort exception boundary."""
        started = time.monotonic()
        try:
            result = self.execute(context)
        except Exception as exc:
            log.exception("%s raised", self.name)
            result = JobResult(
                self.name,
                JobStatus.FAILED,
                message=f"unhandled exception: {exc}",
                error=str(exc),
            )
        result.duration_seconds = time.monotonic() - started
        log.info("%s", result)
        return result


@dataclass
class PipelineContext:
    """Shared state passed between jobs.

    Jobs communicate through this rather than by returning data to each other,
    so a job can be skipped or fail without breaking the ones after it — they
    check for what they need and skip cleanly if it is absent.
    """

    cfg: object
    mode: str
    run_id: str
    as_of: date
    store: object = None
    lake: object = None
    results: list[JobResult] = field(default_factory=list)
    artifacts: dict = field(default_factory=dict)

    def record(self, result: JobResult) -> JobResult:
        self.results.append(result)
        return result

    def get(self, key: str, default=None):
        return self.artifacts.get(key, default)

    def put(self, key: str, value) -> None:
        self.artifacts[key] = value

    @property
    def failed(self) -> list[JobResult]:
        return [r for r in self.results if r.status is JobStatus.FAILED]

    @property
    def ok(self) -> bool:
        return not self.failed


def reap_stale_runs(store, hours: int = STALE_RUN_HOURS) -> int:
    """Mark abandoned RUNNING rows as FAILED.

    A process killed mid-run leaves its row at RUNNING forever, which makes
    "is anything executing?" unanswerable and lets a scheduler stack overlapping
    runs. Reaping on startup keeps the run table honest.
    """
    cutoff = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
    cursor = store.conn.execute(
        "UPDATE runs SET status = 'FAILED', finished_at = ?, "
        "error = 'abandoned; reaped on startup' "
        "WHERE status = 'RUNNING' AND started_at < ?",
        (datetime.now(UTC).isoformat(timespec="seconds"), cutoff),
    )
    n = cursor.rowcount or 0
    if n:
        log.warning("Reaped %d abandoned run(s) still marked RUNNING", n)
    return n
