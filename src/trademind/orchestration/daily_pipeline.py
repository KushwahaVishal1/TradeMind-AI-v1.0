"""The daily pipeline.

Three modes, one job sequence:

``backfill``
    Build everything from history. Predictions come from walk-forward
    out-of-fold output, never from the current model applied backwards.

``daily``
    Incremental. Fetch new bars, rebuild features, resolve yesterday's
    outcomes, predict today, monitor.

``retrain``
    Monitoring and the governance policy only. No new predictions — a
    retraining run that also predicted would be using a model mid-swap.

## Ordering

Outcome resolution runs **before** prediction. Yesterday's outcome is known
today, and resolving first means the monitoring stage sees the freshest
possible evidence. It also keeps the pending queue from growing by one day
every day, which is the failure mode where outcomes are technically resolved
but always one cycle stale.

Monitoring runs **after** prediction so today's features are in the current
window when drift is computed.

## Partial failure

Jobs return results rather than raising. The pipeline continues past a
PARTIAL — one delisted ticker should not stop the system — but stops on a
FAILED job whose output later stages depend on. The run record is always
closed out, so a crash cannot leave a row stuck at RUNNING.
"""

from __future__ import annotations

import logging
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import date

from ..storage import PredictionStore, init_db
from ..storage.lake import ParquetLake
from .base import Job, JobResult, JobStatus, PipelineContext, reap_stale_runs
from .ingestion_job import FeatureJob, IngestionJob
from .monitoring_job import MonitoringJob, RetrainingJob
from .outcome_job import OutcomeJob
from .prediction_job import PredictionJob
from .signal_job import LiveSignalJob

log = logging.getLogger(__name__)

MODES = ("backfill", "daily", "retrain")

# Jobs whose failure makes everything after them meaningless.
CRITICAL = {"ingestion", "features"}


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


@dataclass
class PipelineRun:
    """The outcome of one full pipeline execution."""

    run_id: str
    mode: str
    as_of: date
    results: list[JobResult] = field(default_factory=list)
    status: str = "RUNNING"
    context: PipelineContext | None = None

    @property
    def ok(self) -> bool:
        return self.status == "SUCCESS"

    def render(self) -> str:
        lines = [f"=== {self.mode} run {self.run_id} | as of {self.as_of} | {self.status} ==="]
        for r in self.results:
            lines.append(f"  {r}")

        report = self.context.get("health_report") if self.context else None
        if report is not None:
            lines += ["", report.render()]
        return "\n".join(lines)


def build_jobs(mode: str, provider=None, retrain_fn=None) -> list[Job]:
    """Job sequence for a mode."""
    if mode == "retrain":
        # No prediction stage: predicting with a model that is about to be
        # replaced would attribute those rows to a version that may not survive.
        return [FeatureJob(), MonitoringJob(), RetrainingJob(retrain_fn)]

    jobs = [
        IngestionJob(provider),
        FeatureJob(),
        OutcomeJob(),  # resolve before predicting; see module docstring
    ]
    if mode == "daily":
        jobs.append(LiveSignalJob())
    jobs.extend([PredictionJob(), MonitoringJob(), RetrainingJob(retrain_fn)])
    return jobs


def run_pipeline(
    cfg,
    mode: str = "daily",
    as_of: date | None = None,
    provider=None,
    retrain_fn=None,
    jobs: list[Job] | None = None,
    artifacts: dict | None = None,
) -> PipelineRun:
    """Execute the pipeline for one mode."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    as_of = as_of or date.today()
    conn = init_db(cfg.data_root / "trademind.db")
    store = PredictionStore(conn)

    reap_stale_runs(store)

    run_id = store.start_run(
        mode=mode,
        config_hash=cfg.config_hash,
        git_commit=_git_commit(),
        python_version=platform.python_version(),
    )

    context = PipelineContext(
        cfg=cfg,
        mode=mode,
        run_id=run_id,
        as_of=as_of,
        store=store,
        lake=ParquetLake(cfg.data_root),
        artifacts=dict(artifacts or {}),
    )
    run = PipelineRun(run_id=run_id, mode=mode, as_of=as_of, context=context)

    try:
        for job in jobs or build_jobs(mode, provider, retrain_fn):
            result = context.record(job.run(context))
            run.results.append(result)

            if result.status is JobStatus.FAILED and job.name in CRITICAL:
                run.status = "FAILED"
                store.finish_run(
                    run_id,
                    "FAILED",
                    error=f"{job.name}: {result.message}",
                )
                log.error("Pipeline halted: %s failed", job.name)
                return run

        run.status = "SUCCESS" if context.ok else "FAILED"
        store.finish_run(
            run_id,
            run.status,
            error=None
            if context.ok
            else "; ".join(f"{r.name}: {r.message}" for r in context.failed),
        )
        return run

    except Exception as exc:
        log.exception("Pipeline crashed")
        run.status = "FAILED"
        store.finish_run(run_id, "FAILED", error=str(exc))
        return run

    finally:
        conn.close()
