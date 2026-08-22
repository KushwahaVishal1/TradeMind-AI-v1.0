"""Ingestion job: fetch and validate market data."""

from __future__ import annotations

from ..ingestion import MarketDataIngestion, YFinanceProvider
from ..storage.lake import ParquetLake
from .base import Job, JobResult, JobStatus, PipelineContext


class IngestionJob(Job):
    name = "ingestion"

    def __init__(self, provider=None) -> None:
        self.provider = provider

    def execute(self, context: PipelineContext) -> JobResult:
        cfg = context.cfg
        lake = context.lake or ParquetLake(cfg.data_root)

        ingestion = MarketDataIngestion(
            provider=self.provider or YFinanceProvider(),
            lake=lake,
            store=context.store,
        )
        summary = ingestion.ingest_universe(
            symbols=cfg.universe,
            start=cfg.history_start,
            end=context.as_of,
            run_id=context.run_id,
            incremental=(context.mode == "daily"),
        )

        n_rows = sum(r.rows for r in summary.succeeded)
        context.put("ingestion_summary", summary)

        if not summary.succeeded:
            return JobResult(
                self.name,
                JobStatus.FAILED,
                "no symbol ingested successfully",
                records=0,
            )
        if summary.failed:
            # Partial is not failure: the pipeline can produce signals for the
            # symbols that worked, and forcing an all-or-nothing rule would let
            # one delisted ticker stop the whole system.
            return JobResult(
                self.name,
                JobStatus.PARTIAL,
                f"{len(summary.failed)} of {len(summary.results)} symbols failed",
                records=n_rows,
                details={"failed": [r.symbol for r in summary.failed]},
            )
        return JobResult(self.name, JobStatus.SUCCESS, records=n_rows)


class FeatureJob(Job):
    """Rebuild the feature panel from stored bars."""

    name = "features"

    def execute(self, context: PipelineContext) -> JobResult:
        from ..features import build_panel, coverage_report, feature_columns

        cfg = context.cfg
        lake = context.lake or ParquetLake(cfg.data_root)

        symbols = lake.symbols()
        if not symbols:
            return JobResult(self.name, JobStatus.FAILED, "no market data in the lake")

        frames = {s: lake.read_raw(s) for s in symbols}
        frames = {k: v for k, v in frames.items() if v is not None and not v.empty}

        panel = build_panel(
            frames,
            horizon=cfg.get("features.target_horizon_days"),
        )
        if panel.empty:
            return JobResult(self.name, JobStatus.FAILED, "empty feature panel")

        version = cfg.feature_version
        for symbol, group in panel.groupby("symbol"):
            lake.write_features(symbol, version, group.reset_index(drop=True))

        context.put("panel", panel)
        context.put("feature_columns", feature_columns(panel))

        worst = coverage_report(panel).iloc[0]
        message = ""
        if worst["missing_rate"] > 0.10:
            message = f"{worst['feature']} is {worst['missing_rate']:.0%} missing"
        return JobResult(self.name, JobStatus.SUCCESS, message, records=len(panel))
