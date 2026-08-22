"""Ingestion pipeline: fetch, enrich with corporate actions, validate, persist.

The ordering matters and is not arbitrary:

1. **Fetch** raw bars from the provider.
2. **Apply corporate actions** — reconstruct as-traded prices, build the
   total-return series.
3. **Validate** the enriched frame.
4. **Persist** only if no ERROR-severity finding was raised.

Validation runs *after* enrichment because several checks — split
reconciliation above all — can only be evaluated once actions are attached.
And persistence runs last because writing a frame that failed validation would
mean the lake could contain data the pipeline already knows is broken.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from ..storage.lake import ParquetLake
from ..storage.predictions import PredictionStore
from .base import MarketDataProvider
from .calendar import TradingCalendar
from .corporate_actions import apply_corporate_actions
from .data_validator import DataValidator, Issue, ValidationReport

log = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    symbol: str
    rows: int
    written: bool
    report: ValidationReport
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.written and self.error is None


@dataclass
class IngestionSummary:
    results: list[IngestionResult] = field(default_factory=list)

    @property
    def succeeded(self) -> list[IngestionResult]:
        return [r for r in self.results if r.ok]

    @property
    def failed(self) -> list[IngestionResult]:
        return [r for r in self.results if not r.ok]

    @property
    def all_issues(self) -> list[Issue]:
        return [i for r in self.results for i in r.report.issues]

    def render(self) -> str:
        lines = [
            f"Ingested {len(self.succeeded)}/{len(self.results)} symbols, "
            f"{sum(r.rows for r in self.succeeded):,} rows"
        ]
        for r in self.failed:
            reason = r.error or "; ".join(i.code for i in r.report.errors)
            lines.append(f"  FAILED {r.symbol}: {reason}")
        warn = [i for i in self.all_issues if i.severity == "WARNING"]
        if warn:
            lines.append(f"  {len(warn)} warning(s) logged to data_quality_issues")
        return "\n".join(lines)


class MarketDataIngestion:
    """Runs the ingestion pipeline for a universe of symbols."""

    def __init__(
        self,
        provider: MarketDataProvider,
        lake: ParquetLake,
        store: PredictionStore | None = None,
        calendar: TradingCalendar | None = None,
    ) -> None:
        self.provider = provider
        self.lake = lake
        self.store = store
        self.calendar = calendar or TradingCalendar()
        self.validator = DataValidator(self.calendar)

    def ingest_symbol(
        self,
        symbol: str,
        start: date,
        end: date,
        *,
        run_id: str | None = None,
        incremental: bool = False,
    ) -> IngestionResult:
        if incremental:
            last = self.lake.last_date(symbol)
            if last is not None:
                # Re-fetch a short overlap so that restated bars and any
                # corporate action announced since the last run are picked up.
                start = max(start, last - pd.Timedelta(days=7).to_pytimedelta())
                log.debug("%s: incremental from %s (last stored %s)",
                          symbol, start, last)

        try:
            raw = self.provider.fetch(symbol, start, end)
        except Exception as exc:
            log.error("%s: fetch failed: %s", symbol, exc)
            return IngestionResult(
                symbol, 0, False,
                ValidationReport(symbol, [Issue("ERROR", "FETCH_FAILED", str(exc), symbol)]),
                error=str(exc),
            )

        if raw.empty:
            report = ValidationReport(
                symbol, [Issue("ERROR", "NO_DATA", "Provider returned zero rows", symbol)]
            )
            self._persist_issues(report, run_id)
            return IngestionResult(symbol, 0, False, report, error="no data")

        enriched = apply_corporate_actions(raw)
        report = self.validator.validate(enriched, symbol)
        self._persist_issues(report, run_id)

        if not report.ok:
            log.error("%s: %d error(s), refusing to write. %s",
                      symbol, len(report.errors), report.errors[0])
            return IngestionResult(symbol, len(enriched), False, report)

        self.lake.append_raw(symbol, enriched)
        log.info("%s: %d rows | %s", symbol, len(enriched), report.summary())
        return IngestionResult(symbol, len(enriched), True, report)

    def ingest_universe(
        self,
        symbols: list[str],
        start: date,
        end: date,
        *,
        run_id: str | None = None,
        incremental: bool = False,
    ) -> IngestionSummary:
        summary = IngestionSummary()
        for symbol in symbols:
            summary.results.append(
                self.ingest_symbol(symbol, start, end,
                                   run_id=run_id, incremental=incremental)
            )
        log.info("\n%s", summary.render())
        return summary

    # -- helpers ----------------------------------------------------------

    def _persist_issues(self, report: ValidationReport, run_id: str | None) -> None:
        """Data-quality findings go to the operational store.

        Phase 8's retraining policy reads this table: an unresolved ERROR blocks
        retraining regardless of how severe the other signals look.
        """
        if self.store is None:
            return
        for issue in report.issues:
            if issue.severity == "INFO":
                continue  # informational only; not worth persisting
            self.store.log_issue(
                severity=issue.severity,
                code=issue.code,
                detail=issue.detail,
                symbol=issue.symbol,
                issue_date=issue.issue_date.isoformat() if issue.issue_date else None,
                run_id=run_id,
            )
