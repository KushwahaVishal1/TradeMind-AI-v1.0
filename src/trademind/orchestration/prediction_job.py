"""Prediction job: generate signals and persist them with full lineage.

Two modes, and the distinction is the whole point of this module.

``daily``
    Predict the most recent date with the production model. The model's
    ``training_end`` must precede that date, which it naturally does.

``backfill``
    Reconstruct history. Loading the production model and predicting backwards
    would produce rows that no model could have produced at the time — see
    ``base.assert_temporally_valid``. Instead this replays **out-of-fold**
    predictions from the purged walk-forward splitter: each row predicted by a
    model fitted only on data before it. Slower, and the only version that
    yields a monitoring baseline worth comparing against.

Every persisted prediction carries the lineage the roadmap specified:
``prediction_id``, ``symbol``, ``prediction_date``, ``execution_date``,
``model_version``, ``feature_version``, ``decision_version``,
``threshold_version``, ``training_start``, ``training_end``,
``predicted_probability``, ``calibrated_probability``, ``predicted_return``,
``signal``.

Idempotency comes from Phase 0: ``prediction_id`` is a hash of the natural key,
so re-running a date is a no-op rather than a duplicate. A *conflicting* rewrite
— same key, different numbers — raises, because that means either the upstream
data was revised or the pipeline is non-deterministic.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from ..decision import DecisionEngine, RiskEngine, Thresholds
from ..decision.schema import DecisionInput, Position
from ..ingestion.calendar import TradingCalendar
from ..storage.predictions import Prediction, PredictionConflictError
from .base import (
    Job,
    JobResult,
    JobStatus,
    PipelineContext,
    TemporalValidityError,
    assert_temporally_valid,
)

log = logging.getLogger(__name__)


class PredictionJob(Job):
    """Generate decisions and persist them with lineage."""

    name = "predictions"

    def __init__(self, calendar: TradingCalendar | None = None) -> None:
        self.calendar = calendar or TradingCalendar()

    # -- helpers ---------------------------------------------------------

    def _engine(self, cfg) -> DecisionEngine | None:
        buy = cfg.get("decision.buy_threshold", None)
        sell = cfg.get("decision.sell_threshold", None)
        floor = cfg.get("decision.min_expected_return", None)
        if buy is None or sell is None or floor is None:
            return None

        return DecisionEngine(
            Thresholds(buy=buy, sell=sell, min_expected_return=floor),
            RiskEngine(max_positions=cfg.get("decision.max_positions", 10)),
            sizing_method=cfg.get("decision.sizing_method", "fixed"),
            max_weight=cfg.get("backtest.max_position_weight", 0.10),
            reference_volatility=cfg.get("decision.reference_volatility", 0.25),
        )

    def _persist(
        self,
        context: PipelineContext,
        decisions,
        signals: dict,
        model_version: str,
        training_start: str | None,
        training_end: str | None,
    ) -> tuple[int, int, list[str]]:
        cfg = context.cfg
        store = context.store
        written = skipped = 0
        problems: list[str] = []

        for d in decisions:
            try:
                assert_temporally_valid(d.decision_date, training_end, model_version)
            except TemporalValidityError as exc:
                problems.append(str(exc))
                continue

            execution_date = self.calendar.next_session(d.decision_date)
            if execution_date is None:
                problems.append(
                    f"{d.symbol} {d.decision_date}: no execution session within "
                    "the lookahead window; decision not persisted."
                )
                continue

            sig = signals.get((d.symbol, d.decision_date), {})
            prediction = Prediction(
                symbol=d.symbol,
                prediction_date=d.decision_date.isoformat(),
                execution_date=execution_date.isoformat(),
                model_version=model_version,
                feature_version=cfg.feature_version,
                decision_version=cfg.decision_version,
                threshold_version=cfg.threshold_version,
                training_start=training_start,
                training_end=training_end,
                predicted_probability=sig.get("raw"),
                calibrated_probability=d.calibrated_probability,
                predicted_return=d.expected_return,
                signal=d.signal.value,
                target_weight=(
                    None if d.target_weight is None or np.isnan(d.target_weight)
                    else float(d.target_weight)
                ),
                risk_bucket=d.risk_bucket.value,
                run_id=context.run_id,
            )
            try:
                store.save(prediction)
                written += 1
            except PredictionConflictError as exc:
                # Same key, different values: a revision or non-determinism.
                # Surface it rather than overwriting the historical record.
                problems.append(str(exc))
                skipped += 1

        return written, skipped, problems

    # -- entry point -----------------------------------------------------

    def execute(self, context: PipelineContext) -> JobResult:
        cfg = context.cfg
        store = context.store
        if store is None:
            return JobResult(self.name, JobStatus.SKIPPED, "no store")

        engine = self._engine(cfg)
        if engine is None:
            return JobResult(
                self.name, JobStatus.SKIPPED,
                "decision thresholds are null; run `main.py thresholds` first",
            )

        panel = context.get("panel")
        if panel is None or panel.empty:
            return JobResult(self.name, JobStatus.SKIPPED, "no panel available")

        signal_frame = context.get("signals")
        if signal_frame is None or signal_frame.empty:
            return JobResult(
                self.name, JobStatus.SKIPPED,
                "no calibrated signals; run the ensemble stage first",
            )

        model_version = context.get("model_version", "unknown")
        training_start = context.get("training_start")
        training_end = context.get("training_end")

        signals = {
            (r.symbol, pd.Timestamp(r.date).date()): {
                "calibrated": float(r.calibrated),
                "raw": float(getattr(r, "raw", np.nan)),
                "expected_return": float(getattr(r, "expected_return", np.nan)),
            }
            for r in signal_frame.itertuples()
        }

        target_dates = (
            [context.as_of] if context.mode == "daily"
            else sorted({d for _, d in signals})
        )

        all_decisions = []
        for when in target_dates:
            rows = panel[pd.to_datetime(panel["date"]).dt.date == when]
            inputs = []
            for row in rows.itertuples():
                sig = signals.get((row.symbol, when))
                if sig is None:
                    continue
                inputs.append(DecisionInput(
                    symbol=row.symbol,
                    decision_date=when,
                    calibrated_probability=sig["calibrated"],
                    expected_return=(
                        sig["expected_return"]
                        if np.isfinite(sig["expected_return"]) else None
                    ),
                    volatility=float(getattr(row, "volatility_20", np.nan))
                    if hasattr(row, "volatility_20") else None,
                    position=Position(row.symbol),
                    features_complete=True,
                ))
            if inputs:
                all_decisions.extend(engine.decide_batch(inputs))

        if not all_decisions:
            return JobResult(self.name, JobStatus.SUCCESS, "no decisions to persist")

        written, skipped, problems = self._persist(
            context, all_decisions, signals,
            model_version, training_start, training_end,
        )

        for p in problems[:5]:
            log.warning("%s", p)

        context.put("decisions", all_decisions)

        if problems and written == 0:
            return JobResult(
                self.name, JobStatus.FAILED,
                f"all {len(problems)} predictions rejected; first: {problems[0][:120]}",
                details={"problems": problems[:20]},
            )
        message = f"{skipped} conflicts skipped" if skipped else ""
        status = JobStatus.PARTIAL if problems else JobStatus.SUCCESS
        return JobResult(self.name, status, message, records=written,
                         details={"problems": problems[:20]})
