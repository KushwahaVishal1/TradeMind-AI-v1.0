"""Prediction persistence, outcome resolution, and run tracking.

The idempotency guarantee Phase 9 requires lives here.

``prediction_id`` is a deterministic hash of the natural key, so writing the
same prediction twice cannot create two rows even across processes or machines.
Re-running yesterday's job is free.

The harder case is a *conflicting* rewrite: same natural key, different numbers.
That means either the upstream data was revised or the pipeline is
non-deterministic — both of which are bugs worth surfacing. The default is to
raise. Silent overwriting would let a model quietly rewrite its own track
record, which is exactly the failure mode monitoring exists to catch.
"""

from __future__ import annotations

import hashlib
import sqlite3
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

SIGNALS = ("BUY", "HOLD", "SELL")


class PredictionConflictError(RuntimeError):
    """A prediction already exists for this key with different values."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Prediction:
    """One decision for one symbol on one day, with full lineage."""

    symbol: str
    prediction_date: str  # ISO date, the 't' whose features were used
    model_version: str
    feature_version: str
    decision_version: str
    threshold_version: str
    signal: str

    execution_date: str | None = None  # 't+1' — the session we act on
    predicted_probability: float | None = None
    calibrated_probability: float | None = None
    predicted_return: float | None = None
    target_weight: float | None = None
    risk_bucket: str | None = None
    training_start: str | None = None
    training_end: str | None = None
    run_id: str | None = None

    def __post_init__(self) -> None:
        if self.signal not in SIGNALS:
            raise ValueError(f"signal must be one of {SIGNALS}, got {self.signal!r}")
        for name in ("predicted_probability", "calibrated_probability"):
            v = getattr(self, name)
            if v is not None and not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {v}")
        if self.execution_date is not None and self.execution_date <= self.prediction_date:
            # Same-day execution is the second-most-common form of look-ahead
            # bias after leaky features. Refuse it at the storage boundary.
            raise ValueError(
                f"execution_date {self.execution_date} must be strictly after "
                f"prediction_date {self.prediction_date}; same-session execution "
                "is look-ahead bias."
            )

    @property
    def natural_key(self) -> tuple[str, ...]:
        return (
            self.symbol,
            self.prediction_date,
            self.model_version,
            self.feature_version,
            self.decision_version,
            self.threshold_version,
        )

    @property
    def prediction_id(self) -> str:
        digest = hashlib.sha256("|".join(self.natural_key).encode()).hexdigest()
        return digest[:32]


# Fields compared when deciding whether a re-write conflicts.
_PAYLOAD_FIELDS = (
    "signal",
    "predicted_probability",
    "calibrated_probability",
    "predicted_return",
    "target_weight",
    "execution_date",
)


def _values_differ(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is not b
        return abs(float(a) - float(b)) > 1e-12
    return a != b


class PredictionStore:
    """Repository over the predictions/outcomes/runs tables."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    # -- runs ------------------------------------------------------------

    def start_run(
        self,
        mode: str,
        config_hash: str,
        git_commit: str | None = None,
        python_version: str | None = None,
    ) -> str:
        run_id = uuid.uuid4().hex[:16]
        self.conn.execute(
            "INSERT INTO runs (run_id, mode, started_at, status, git_commit, "
            "config_hash, python_version) VALUES (?, ?, ?, 'RUNNING', ?, ?, ?)",
            (run_id, mode, _now(), git_commit, config_hash, python_version),
        )
        return run_id

    def finish_run(self, run_id: str, status: str, error: str | None = None) -> None:
        self.conn.execute(
            "UPDATE runs SET finished_at = ?, status = ?, error = ? WHERE run_id = ?",
            (_now(), status, error, run_id),
        )

    # -- predictions -----------------------------------------------------

    def save(self, pred: Prediction, *, allow_overwrite: bool = False) -> str:
        """Insert a prediction. Returns its id.

        Writing an identical prediction again is a silent no-op. Writing a
        different one under the same key raises unless explicitly permitted.
        """
        pid = pred.prediction_id
        existing = self.conn.execute(
            "SELECT * FROM predictions WHERE prediction_id = ?", (pid,)
        ).fetchone()

        if existing is not None:
            diffs = {
                f: (existing[f], getattr(pred, f))
                for f in _PAYLOAD_FIELDS
                if _values_differ(existing[f], getattr(pred, f))
            }
            if not diffs:
                return pid  # true idempotent replay
            if not allow_overwrite:
                raise PredictionConflictError(
                    f"Prediction {pid} ({pred.symbol} {pred.prediction_date}) "
                    f"already exists with different values: {diffs}. "
                    "Bump a version field to record a new prediction, or pass "
                    "allow_overwrite=True if the source data was corrected."
                )
            sets = ", ".join(f"{f} = ?" for f in _PAYLOAD_FIELDS)
            self.conn.execute(
                f"UPDATE predictions SET {sets} WHERE prediction_id = ?",
                (*(getattr(pred, f) for f in _PAYLOAD_FIELDS), pid),
            )
            return pid

        row = asdict(pred)
        row["prediction_id"] = pid
        row["created_at"] = _now()
        row["status"] = "PREDICTED"
        cols = ", ".join(row)
        marks = ", ".join("?" * len(row))
        self.conn.execute(
            f"INSERT INTO predictions ({cols}) VALUES ({marks})", tuple(row.values())
        )
        return pid

    def save_many(
        self, preds: Iterable[Prediction], *, allow_overwrite: bool = False
    ) -> list[str]:
        return [self.save(p, allow_overwrite=allow_overwrite) for p in preds]

    def get(self, prediction_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM predictions WHERE prediction_id = ?", (prediction_id,)
        ).fetchone()

    def pending(self, as_of: str | None = None) -> list[sqlite3.Row]:
        """Predictions awaiting an outcome, oldest first."""
        sql = "SELECT * FROM predictions WHERE status = 'PREDICTED'"
        params: Sequence[Any] = ()
        if as_of:
            sql += " AND execution_date <= ?"
            params = (as_of,)
        return self.conn.execute(sql + " ORDER BY prediction_date", params).fetchall()

    # -- outcomes --------------------------------------------------------

    def resolve(self, prediction_id: str, actual_return: float) -> None:
        """Attach the realized outcome and flip the prediction to RESOLVED."""
        pred = self.get(prediction_id)
        if pred is None:
            raise KeyError(f"Unknown prediction_id {prediction_id}")
        if pred["status"] == "RESOLVED":
            raise RuntimeError(
                f"Prediction {prediction_id} is already resolved; outcomes are write-once."
            )

        actual_direction = 1 if actual_return > 0 else 0
        prob = pred["calibrated_probability"]
        if prob is None:
            prob = pred["predicted_probability"]
        direction_correct = (
            None if prob is None else int((prob >= 0.5) == (actual_direction == 1))
        )
        err = (
            None
            if pred["predicted_return"] is None
            else float(pred["predicted_return"]) - actual_return
        )

        self.conn.execute(
            "INSERT INTO outcomes (prediction_id, resolved_at, actual_return, "
            "actual_direction, direction_correct, return_error) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (prediction_id, _now(), actual_return, actual_direction, direction_correct, err),
        )
        self.conn.execute(
            "UPDATE predictions SET status = 'RESOLVED' WHERE prediction_id = ?",
            (prediction_id,),
        )

    def resolved(self, since: str | None = None) -> list[sqlite3.Row]:
        sql = "SELECT * FROM resolved_predictions"
        params: Sequence[Any] = ()
        if since:
            sql += " WHERE prediction_date >= ?"
            params = (since,)
        return self.conn.execute(sql + " ORDER BY prediction_date", params).fetchall()

    # -- data quality ----------------------------------------------------

    def log_issue(
        self,
        severity: str,
        code: str,
        detail: str = "",
        symbol: str | None = None,
        issue_date: str | None = None,
        run_id: str | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO data_quality_issues (run_id, symbol, issue_date, "
            "severity, code, detail, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, symbol, issue_date, severity, code, detail, _now()),
        )
