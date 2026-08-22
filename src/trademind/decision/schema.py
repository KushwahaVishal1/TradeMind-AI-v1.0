"""Decision types and the locked V1 semantics.

## V1 semantics — do not change without a version bump

::

    BUY                      open or increase a long
    HOLD                     maintain the existing position, whatever it is
    SELL                     exit or reduce a long
    SELL while flat          remain flat (never opens a short)
    rejected / stale input   preserve the existing position

The last rule is the one that matters operationally. When something goes wrong —
stale data, a missing feature, a model that failed to load — the safe action is
to *do nothing*, not to liquidate. Liquidating on error turns a data outage into
a realised loss and a round-trip cost. Preserving the position means an outage
costs nothing except the opportunity to act.

V1 is long-only. There is no SHORT signal, and SELL is an exit rather than a
reversal. Adding shorts would change the cost model, the risk engine, and the
label's meaning, so it is deferred rather than half-supported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class Signal(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"

    def __str__(self) -> str:
        return self.value


class RejectReason(str, Enum):
    """Why a decision fell back to preserving the current position."""

    NONE = "NONE"
    STALE_DATA = "STALE_DATA"
    MISSING_FEATURES = "MISSING_FEATURES"
    MISSING_PROBABILITY = "MISSING_PROBABILITY"
    BELOW_COST_FLOOR = "BELOW_COST_FLOOR"
    POSITION_LIMIT = "POSITION_LIMIT"
    NO_EXECUTION_SESSION = "NO_EXECUTION_SESSION"


class RiskBucket(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


@dataclass(frozen=True)
class Position:
    """Current holding in one symbol. Zero shares means flat."""

    symbol: str
    shares: float = 0.0
    cost_basis: float = 0.0

    @property
    def is_flat(self) -> bool:
        return abs(self.shares) < 1e-9

    @property
    def is_long(self) -> bool:
        return self.shares > 1e-9


@dataclass(frozen=True)
class DecisionInput:
    """Everything the engine needs for one symbol on one date."""

    symbol: str
    decision_date: date
    calibrated_probability: float | None = None
    expected_return: float | None = None
    volatility: float | None = None
    position: Position | None = None
    data_age_sessions: int = 0
    n_open_positions: int = 0
    features_complete: bool = True

    @property
    def current_position(self) -> Position:
        return self.position or Position(self.symbol)


@dataclass(frozen=True)
class Decision:
    """One decision, with the reasoning attached.

    ``rationale`` is not decoration. When a signal looks wrong three months
    later, the question is always *why did it fire*, and reconstructing that
    from thresholds and a probability is guesswork. Recording it costs a string.
    """

    symbol: str
    decision_date: date
    signal: Signal
    calibrated_probability: float | None = None
    expected_return: float | None = None
    target_weight: float = 0.0
    risk_bucket: RiskBucket = RiskBucket.MEDIUM
    rejected: bool = False
    reject_reason: RejectReason = RejectReason.NONE
    rationale: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        """A decision that changes the portfolio. HOLD never does."""
        return not self.rejected and self.signal is not Signal.HOLD

    def to_row(self) -> dict:
        return {
            "symbol": self.symbol,
            "decision_date": self.decision_date.isoformat(),
            "signal": self.signal.value,
            "calibrated_probability": self.calibrated_probability,
            "expected_return": self.expected_return,
            "target_weight": self.target_weight,
            "risk_bucket": self.risk_bucket.value,
            "rejected": self.rejected,
            "reject_reason": self.reject_reason.value,
            "rationale": self.rationale,
        }


def preserve_position(inp: DecisionInput, reason: RejectReason, detail: str = "") -> Decision:
    """The safe fallback: hold whatever is currently held.

    Every rejection path routes through here, so the "do nothing on error"
    guarantee lives in one place and is testable as one behaviour.
    """
    return Decision(
        symbol=inp.symbol,
        decision_date=inp.decision_date,
        signal=Signal.HOLD,
        calibrated_probability=inp.calibrated_probability,
        expected_return=inp.expected_return,
        target_weight=(0.0 if inp.current_position.is_flat else float("nan")),
        rejected=True,
        reject_reason=reason,
        rationale=detail or f"Rejected ({reason.value}); position preserved.",
    )
