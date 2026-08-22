"""Risk gates and risk classification.

Gates run *before* the signal is generated, not after. A stale-data check that
runs after the model has already produced a signal is a check on a number that
should never have been computed — and in an agentic pipeline, computing it
creates the temptation to use it.

Every gate failure routes through ``preserve_position``, so the "do nothing on
error" guarantee holds uniformly.

## The staleness gate is the one that earns its place

Market data goes missing. A holiday the calendar did not know about, a provider
outage, a symbol that stopped trading. Without this gate, the feature pipeline
happily computes a 20-day return from a series whose last five sessions are
absent, the model produces a confident probability from it, and the system
trades on a picture of the market that is a week old.
"""

from __future__ import annotations

import numpy as np

from .schema import (
    Decision,
    DecisionInput,
    RejectReason,
    RiskBucket,
    preserve_position,
)

# Beyond this many sessions since the last bar, the features describe a market
# that no longer exists.
MAX_DATA_AGE_SESSIONS = 3

# Annualised volatility boundaries for the risk buckets.
VOL_LOW = 0.20
VOL_MEDIUM = 0.35
VOL_HIGH = 0.60


def classify_risk(volatility: float | None) -> RiskBucket:
    """Bucket a position by annualised volatility.

    Reported on every decision rather than used to block one. The buckets are
    for the dashboard and the report; blocking on volatility would be a
    strategy decision, and this system does not have the evidence to make it.
    """
    if volatility is None or not np.isfinite(volatility):
        return RiskBucket.MEDIUM
    if volatility < VOL_LOW:
        return RiskBucket.LOW
    if volatility < VOL_MEDIUM:
        return RiskBucket.MEDIUM
    if volatility < VOL_HIGH:
        return RiskBucket.HIGH
    return RiskBucket.EXTREME


class RiskEngine:
    """Pre-decision gates.

    Parameters
    ----------
    max_positions
        Cap on simultaneously open positions. Applies only to *new* entries —
        an existing position is never force-closed by the cap, because
        liquidating on a limit breach turns a portfolio-construction constraint
        into an unplanned trade.
    max_data_age_sessions
        Refuse to act on bars older than this.
    """

    def __init__(
        self,
        max_positions: int = 10,
        max_data_age_sessions: int = MAX_DATA_AGE_SESSIONS,
    ) -> None:
        self.max_positions = max_positions
        self.max_data_age_sessions = max_data_age_sessions

    def check(self, inp: DecisionInput) -> Decision | None:
        """Return a rejection Decision, or None if the input passes."""
        if inp.data_age_sessions > self.max_data_age_sessions:
            return preserve_position(
                inp,
                RejectReason.STALE_DATA,
                f"Latest bar is {inp.data_age_sessions} sessions old "
                f"(limit {self.max_data_age_sessions}); features describe a "
                "market that has moved on.",
            )

        if not inp.features_complete:
            return preserve_position(
                inp,
                RejectReason.MISSING_FEATURES,
                "Feature row is incomplete; the model would be extrapolating "
                "from imputed values.",
            )

        if inp.calibrated_probability is not None and not np.isfinite(
            inp.calibrated_probability
        ):
            return preserve_position(
                inp,
                RejectReason.MISSING_PROBABILITY,
                "Calibrated probability is not finite.",
            )

        # Position cap: blocks new entries only.
        if inp.current_position.is_flat and inp.n_open_positions >= self.max_positions:
            return preserve_position(
                inp,
                RejectReason.POSITION_LIMIT,
                f"Already holding {inp.n_open_positions} positions "
                f"(limit {self.max_positions}); no new entries.",
            )

        return None
