"""Probability and expected return -> BUY / HOLD / SELL.

Two gates, and both must pass for a BUY:

1. ``calibrated_probability >= buy`` -- the model is confident enough.
2. ``expected_return >= min_expected_return`` -- the edge clears the cost floor.

The second gate is what makes the first one meaningful. A 0.72 probability on a
move worth 3 bps is a confident prediction of a losing trade: confidence about
direction says nothing about magnitude, and the round trip costs 26 bps
regardless. Systems that threshold on probability alone trade constantly on
marginal edges and bleed out on costs while looking accurate.
"""

from __future__ import annotations

from .schema import Decision, DecisionInput, RejectReason, Signal, preserve_position
from .threshold_optimizer import Thresholds


def generate_signal(inp: DecisionInput, thresholds: Thresholds) -> Decision:
    """Apply the locked V1 semantics to one symbol on one date."""
    position = inp.current_position

    if inp.calibrated_probability is None:
        return preserve_position(inp, RejectReason.MISSING_PROBABILITY)

    p = inp.calibrated_probability
    expected = inp.expected_return

    # --- BUY: confident AND the edge clears costs -----------------------
    if p >= thresholds.buy:
        if expected is None:
            return preserve_position(
                inp,
                RejectReason.MISSING_PROBABILITY,
                "Probability clears the buy threshold but no expected return "
                "is available to check against the cost floor.",
            )
        if expected < thresholds.min_expected_return:
            # The important rejection. Confident, but not worth the round trip.
            return Decision(
                symbol=inp.symbol,
                decision_date=inp.decision_date,
                signal=Signal.HOLD,
                calibrated_probability=p,
                expected_return=expected,
                target_weight=0.0 if position.is_flat else float("nan"),
                rejected=True,
                reject_reason=RejectReason.BELOW_COST_FLOOR,
                rationale=(
                    f"p={p:.3f} clears buy={thresholds.buy:.3f}, but expected "
                    f"return {expected:.5f} is below the cost floor "
                    f"{thresholds.min_expected_return:.5f}. Trading here loses "
                    "money on costs."
                ),
            )
        return Decision(
            symbol=inp.symbol,
            decision_date=inp.decision_date,
            signal=Signal.BUY,
            calibrated_probability=p,
            expected_return=expected,
            rationale=(
                f"p={p:.3f} >= buy={thresholds.buy:.3f} and expected return "
                f"{expected:.5f} >= floor {thresholds.min_expected_return:.5f}."
            ),
        )

    # --- SELL: conviction has lapsed ------------------------------------
    if p < thresholds.sell:
        if position.is_flat:
            # Locked semantics: SELL while flat stays flat. Never opens a short.
            return Decision(
                symbol=inp.symbol,
                decision_date=inp.decision_date,
                signal=Signal.SELL,
                calibrated_probability=p,
                expected_return=expected,
                target_weight=0.0,
                rationale=(
                    f"p={p:.3f} < sell={thresholds.sell:.3f}; already flat, so "
                    "no action. V1 is long-only."
                ),
            )
        return Decision(
            symbol=inp.symbol,
            decision_date=inp.decision_date,
            signal=Signal.SELL,
            calibrated_probability=p,
            expected_return=expected,
            target_weight=0.0,
            rationale=f"p={p:.3f} < sell={thresholds.sell:.3f}; exiting long.",
        )

    # --- HOLD: inside the band ------------------------------------------
    return Decision(
        symbol=inp.symbol,
        decision_date=inp.decision_date,
        signal=Signal.HOLD,
        calibrated_probability=p,
        expected_return=expected,
        target_weight=float("nan") if position.is_long else 0.0,
        rationale=(
            f"p={p:.3f} sits in the hold band [{thresholds.sell:.3f}, {thresholds.buy:.3f})."
        ),
    )
