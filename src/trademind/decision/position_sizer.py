"""Position sizing.

Three methods, in increasing order of how much rope they give you:

``fixed``
    Every position gets the same weight. Boring, robust, and the right default
    when the signal is weak -- sizing by confidence amplifies a noisy estimate.

``volatility_target``
    Weight scaled by ``reference_volatility / symbol_volatility``, so each
    position contributes similar risk. A 40%-vol name and a 15%-vol name at
    equal weight are not equal bets, and equal-weighting quietly concentrates
    portfolio risk in whatever is most volatile.

    The reference is the volatility at which a position takes full size. It has
    to be a *position-level* number, not a portfolio volatility target: dividing
    a 20% portfolio target by a 25% symbol volatility yields 0.8, which the
    weight cap then flattens to the maximum for every symbol, silently
    disabling the scaling. Set it near the universe's typical volatility.

``confidence``
    Scales with how far the probability sits above the buy threshold. Only
    defensible when probabilities are well calibrated -- with an ECE of 0.04,
    the difference between 0.55 and 0.60 is inside the noise, and sizing on it
    is sizing on nothing.

## Kelly is deliberately absent

Kelly sizing assumes the edge estimate is correct. With an IC near zero and
wide fold-to-fold variance, the estimated edge is mostly noise, and Kelly on a
noisy edge sizes aggressively in exactly the situations where it is most wrong.
Half-Kelly does not fix a wrong sign.
"""

from __future__ import annotations

import numpy as np

from .schema import Decision, DecisionInput, Signal

ANNUALISATION = np.sqrt(252)


def size_position(
    decision: Decision,
    inp: DecisionInput,
    method: str = "fixed",
    max_weight: float = 0.10,
    reference_volatility: float = 0.25,
    buy_threshold: float = 0.55,
) -> float:
    """Target portfolio weight for a decision. Zero for SELL, NaN for HOLD.

    NaN is meaningful: it means *leave the position alone*, which is different
    from a target of zero. Returning 0.0 for HOLD would silently liquidate every
    held position on every quiet day.
    """
    if decision.signal is Signal.SELL:
        return 0.0
    if decision.signal is Signal.HOLD:
        return 0.0 if inp.current_position.is_flat else float("nan")

    if method == "fixed":
        weight = max_weight

    elif method == "volatility_target":
        vol = inp.volatility
        if vol is None or not np.isfinite(vol) or vol <= 0:
            # No volatility estimate: half size rather than guessing. Silently
            # substituting a default would size an unknown-risk position as if
            # it were average.
            weight = max_weight * 0.5
        else:
            weight = max_weight * (reference_volatility / vol)

    elif method == "confidence":
        p = decision.calibrated_probability or 0.0
        headroom = max(1e-9, 1.0 - buy_threshold)
        scale = np.clip((p - buy_threshold) / headroom, 0.0, 1.0)
        # Floor at half weight: a signal worth taking is worth taking properly.
        weight = max_weight * (0.5 + 0.5 * scale)

    else:
        raise ValueError(f"Unknown sizing method: {method}")

    return float(np.clip(weight, 0.0, max_weight))
