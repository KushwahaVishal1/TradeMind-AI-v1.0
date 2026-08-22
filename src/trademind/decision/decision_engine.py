"""Decision engine: risk gates -> signal -> sizing -> risk label.

The ordering is fixed and the reason is in ``risk_engine.py``: gates run before
the signal exists, so a rejected input never produces a tradeable number.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import pandas as pd

from .position_sizer import size_position
from .risk_engine import RiskEngine, classify_risk
from .schema import Decision, DecisionInput, Signal
from .signal_generator import generate_signal
from .threshold_optimizer import Thresholds

log = logging.getLogger(__name__)


@dataclass
class DecisionEngine:
    """Turns model output into portfolio instructions."""

    thresholds: Thresholds
    risk_engine: RiskEngine = None
    sizing_method: str = "fixed"
    max_weight: float = 0.10
    reference_volatility: float = 0.25

    def __post_init__(self) -> None:
        if self.risk_engine is None:
            self.risk_engine = RiskEngine()

    def decide(self, inp: DecisionInput) -> Decision:
        rejected = self.risk_engine.check(inp)
        if rejected is not None:
            return replace(rejected, risk_bucket=classify_risk(inp.volatility))

        decision = generate_signal(inp, self.thresholds)
        weight = size_position(
            decision,
            inp,
            method=self.sizing_method,
            max_weight=self.max_weight,
            reference_volatility=self.reference_volatility,
            buy_threshold=self.thresholds.buy,
        )
        return replace(
            decision,
            target_weight=weight,
            risk_bucket=classify_risk(inp.volatility),
        )

    def decide_batch(self, inputs: list[DecisionInput]) -> list[Decision]:
        """Decide for a whole cross-section on one date.

        Processes higher-probability candidates first so that when the position
        cap binds, it binds on the weakest signals rather than on whichever
        symbol happens to sort first alphabetically.
        """
        ordered = sorted(
            inputs,
            key=lambda i: (
                i.calibrated_probability is None,
                -(i.calibrated_probability or 0.0),
            ),
        )

        open_count = sum(1 for i in inputs if not i.current_position.is_flat)
        out = []
        for inp in ordered:
            decision = self.decide(replace(inp, n_open_positions=open_count))
            if decision.signal is Signal.BUY and not decision.rejected:
                if inp.current_position.is_flat:
                    open_count += 1
            elif decision.signal is Signal.SELL and not decision.rejected and inp.current_position.is_long:
                open_count -= 1
            out.append(decision)
        return out

    def to_frame(self, decisions: list[Decision]) -> pd.DataFrame:
        return pd.DataFrame([d.to_row() for d in decisions])
