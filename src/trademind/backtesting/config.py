"""Backtest configuration and transaction costs.

Costs are tracked as three separate components rather than one blended number,
because they behave differently and the distinction matters when explaining a
result:

``commission``
    Broker fee. Predictable, symmetric, the smallest of the three.
``spread``
    Half the bid-ask spread, paid on entry and exit. A property of the market.
``slippage``
    Difference between the decision price and the achieved price. Grows with
    order size and with volatility, and is the component most likely to be
    understated in a backtest.

Blending them into "transaction costs: 13 bps" hides which assumption is doing
the work. If a strategy is marginal, the honest question is *which* cost
component would have to be wrong for the result to change, and a single number
cannot answer it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostConfig:
    """Per-side costs in basis points."""

    commission_bps: float = 3.0
    spread_bps: float = 5.0
    slippage_bps: float = 5.0

    @property
    def total_bps(self) -> float:
        return self.commission_bps + self.spread_bps + self.slippage_bps

    @property
    def per_side(self) -> float:
        return self.total_bps / 10000.0

    @property
    def round_trip(self) -> float:
        return 2.0 * self.per_side


@dataclass(frozen=True)
class CostBreakdown:
    """Costs of one fill, itemised."""

    commission: float = 0.0
    spread: float = 0.0
    slippage: float = 0.0

    @property
    def total(self) -> float:
        return self.commission + self.spread + self.slippage

    def __add__(self, other: CostBreakdown) -> CostBreakdown:
        return CostBreakdown(
            self.commission + other.commission,
            self.spread + other.spread,
            self.slippage + other.slippage,
        )


def compute_costs(notional: float, costs: CostConfig) -> CostBreakdown:
    """Itemised cost of trading ``notional`` currency on one side."""
    n = abs(notional)
    return CostBreakdown(
        commission=n * costs.commission_bps / 10000.0,
        spread=n * costs.spread_bps / 10000.0,
        slippage=n * costs.slippage_bps / 10000.0,
    )


@dataclass(frozen=True)
class BacktestConfig:
    """Everything the engine needs, in one immutable object."""

    initial_capital: float = 1_000_000.0
    costs: CostConfig = CostConfig()
    max_position_weight: float = 0.10
    max_positions: int = 10

    # Fills happen at the open of the session after the decision. Holding cash
    # earns nothing; a rate would be another assumption to defend and it does
    # not change the comparison against benchmarks that are fully invested.
    allow_fractional_shares: bool = False

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if not 0 < self.max_position_weight <= 1.0:
            raise ValueError("max_position_weight must be in (0, 1]")

    @classmethod
    def from_config(cls, cfg) -> BacktestConfig:
        return cls(
            initial_capital=cfg.get("backtest.initial_capital"),
            costs=CostConfig(
                commission_bps=cfg.get("backtest.commission_bps"),
                spread_bps=cfg.get("backtest.spread_bps"),
                slippage_bps=cfg.get("backtest.slippage_bps"),
            ),
            max_position_weight=cfg.get("backtest.max_position_weight"),
            max_positions=cfg.get("decision.max_positions", 10),
        )
