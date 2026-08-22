"""Order execution: decisions at t fill at the open of t+1, at as-traded prices.

Two rules, both enforced here rather than assumed:

**No same-session fills.** A decision made from the close of *t* cannot fill on
*t*. ``execute_orders`` takes the execution session explicitly and the engine
never passes it the decision date.

**Raw prices only.** Fills use ``open_raw`` -- the price actually quoted that
morning -- never ``adj_close`` or ``open_split``. Adjusted series are restated
retroactively by every later dividend, so filling at one means transacting at a
price that did not exist on the day and that will change again next quarter.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date

from .config import BacktestConfig, compute_costs
from .portfolio import Portfolio, Trade

log = logging.getLogger(__name__)


class ExecutionError(RuntimeError):
    """An order could not be executed and the reason is not recoverable."""


@dataclass(frozen=True)
class Order:
    """An instruction produced on ``decision_date``, to fill on a later session."""

    symbol: str
    decision_date: date
    target_weight: float  # NaN means "leave the position alone"
    signal: str

    @property
    def is_exit(self) -> bool:
        return self.target_weight == 0.0 and self.signal == "SELL"


def size_order(
    target_weight: float,
    equity: float,
    price: float,
    current_shares: float,
    config: BacktestConfig,
) -> float:
    """Share delta needed to reach ``target_weight``. Positive buys.

    Whole shares unless fractional trading is enabled. Rounding *down* on
    entries matters: rounding up would occasionally require more cash than the
    target weight allows and fail the cash check for reasons that have nothing
    to do with the strategy.
    """
    if price <= 0:
        raise ExecutionError(f"Cannot size an order at price {price}")

    target_shares = (target_weight * equity) / price
    if not config.allow_fractional_shares:
        target_shares = math.floor(target_shares)

    return target_shares - current_shares


def execute_orders(
    orders: list[Order],
    portfolio: Portfolio,
    execution_date: date,
    open_prices: dict[str, float],
    config: BacktestConfig,
) -> list[Trade]:
    """Fill orders at the open of ``execution_date``.

    Sells run before buys so that proceeds are available to fund entries on the
    same session. Without that ordering a fully invested portfolio could never
    rotate, which would be an artifact of the loop rather than the strategy.
    """
    for order in orders:
        if order.decision_date >= execution_date:
            raise ExecutionError(
                f"{order.symbol}: decision date {order.decision_date} is not "
                f"before execution date {execution_date}. Same-session "
                "execution is look-ahead bias."
            )

    equity = portfolio.equity_balance_sheet(open_prices)
    fills: list[Trade] = []

    actionable = [o for o in orders if not math.isnan(o.target_weight)]
    sells = [o for o in actionable if o.target_weight == 0.0]
    buys = [o for o in actionable if o.target_weight > 0.0]

    for order in sells:
        holding = portfolio.position(order.symbol)
        if holding.is_flat:
            continue
        price = open_prices.get(order.symbol)
        if price is None or price <= 0:
            log.warning(
                "%s: no execution price for %s; position preserved",
                execution_date,
                order.symbol,
            )
            continue
        costs = compute_costs(holding.shares * price, config.costs)
        fills.append(
            portfolio.sell(order.symbol, holding.shares, price, costs, execution_date)
        )

    for order in buys:
        price = open_prices.get(order.symbol)
        if price is None or price <= 0:
            log.warning(
                "%s: no execution price for %s; skipping entry", execution_date, order.symbol
            )
            continue

        holding = portfolio.position(order.symbol)
        weight = min(order.target_weight, config.max_position_weight)
        delta = size_order(weight, equity, price, holding.shares, config)
        if delta <= 0:
            continue

        costs = compute_costs(delta * price, config.costs)
        outlay = delta * price + costs.total
        if outlay > portfolio.cash:
            # Scale down to what cash allows rather than failing the session.
            affordable = math.floor(portfolio.cash / (price * (1 + config.costs.per_side)))
            if affordable <= 0:
                log.debug("%s: insufficient cash for %s", execution_date, order.symbol)
                continue
            delta = affordable
            costs = compute_costs(delta * price, config.costs)

        fills.append(portfolio.buy(order.symbol, delta, price, costs, execution_date))

    return fills
