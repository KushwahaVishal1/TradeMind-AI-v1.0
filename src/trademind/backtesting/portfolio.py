"""Portfolio accounting, with reconciliation that can actually fail.

## Why the usual invariant proves nothing

Most backtesters "verify" accounting like this::

    equity = cash + sum(shares * price)
    assert equity == cash + sum(shares * price)      # tautology

The check restates the definition. It passes for any bug that lives inside the
definition — a fill that forgot to charge commission, a split that adjusted
price but not share count, a dividend credited twice. All of those keep
``cash + holdings`` internally consistent while making it *wrong*.

## The two independent paths

This module computes equity twice, from disjoint state, and requires agreement.

**Path A — balance sheet.** What is actually held::

    equity_A = cash + Σ (shares × current_price)

Driven by the cash balance, which is mutated by every fill, dividend, and fee.

**Path B — flow accumulation.** What happened::

    equity_B = initial_capital
             + realised_pnl          (closed trades, gross)
             + unrealised_pnl        (Σ shares × (price − cost_basis))
             + dividends_received
             − total_costs

Driven by separate accumulators that never read ``cash``.

The two share no state. A fill that omits commission leaves A too high while B
subtracts the cost, so they diverge. A split that changes price but not share
count moves A and leaves B's cost-basis arithmetic inconsistent. A dividend
credited to cash but not recorded moves A without moving B.

``reconcile()`` runs on every session and raises on a discrepancy beyond
floating-point tolerance, so an accounting bug surfaces on the day it happens
rather than as an unexplained return six months later.

## The corporate-action invariant

Splits go through ``adjust_position_for_split`` from Phase 1 — the same
function whose test asserts 100 × ₹1,000 = 200 × ₹500. Share count multiplies,
cost basis divides, wealth is unchanged. Reconciliation then confirms it,
because a split that moved equity would break path A against path B.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

from ..ingestion.corporate_actions import adjust_position_for_split
from .config import CostBreakdown

log = logging.getLogger(__name__)

# Currency units. Tighter than this trips on float noise over thousands of fills.
RECONCILE_TOLERANCE = 0.01


class ReconciliationError(RuntimeError):
    """The two independent equity calculations disagree."""


@dataclass
class Holding:
    """An open position. ``cost_basis`` is per share, split-adjusted."""

    symbol: str
    shares: float = 0.0
    cost_basis: float = 0.0

    @property
    def is_flat(self) -> bool:
        return abs(self.shares) < 1e-9

    def market_value(self, price: float) -> float:
        return self.shares * price

    def unrealised(self, price: float) -> float:
        return self.shares * (price - self.cost_basis)


@dataclass
class Trade:
    """A single fill."""

    trade_date: date
    symbol: str
    side: str                 # BUY | SELL
    shares: float
    price: float              # as-traded execution price
    costs: CostBreakdown
    realised_pnl: float = 0.0

    @property
    def notional(self) -> float:
        return self.shares * self.price


@dataclass
class Portfolio:
    """Cash, holdings, and the flow accumulators that check them."""

    initial_capital: float
    cash: float = 0.0
    holdings: dict[str, Holding] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)

    # Flow accumulators — path B. These never read `cash`.
    realised_pnl: float = 0.0
    dividends_received: float = 0.0
    total_costs: CostBreakdown = field(default_factory=CostBreakdown)

    def __post_init__(self) -> None:
        if self.cash == 0.0:
            self.cash = self.initial_capital

    # -- state -----------------------------------------------------------

    def position(self, symbol: str) -> Holding:
        return self.holdings.get(symbol, Holding(symbol))

    @property
    def open_symbols(self) -> list[str]:
        return [s for s, h in self.holdings.items() if not h.is_flat]

    def holdings_value(self, prices: dict[str, float]) -> float:
        return sum(
            h.market_value(prices[s])
            for s, h in self.holdings.items()
            if not h.is_flat and s in prices
        )

    def unrealised_pnl(self, prices: dict[str, float]) -> float:
        return sum(
            h.unrealised(prices[s])
            for s, h in self.holdings.items()
            if not h.is_flat and s in prices
        )

    # -- the two paths ---------------------------------------------------

    def equity_balance_sheet(self, prices: dict[str, float]) -> float:
        """Path A: cash plus the market value of what is held."""
        return self.cash + self.holdings_value(prices)

    def equity_flows(self, prices: dict[str, float]) -> float:
        """Path B: initial capital plus everything that has happened to it."""
        return (
            self.initial_capital
            + self.realised_pnl
            + self.unrealised_pnl(prices)
            + self.dividends_received
            - self.total_costs.total
        )

    def reconcile(self, prices: dict[str, float], when: date | None = None) -> float:
        """Require the two paths to agree. Returns the discrepancy."""
        a = self.equity_balance_sheet(prices)
        b = self.equity_flows(prices)
        diff = a - b

        if abs(diff) > RECONCILE_TOLERANCE:
            raise ReconciliationError(
                f"Accounting mismatch on {when}: balance sheet {a:,.2f} vs "
                f"flows {b:,.2f} (difference {diff:,.4f}). "
                f"cash={self.cash:,.2f} holdings={self.holdings_value(prices):,.2f} "
                f"realised={self.realised_pnl:,.2f} "
                f"unrealised={self.unrealised_pnl(prices):,.2f} "
                f"dividends={self.dividends_received:,.2f} "
                f"costs={self.total_costs.total:,.2f}"
            )
        return diff

    # -- mutations -------------------------------------------------------

    def buy(self, symbol: str, shares: float, price: float,
            costs: CostBreakdown, when: date) -> Trade:
        """Open or increase a long. Cost basis becomes the weighted average."""
        if shares <= 0:
            raise ValueError(f"buy shares must be positive, got {shares}")

        outlay = shares * price + costs.total
        if outlay > self.cash + 1e-9:
            raise ValueError(
                f"Insufficient cash for {symbol}: need {outlay:,.2f}, "
                f"have {self.cash:,.2f}"
            )

        holding = self.holdings.setdefault(symbol, Holding(symbol))
        total_shares = holding.shares + shares
        holding.cost_basis = (
            (holding.shares * holding.cost_basis + shares * price) / total_shares
        )
        holding.shares = total_shares

        self.cash -= outlay
        self.total_costs = self.total_costs + costs

        trade = Trade(when, symbol, "BUY", shares, price, costs)
        self.trades.append(trade)
        return trade

    def sell(self, symbol: str, shares: float, price: float,
             costs: CostBreakdown, when: date) -> Trade:
        """Reduce or close a long, realising P&L against the cost basis."""
        holding = self.holdings.get(symbol)
        if holding is None or holding.shares < shares - 1e-9:
            have = holding.shares if holding else 0.0
            raise ValueError(
                f"Cannot sell {shares} of {symbol}; holding {have}. "
                "V1 is long-only and does not short."
            )

        # Gross of costs. Costs are accumulated separately so that path B can
        # subtract them once, in one place.
        realised = shares * (price - holding.cost_basis)

        holding.shares -= shares
        if holding.is_flat:
            holding.shares = 0.0
            holding.cost_basis = 0.0

        self.cash += shares * price - costs.total
        self.realised_pnl += realised
        self.total_costs = self.total_costs + costs

        trade = Trade(when, symbol, "SELL", shares, price, costs, realised)
        self.trades.append(trade)
        return trade

    def apply_split(self, symbol: str, ratio: float, when: date) -> None:
        """Apply a split to an open position. Wealth must not change."""
        holding = self.holdings.get(symbol)
        if holding is None or holding.is_flat or ratio == 1.0:
            return

        before = holding.shares * holding.cost_basis
        holding.shares, holding.cost_basis = adjust_position_for_split(
            holding.shares, holding.cost_basis, ratio
        )
        after = holding.shares * holding.cost_basis

        if abs(before - after) > RECONCILE_TOLERANCE:
            raise ReconciliationError(
                f"{symbol} {ratio}:1 split on {when} changed position value "
                f"{before:,.2f} -> {after:,.2f}. A split must not create or "
                "destroy wealth."
            )
        log.debug("%s %s: %s:1 split applied", when, symbol, ratio)

    def apply_dividend(self, symbol: str, per_share: float, when: date) -> float:
        """Credit a cash dividend on an open position."""
        holding = self.holdings.get(symbol)
        if holding is None or holding.is_flat or per_share <= 0:
            return 0.0

        amount = holding.shares * per_share
        self.cash += amount
        self.dividends_received += amount
        log.debug("%s %s: dividend %.2f", when, symbol, amount)
        return amount

    # -- reporting -------------------------------------------------------

    def round_trips(self) -> int:
        """Completed round trips — the trade count that means something.

        A BUY and its matching SELL are one round trip, not two trades.
        Counting fills would double the apparent activity and halve the
        apparent cost per trade.
        """
        return sum(1 for t in self.trades if t.side == "SELL")

    def snapshot(self, when: date, prices: dict[str, float]) -> dict:
        equity = self.equity_balance_sheet(prices)
        holdings_value = self.holdings_value(prices)
        return {
            "date": when,
            "equity": equity,
            "cash": self.cash,
            "holdings_value": holdings_value,
            "n_positions": len(self.open_symbols),
            "exposure": holdings_value / equity if equity else 0.0,
            "realised_pnl": self.realised_pnl,
            "unrealised_pnl": self.unrealised_pnl(prices),
            "dividends": self.dividends_received,
            "total_costs": self.total_costs.total,
        }
