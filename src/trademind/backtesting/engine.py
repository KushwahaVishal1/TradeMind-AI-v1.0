"""The event loop.

Session ordering is the whole design, and it encodes the timing contract:

1. **Corporate actions** — splits and dividends on today's ex-date, applied
   before anything is priced or traded.
2. **Execute** yesterday's orders at *today's open*, using ``open_raw``.
3. **Mark to market** at today's close and reconcile both equity paths.
4. **Decide** from today's data, producing orders for *tomorrow*.

Step 4 comes last on purpose. If decisions were generated before execution, a
decision made from today's close could fill at today's open — a fill at a price
that occurred hours before the information existed. Putting decisions at the end
of the session makes that mechanically impossible rather than merely forbidden.

Reconciliation runs every session (step 3), so an accounting error surfaces on
the day it occurs with the date attached.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..decision.decision_engine import DecisionEngine
from ..decision.schema import DecisionInput, Position, Signal
from .config import BacktestConfig
from .execution import Order, execute_orders
from .portfolio import Portfolio

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Equity curve, trades, and the accounting proof."""

    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    decisions: pd.DataFrame
    config: BacktestConfig
    max_reconciliation_error: float = 0.0
    benchmarks: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve["equity"].iloc[-1]) if len(self.equity_curve) else 0.0

    @property
    def total_return(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        first = float(self.equity_curve["equity"].iloc[0])
        return self.final_equity / first - 1.0 if first else 0.0


def _prices_on(panel: pd.DataFrame, when: pd.Timestamp, col: str) -> dict[str, float]:
    rows = panel[panel["date"] == when]
    return {
        r.symbol: float(getattr(r, col))
        for r in rows.itertuples()
        if getattr(r, col) is not None and np.isfinite(getattr(r, col)) and getattr(r, col) > 0
    }


def run_backtest(
    panel: pd.DataFrame,
    signals: pd.DataFrame,
    engine: DecisionEngine,
    config: BacktestConfig,
) -> BacktestResult:
    """Run the event loop over a feature panel and a set of calibrated signals.

    ``panel`` supplies prices and corporate actions; ``signals`` supplies
    ``calibrated`` probability and optionally ``expected_return``, keyed on
    ``(date, symbol)``.
    """
    required = {"date", "symbol", "open_raw", "close_raw", "split_ratio", "dividend"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Panel lacks required columns: {sorted(missing)}")

    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    signal_lookup = (
        {(r.date, r.symbol): r for r in signals.itertuples()} if not signals.empty else {}
    )

    sessions = sorted(panel["date"].unique())
    portfolio = Portfolio(config.initial_capital)

    pending: list[Order] = []
    curve, decision_rows = [], []
    max_error = 0.0

    for when in sessions:
        session_date = pd.Timestamp(when).date()
        today = panel[panel["date"] == when]

        # 1. Corporate actions, before anything is priced.
        for row in today.itertuples():
            ratio = float(row.split_ratio)
            if ratio != 1.0:
                portfolio.apply_split(row.symbol, ratio, session_date)
            dividend = float(row.dividend)
            if dividend > 0:
                portfolio.apply_dividend(row.symbol, dividend, session_date)

        # 2. Fill yesterday's orders at today's open.
        if pending:
            opens = _prices_on(panel, when, "open_raw")
            execute_orders(pending, portfolio, session_date, opens, config)
            pending = []

        # 3. Mark to market and reconcile both paths.
        closes = _prices_on(panel, when, "close_raw")
        if closes:
            max_error = max(max_error, abs(portfolio.reconcile(closes, session_date)))
            curve.append(portfolio.snapshot(session_date, closes))

        # 4. Decide from today's data, for tomorrow.
        inputs = []
        for row in today.itertuples():
            sig = signal_lookup.get((when, row.symbol))
            if sig is None:
                continue
            holding = portfolio.position(row.symbol)
            inputs.append(
                DecisionInput(
                    symbol=row.symbol,
                    decision_date=session_date,
                    calibrated_probability=float(getattr(sig, "calibrated", np.nan)),
                    expected_return=float(getattr(sig, "expected_return", np.nan))
                    if hasattr(sig, "expected_return")
                    else None,
                    volatility=float(getattr(row, "volatility_20", np.nan))
                    if hasattr(row, "volatility_20")
                    else None,
                    position=Position(row.symbol, holding.shares, holding.cost_basis),
                    n_open_positions=len(portfolio.open_symbols),
                )
            )

        if inputs:
            decisions = engine.decide_batch(inputs)
            for d in decisions:
                decision_rows.append(d.to_row())
                if d.signal is not Signal.HOLD or not np.isnan(d.target_weight):
                    pending.append(
                        Order(
                            symbol=d.symbol,
                            decision_date=session_date,
                            target_weight=d.target_weight,
                            signal=d.signal.value,
                        )
                    )

    trades = pd.DataFrame(
        [
            {
                "date": t.trade_date,
                "symbol": t.symbol,
                "side": t.side,
                "shares": t.shares,
                "price": t.price,
                "notional": t.notional,
                "commission": t.costs.commission,
                "spread": t.costs.spread,
                "slippage": t.costs.slippage,
                "total_cost": t.costs.total,
                "realised_pnl": t.realised_pnl,
            }
            for t in portfolio.trades
        ]
    )

    log.info(
        "Backtest complete: %d sessions | %d fills | %d round trips | "
        "max reconciliation error %.6f",
        len(curve),
        len(portfolio.trades),
        portfolio.round_trips(),
        max_error,
    )

    return BacktestResult(
        equity_curve=pd.DataFrame(curve),
        trades=trades,
        decisions=pd.DataFrame(decision_rows),
        config=config,
        max_reconciliation_error=max_error,
    )
