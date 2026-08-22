"""Backtester tests.

The accounting tests carry the most weight. Each one plants a specific bug that
``cash + holdings`` alone would not catch, and checks that the dual-path
reconciliation does.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from trademind.backtesting.benchmarks import buy_and_hold, equal_weight_rebalanced
from trademind.backtesting.config import BacktestConfig, CostConfig, compute_costs
from trademind.backtesting.engine import run_backtest
from trademind.backtesting.execution import (
    ExecutionError,
    Order,
    execute_orders,
    size_order,
)
from trademind.backtesting.portfolio import (
    Portfolio,
    ReconciliationError,
    RECONCILE_TOLERANCE,
)
from trademind.backtesting.report import (
    max_drawdown,
    performance_metrics,
    render_report,
    sharpe_ratio,
)
from trademind.decision import DecisionEngine, RiskEngine, Thresholds

CONFIG = BacktestConfig(initial_capital=1_000_000.0)
FREE = BacktestConfig(initial_capital=1_000_000.0, costs=CostConfig(0, 0, 0))
D1 = date(2023, 6, 15)
D2 = date(2023, 6, 16)


# =====================================================================
# Dual-path reconciliation
# =====================================================================

def test_fresh_portfolio_reconciles():
    p = Portfolio(1_000_000.0)
    assert p.reconcile({}, D1) == pytest.approx(0.0)


def test_buy_reconciles():
    p = Portfolio(1_000_000.0)
    costs = compute_costs(100 * 1000.0, CONFIG.costs)
    p.buy("A", 100, 1000.0, costs, D1)
    assert abs(p.reconcile({"A": 1000.0}, D1)) < RECONCILE_TOLERANCE


def test_price_move_reconciles():
    p = Portfolio(1_000_000.0)
    p.buy("A", 100, 1000.0, compute_costs(100_000.0, CONFIG.costs), D1)
    assert abs(p.reconcile({"A": 1200.0}, D2)) < RECONCILE_TOLERANCE


def test_round_trip_reconciles():
    p = Portfolio(1_000_000.0)
    p.buy("A", 100, 1000.0, compute_costs(100_000.0, CONFIG.costs), D1)
    p.sell("A", 100, 1100.0, compute_costs(110_000.0, CONFIG.costs), D2)
    assert abs(p.reconcile({"A": 1100.0}, D2)) < RECONCILE_TOLERANCE


def test_uncharged_commission_is_caught():
    """The bug 'cash + holdings' cannot see: a fill that forgot its fee.

    Cash is debited for the fee but the accumulator never records it, so the
    balance sheet and the flow path diverge by exactly the missing amount.
    """
    p = Portfolio(1_000_000.0)
    costs = compute_costs(100_000.0, CONFIG.costs)
    p.buy("A", 100, 1000.0, costs, D1)

    p.total_costs = type(costs)()      # wipe the accumulator

    with pytest.raises(ReconciliationError, match="mismatch"):
        p.reconcile({"A": 1000.0}, D1)


def test_unrecorded_dividend_is_caught():
    p = Portfolio(1_000_000.0)
    p.buy("A", 100, 1000.0, compute_costs(100_000.0, CONFIG.costs), D1)
    p.cash += 5000.0                   # cash credited, accumulator not

    with pytest.raises(ReconciliationError):
        p.reconcile({"A": 1000.0}, D1)


def test_phantom_shares_are_caught():
    p = Portfolio(1_000_000.0)
    p.buy("A", 100, 1000.0, compute_costs(100_000.0, CONFIG.costs), D1)
    p.holdings["A"].shares = 150.0     # shares appear from nowhere

    with pytest.raises(ReconciliationError):
        p.reconcile({"A": 1000.0}, D1)


# =====================================================================
# Corporate actions
# =====================================================================

def test_split_preserves_wealth_end_to_end():
    """The ₹100,000 invariant, through the portfolio rather than in isolation."""
    p = Portfolio(1_000_000.0)
    p.buy("A", 100, 1000.0, compute_costs(100_000.0, FREE.costs), D1)

    before = p.equity_balance_sheet({"A": 1000.0})
    p.apply_split("A", 2.0, D2)
    after = p.equity_balance_sheet({"A": 500.0})   # price halves with the split

    assert after == pytest.approx(before)
    assert p.holdings["A"].shares == 200.0
    assert p.holdings["A"].cost_basis == pytest.approx(500.0)


def test_split_then_sale_realises_the_right_pnl():
    """A split must not manufacture or destroy realised profit."""
    p = Portfolio(1_000_000.0)
    p.buy("A", 100, 1000.0, compute_costs(100_000.0, FREE.costs), D1)
    p.apply_split("A", 2.0, D2)
    p.sell("A", 200, 550.0, compute_costs(110_000.0, FREE.costs), D2)

    # 200 x (550 - 500) = 10,000, the same as 100 x (1100 - 1000).
    assert p.realised_pnl == pytest.approx(10_000.0)


def test_reverse_split_preserves_wealth():
    p = Portfolio(1_000_000.0)
    p.buy("A", 300, 100.0, compute_costs(30_000.0, FREE.costs), D1)
    before = p.equity_balance_sheet({"A": 100.0})
    p.apply_split("A", 1 / 3, D2)
    assert p.equity_balance_sheet({"A": 300.0}) == pytest.approx(before)


def test_split_on_a_flat_position_is_a_noop():
    p = Portfolio(1_000_000.0)
    p.apply_split("A", 2.0, D1)
    assert p.reconcile({}, D1) == pytest.approx(0.0)


def test_dividend_credits_cash_and_reconciles():
    p = Portfolio(1_000_000.0)
    p.buy("A", 100, 1000.0, compute_costs(100_000.0, FREE.costs), D1)
    amount = p.apply_dividend("A", 10.0, D2)

    assert amount == pytest.approx(1000.0)
    assert p.dividends_received == pytest.approx(1000.0)
    assert abs(p.reconcile({"A": 1000.0}, D2)) < RECONCILE_TOLERANCE


def test_dividend_on_a_flat_position_pays_nothing():
    p = Portfolio(1_000_000.0)
    assert p.apply_dividend("A", 10.0, D1) == 0.0


# =====================================================================
# Trade accounting
# =====================================================================

def test_cost_basis_is_the_weighted_average():
    p = Portfolio(1_000_000.0)
    p.buy("A", 100, 1000.0, compute_costs(100_000.0, FREE.costs), D1)
    p.buy("A", 100, 1200.0, compute_costs(120_000.0, FREE.costs), D2)
    assert p.holdings["A"].cost_basis == pytest.approx(1100.0)


def test_shorting_is_refused():
    p = Portfolio(1_000_000.0)
    with pytest.raises(ValueError, match="long-only"):
        p.sell("A", 100, 1000.0, compute_costs(100_000.0, FREE.costs), D1)


def test_overspending_is_refused():
    p = Portfolio(1000.0)
    with pytest.raises(ValueError, match="Insufficient cash"):
        p.buy("A", 100, 1000.0, compute_costs(100_000.0, FREE.costs), D1)


def test_round_trips_count_completions_not_fills():
    """A buy and its sell are one round trip, not two trades."""
    p = Portfolio(1_000_000.0)
    p.buy("A", 100, 1000.0, compute_costs(100_000.0, FREE.costs), D1)
    p.buy("B", 100, 1000.0, compute_costs(100_000.0, FREE.costs), D1)
    p.sell("A", 100, 1100.0, compute_costs(110_000.0, FREE.costs), D2)

    assert len(p.trades) == 3
    assert p.round_trips() == 1


def test_costs_are_itemised():
    costs = compute_costs(100_000.0, CostConfig(3, 5, 5))
    assert costs.commission == pytest.approx(30.0)
    assert costs.spread == pytest.approx(50.0)
    assert costs.slippage == pytest.approx(50.0)
    assert costs.total == pytest.approx(130.0)


# =====================================================================
# Execution timing
# =====================================================================

def test_same_session_execution_is_refused():
    """The t -> t+1 rule, enforced rather than assumed."""
    p = Portfolio(1_000_000.0)
    order = Order("A", decision_date=D1, target_weight=0.1, signal="BUY")

    with pytest.raises(ExecutionError, match="look-ahead"):
        execute_orders([order], p, D1, {"A": 1000.0}, CONFIG)


def test_backwards_execution_is_refused():
    p = Portfolio(1_000_000.0)
    order = Order("A", decision_date=D2, target_weight=0.1, signal="BUY")
    with pytest.raises(ExecutionError, match="look-ahead"):
        execute_orders([order], p, D1, {"A": 1000.0}, CONFIG)


def test_next_session_execution_is_allowed():
    p = Portfolio(1_000_000.0)
    order = Order("A", decision_date=D1, target_weight=0.1, signal="BUY")
    fills = execute_orders([order], p, D2, {"A": 1000.0}, CONFIG)
    assert len(fills) == 1


def test_nan_target_weight_leaves_the_position_alone():
    """HOLD must not liquidate."""
    p = Portfolio(1_000_000.0)
    p.buy("A", 100, 1000.0, compute_costs(100_000.0, FREE.costs), D1)

    order = Order("A", decision_date=D1, target_weight=float("nan"), signal="HOLD")
    fills = execute_orders([order], p, D2, {"A": 1000.0}, CONFIG)

    assert fills == []
    assert p.position("A").shares == 100.0


def test_sells_execute_before_buys():
    """Otherwise a fully invested portfolio could never rotate."""
    p = Portfolio(100_000.0)
    p.buy("A", 90, 1000.0, compute_costs(90_000.0, FREE.costs), D1)

    orders = [
        Order("B", decision_date=D1, target_weight=0.5, signal="BUY"),
        Order("A", decision_date=D1, target_weight=0.0, signal="SELL"),
    ]
    fills = execute_orders(orders, p, D2, {"A": 1000.0, "B": 1000.0}, CONFIG)

    assert [f.side for f in fills] == ["SELL", "BUY"]


def test_missing_execution_price_preserves_the_position():
    p = Portfolio(1_000_000.0)
    p.buy("A", 100, 1000.0, compute_costs(100_000.0, FREE.costs), D1)

    order = Order("A", decision_date=D1, target_weight=0.0, signal="SELL")
    fills = execute_orders([order], p, D2, {}, CONFIG)

    assert fills == []
    assert p.position("A").shares == 100.0


def test_order_sizing_floors_to_whole_shares():
    assert size_order(0.1, 1_000_000.0, 3333.0, 0.0, CONFIG) == 30.0


def test_order_sizing_accounts_for_the_existing_holding():
    assert size_order(0.1, 1_000_000.0, 1000.0, 40.0, CONFIG) == 60.0


def test_zero_price_is_refused():
    with pytest.raises(ExecutionError):
        size_order(0.1, 1_000_000.0, 0.0, 0.0, CONFIG)


# =====================================================================
# Engine
# =====================================================================

def make_panel(n=200, symbols=("A", "B"), seed=0, split_at=None):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n)
    rows = []
    for s in symbols:
        close = 1000.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n)))
        splits = np.ones(n)
        if split_at is not None:
            splits[split_at] = 2.0
            close[split_at:] /= 2.0
        rows.append(pd.DataFrame({
            "date": dates, "symbol": s,
            "open_raw": close * (1 + rng.normal(0, 0.002, n)),
            "close_raw": close,
            "volatility_20": np.full(n, 0.25),
            "split_ratio": splits,
            "dividend": np.zeros(n),
        }))
    return pd.concat(rows).sort_values(["date", "symbol"]).reset_index(drop=True)


def make_signals(panel, prob=0.60, expected=0.02):
    return pd.DataFrame({
        "date": panel["date"], "symbol": panel["symbol"],
        "calibrated": prob, "expected_return": expected,
    })


def engine(buy=0.55, sell=0.45, floor=0.0039):
    return DecisionEngine(
        Thresholds(buy=buy, sell=sell, min_expected_return=floor),
        RiskEngine(max_positions=10),
    )


def test_engine_runs_and_reconciles_every_session():
    panel = make_panel(n=150)
    result = run_backtest(panel, make_signals(panel), engine(), CONFIG)

    assert len(result.equity_curve) > 0
    assert result.max_reconciliation_error < RECONCILE_TOLERANCE


def test_engine_survives_a_split():
    panel = make_panel(n=150, split_at=80)
    result = run_backtest(panel, make_signals(panel), engine(), CONFIG)
    assert result.max_reconciliation_error < RECONCILE_TOLERANCE


def test_no_signal_means_no_trades():
    panel = make_panel(n=100)
    result = run_backtest(panel, pd.DataFrame(columns=["date", "symbol"]),
                          engine(), CONFIG)
    assert result.trades.empty
    assert result.final_equity == pytest.approx(CONFIG.initial_capital)


def test_below_cost_floor_produces_no_trades():
    """The Phase 6 gate, reaching the backtester."""
    panel = make_panel(n=120)
    signals = make_signals(panel, prob=0.90, expected=0.0001)
    result = run_backtest(panel, signals, engine(), CONFIG)
    assert result.trades.empty


def test_costs_reduce_the_return():
    panel = make_panel(n=150, seed=3)
    signals = make_signals(panel)

    charged = run_backtest(panel, signals, engine(), CONFIG)
    free = run_backtest(panel, signals, engine(), FREE)

    assert free.total_return > charged.total_return


def test_missing_panel_columns_are_rejected():
    with pytest.raises(ValueError, match="lacks required"):
        run_backtest(pd.DataFrame({"date": [], "symbol": []}),
                     pd.DataFrame(), engine(), CONFIG)


# =====================================================================
# Benchmarks
# =====================================================================

def test_buy_and_hold_pays_costs_too():
    """A cost-free benchmark against a cost-charged strategy is not a comparison."""
    panel = make_panel(n=120)
    bars = {s: g.reset_index(drop=True) for s, g in panel.groupby("symbol")}
    sessions = sorted(pd.Timestamp(d).date() for d in panel["date"].unique())

    for s, g in bars.items():
        g["date"] = pd.to_datetime(g["date"])

    charged = buy_and_hold(bars, sessions, CONFIG)
    free = buy_and_hold(bars, sessions, FREE)

    assert charged["total_costs"].iloc[-1] > 0
    assert free["total_costs"].iloc[-1] == 0


def test_rebalanced_benchmark_trades_more_than_buy_and_hold():
    panel = make_panel(n=200)
    bars = {s: g.reset_index(drop=True) for s, g in panel.groupby("symbol")}
    sessions = sorted(pd.Timestamp(d).date() for d in panel["date"].unique())

    hold = buy_and_hold(bars, sessions, CONFIG)
    rebal = equal_weight_rebalanced(bars, sessions, CONFIG, rebalance_every=21)

    assert rebal["total_costs"].iloc[-1] > hold["total_costs"].iloc[-1]


# =====================================================================
# Reporting
# =====================================================================

def test_drawdown_is_measured_from_the_running_peak():
    equity = pd.Series([100.0, 120.0, 90.0, 130.0])
    depth, _ = max_drawdown(equity)
    assert depth == pytest.approx(90 / 120 - 1)


def test_flat_equity_has_no_drawdown():
    assert max_drawdown(pd.Series([100.0] * 10))[0] == pytest.approx(0.0)


def test_sharpe_of_a_constant_series_is_zero():
    assert sharpe_ratio(pd.Series([0.0] * 100)) == 0.0


def test_short_samples_are_flagged_as_unreliable():
    """A one-year Sharpe has a standard error near 1.0. Say so."""
    curve = pd.DataFrame({
        "date": pd.bdate_range("2023-01-02", periods=200),
        "equity": np.linspace(1_000_000, 1_100_000, 200),
        "exposure": 0.5, "total_costs": 5000.0,
    })
    text = render_report(performance_metrics(curve))
    assert "too short to estimate Sharpe reliably" in text


def test_report_states_the_gross_return_needed_to_break_even():
    curve = pd.DataFrame({
        "date": pd.bdate_range("2023-01-02", periods=300),
        "equity": np.linspace(1_000_000, 1_010_000, 300),
        "exposure": 0.5, "total_costs": 26_000.0,
    })
    m = performance_metrics(curve, initial_capital=1_000_000.0)

    assert m["cost_drag"] == pytest.approx(0.026)
    assert m["gross_return_required"] == pytest.approx(m["total_return"] + 0.026)


def test_turnover_is_annualised():
    curve = pd.DataFrame({
        "date": pd.bdate_range("2023-01-02", periods=252),
        "equity": np.linspace(1_000_000, 1_000_000, 252),
        "exposure": 0.5, "total_costs": 1000.0,
    })
    trades = pd.DataFrame({
        "side": ["BUY", "SELL"], "notional": [500_000.0, 500_000.0],
        "total_cost": [500.0, 500.0], "realised_pnl": [0.0, 1000.0],
        "commission": [100.0, 100.0], "spread": [200.0, 200.0],
        "slippage": [200.0, 200.0],
    })
    m = performance_metrics(curve, trades, initial_capital=1_000_000.0)
    assert m["annual_turnover"] == pytest.approx(1.0, rel=0.05)


def test_empty_curve_reports_nothing():
    assert performance_metrics(pd.DataFrame()) == {}
