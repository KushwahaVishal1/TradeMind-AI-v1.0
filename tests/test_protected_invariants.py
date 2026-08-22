"""Protected invariants — the eight rules that block a merge.

Each invariant appears twice:

**The invariant test** asserts the property holds.

**The mutation test** deliberately breaks the property and asserts the guard
fires.

The pairing is the point. An invariant test that cannot fail proves nothing,
and a suite of always-green tests is worse than no suite, because it manufactures
confidence. Every guard here is verified to be load-bearing.

CI runs this file as a separate, required job. It is fast — no network, no
model fitting, no large data — because a slow gate gets bypassed, and a bypassed
gate is not a gate.

Marked with ``@pytest.mark.protected``.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.protected


# =====================================================================
# 1. Final-test immutability
# =====================================================================

def _panel(n_days=800, symbols=("A", "B"), seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    return pd.concat([
        pd.DataFrame({
            "date": dates, "symbol": s,
            "f1": rng.normal(size=n_days), "f2": rng.normal(size=n_days),
            "y": rng.normal(0, 0.015, n_days),
        }) for s in symbols
    ]).sort_values(["date", "symbol"]).reset_index(drop=True)


def test_invariant_1_final_test_is_never_reachable_for_fitting():
    from trademind.validation import split_development

    panel = _panel()
    dev = split_development(panel, date(2021, 1, 1))
    assert dev.panel["date"].max() < pd.Timestamp("2021-01-01")


def test_mutation_1_tampered_final_test_is_detected():
    """Break it: alter a locked row and confirm the fingerprint catches it."""
    from trademind.validation import FinalTestLock, FinalTestViolation

    panel = _panel()
    lock = FinalTestLock.create(panel, date(2021, 1, 1))

    tampered = panel.copy()
    tampered.loc[tampered["date"] >= pd.Timestamp("2021-01-01"), "f1"] *= 1.001

    with pytest.raises(FinalTestViolation, match="has changed"):
        lock.unlock(tampered, reason="evaluation")


def test_mutation_1b_locked_rows_reaching_a_fit_are_detected():
    from trademind.validation import FinalTestViolation, assert_no_locked_data

    with pytest.raises(FinalTestViolation):
        assert_no_locked_data(_panel(), date(2021, 1, 1), context="fit")


# =====================================================================
# 2. No future information in features
# =====================================================================

def _bars(n=500, seed=3):
    from trademind.ingestion.corporate_actions import apply_corporate_actions

    rng = np.random.default_rng(seed)
    close = 500.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n)))
    intraday = np.abs(rng.normal(0, 0.008, n))
    df = pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=n),
        "symbol": "T.NS",
        "open_split": close * (1 + rng.normal(0, 0.003, n)),
        "high_split": close * (1 + intraday),
        "low_split": close * (1 - intraday),
        "close_split": close,
        "volume": rng.integers(50_000, 500_000, n).astype("int64"),
        "dividend": np.zeros(n), "split_ratio": np.ones(n),
    })
    df["high_split"] = df[["open_split", "high_split", "close_split"]].max(axis=1)
    df["low_split"] = df[["open_split", "low_split", "close_split"]].min(axis=1)
    return apply_corporate_actions(df)


def test_invariant_2_features_do_not_change_when_the_future_changes():
    from trademind.features import build_symbol_features, feature_columns

    bars = _bars()
    cut = 350
    original = build_symbol_features(bars)

    mutated_bars = bars.copy()
    for col in ("open_split", "high_split", "low_split", "close_split",
                "adj_close", "open_raw", "high_raw", "low_raw", "close_raw"):
        mutated_bars.loc[cut + 1:, col] *= 0.5
    mutated = build_symbol_features(mutated_bars)

    offenders = [
        col for col in feature_columns(original)
        if not np.allclose(
            original[col].iloc[: cut + 1].to_numpy(dtype=float),
            mutated[col].iloc[: cut + 1].to_numpy(dtype=float),
            equal_nan=True, rtol=1e-9,
        )
    ]
    assert not offenders, f"look-ahead in: {offenders}"


def test_mutation_2_a_leaky_feature_is_caught():
    """Break it: plant the full-sample rank mistake and confirm detection."""
    from trademind.features import build_symbol_features

    bars = _bars(n=400)
    cut = 300

    def leaky(df):
        out = build_symbol_features(df)
        out["LEAKY"] = out["volatility_20"].rank(pct=True)   # sees the future
        return out

    original = leaky(bars)
    mutated_bars = bars.copy()
    mutated_bars.loc[cut + 1:, ["adj_close", "close_split"]] *= 3.0
    mutated = leaky(mutated_bars)

    assert not np.allclose(
        original["LEAKY"].iloc[: cut + 1], mutated["LEAKY"].iloc[: cut + 1],
        equal_nan=True,
    ), "the leakage detector failed on a known-leaky feature"


# =====================================================================
# 3. No same-day execution
# =====================================================================

def test_invariant_3_storage_rejects_same_day_execution():
    from trademind.storage import Prediction

    with pytest.raises(ValueError, match="look-ahead"):
        Prediction(
            symbol="A", prediction_date="2023-06-15",
            execution_date="2023-06-15",
            model_version="m", feature_version="f", decision_version="d",
            threshold_version="t", signal="BUY",
        )


def test_invariant_3b_backtester_rejects_same_day_execution():
    from trademind.backtesting import ExecutionError, Order, Portfolio, execute_orders
    from trademind.backtesting.config import BacktestConfig

    order = Order("A", decision_date=date(2023, 6, 15), target_weight=0.1,
                  signal="BUY")
    with pytest.raises(ExecutionError, match="look-ahead"):
        execute_orders([order], Portfolio(1_000_000.0), date(2023, 6, 15),
                       {"A": 1000.0}, BacktestConfig())


def test_mutation_3_a_model_predicting_its_own_training_window_is_refused():
    """Break it: the backfill contamination case."""
    from trademind.orchestration import TemporalValidityError, assert_temporally_valid

    with pytest.raises(TemporalValidityError, match="cannot predict"):
        assert_temporally_valid(date(2023, 1, 15), date(2024, 6, 30), "prod-1")


# =====================================================================
# 4. Corporate-action split continuity
# =====================================================================

def test_invariant_4_a_split_preserves_wealth():
    """100 x Rs.1,000 = 200 x Rs.500."""
    from trademind.ingestion import adjust_position_for_split

    shares, basis = adjust_position_for_split(100.0, 1000.0, 2.0)
    assert shares == 200.0 and basis == 500.0
    assert shares * basis == pytest.approx(100_000.0)


def test_invariant_4b_a_split_preserves_portfolio_equity():
    from trademind.backtesting import Portfolio, compute_costs
    from trademind.backtesting.config import CostConfig

    free = CostConfig(0, 0, 0)
    portfolio = Portfolio(1_000_000.0)
    portfolio.buy("A", 100, 1000.0, compute_costs(100_000.0, free), date(2023, 6, 15))

    before = portfolio.equity_balance_sheet({"A": 1000.0})
    portfolio.apply_split("A", 2.0, date(2023, 6, 16))
    after = portfolio.equity_balance_sheet({"A": 500.0})

    assert after == pytest.approx(before)


def test_mutation_4_a_wrong_split_transform_is_detected(monkeypatch):
    """Break it: replace the split maths with a version that destroys wealth.

    The guard has to be tested by breaking the *transform*, not the state.
    Corrupting the cost basis alone does not work: `apply_split` compares
    shares x basis before and after, and the transform preserves that product
    for any starting values. So the meaningful mutation is a bad transform.
    """
    from trademind.backtesting import Portfolio, compute_costs
    from trademind.backtesting.config import CostConfig
    from trademind.backtesting.portfolio import ReconciliationError
    import trademind.backtesting.portfolio as portfolio_module

    portfolio = Portfolio(1_000_000.0)
    portfolio.buy("A", 100, 1000.0,
                  compute_costs(100_000.0, CostConfig(0, 0, 0)), date(2023, 6, 15))

    # A plausible bug: share count multiplied, cost basis left alone.
    monkeypatch.setattr(
        portfolio_module, "adjust_position_for_split",
        lambda shares, basis, ratio: (shares * ratio, basis),
    )

    with pytest.raises(ReconciliationError, match="create or destroy wealth"):
        portfolio.apply_split("A", 2.0, date(2023, 6, 16))


def test_invariant_4c_reconciliation_cannot_detect_a_missing_split():
    """An honest limitation, asserted so it stays documented.

    If a split happens and the provider never reports it, both equity paths use
    the same (unadjusted) share count and therefore still agree. Reconciliation
    is blind to it by construction.

    The detection for that case lives upstream, in the Phase 1 validator:
    `ABNORMAL_JUMP` flags a large single-session move with no recorded
    corporate action, which is exactly the signature of a missed split.
    """
    from trademind.backtesting import Portfolio, compute_costs
    from trademind.backtesting.config import CostConfig
    from trademind.backtesting.portfolio import RECONCILE_TOLERANCE

    portfolio = Portfolio(1_000_000.0)
    portfolio.buy("A", 100, 1000.0,
                  compute_costs(100_000.0, CostConfig(0, 0, 0)), date(2023, 6, 15))

    # Price halves as if a 2:1 split occurred, but no split was applied.
    assert abs(portfolio.reconcile({"A": 500.0}, date(2023, 6, 16))) \
        < RECONCILE_TOLERANCE

    # The upstream guard is what catches it.
    from trademind.ingestion.data_validator import JUMP_WARN_THRESHOLD
    assert 0.5 > JUMP_WARN_THRESHOLD


# =====================================================================
# 5. Portfolio conservation and P&L reconciliation
# =====================================================================

def _funded_portfolio():
    from trademind.backtesting import Portfolio, compute_costs
    from trademind.backtesting.config import CostConfig

    portfolio = Portfolio(1_000_000.0)
    portfolio.buy("A", 100, 1000.0,
                  compute_costs(100_000.0, CostConfig(3, 5, 5)), date(2023, 6, 15))
    return portfolio


def test_invariant_5_equity_paths_agree():
    from trademind.backtesting.portfolio import RECONCILE_TOLERANCE

    portfolio = _funded_portfolio()
    assert abs(portfolio.reconcile({"A": 1200.0}, date(2023, 6, 16))) \
        < RECONCILE_TOLERANCE


def test_mutation_5a_an_uncharged_fee_is_detected():
    """Break it: the bug `cash + holdings` cannot see."""
    from trademind.backtesting.config import CostBreakdown
    from trademind.backtesting.portfolio import ReconciliationError

    portfolio = _funded_portfolio()
    portfolio.total_costs = CostBreakdown()          # wipe the accumulator

    with pytest.raises(ReconciliationError):
        portfolio.reconcile({"A": 1000.0}, date(2023, 6, 15))


def test_mutation_5b_phantom_shares_are_detected():
    from trademind.backtesting.portfolio import ReconciliationError

    portfolio = _funded_portfolio()
    portfolio.holdings["A"].shares = 150.0

    with pytest.raises(ReconciliationError):
        portfolio.reconcile({"A": 1000.0}, date(2023, 6, 15))


def test_mutation_5c_an_unrecorded_dividend_is_detected():
    from trademind.backtesting.portfolio import ReconciliationError

    portfolio = _funded_portfolio()
    portfolio.cash += 5000.0                          # credited, not recorded

    with pytest.raises(ReconciliationError):
        portfolio.reconcile({"A": 1000.0}, date(2023, 6, 15))


# =====================================================================
# 6. HOLD carries the position
# =====================================================================

def _decision_inputs():
    from trademind.decision import DecisionInput, Position

    long_position = Position("A", shares=100.0, cost_basis=1000.0)
    return (
        DecisionInput("A", date(2023, 6, 15), calibrated_probability=0.50,
                      expected_return=0.01, position=long_position),
        DecisionInput("A", date(2023, 6, 15), calibrated_probability=0.50,
                      expected_return=0.01, position=Position("A")),
    )


def test_invariant_6_hold_on_a_long_returns_nan_not_zero():
    """Zero would liquidate every position on every quiet day."""
    from trademind.decision import Signal, Thresholds, generate_signal

    held, _ = _decision_inputs()
    decision = generate_signal(
        held, Thresholds(buy=0.55, sell=0.45, min_expected_return=0.0039)
    )
    assert decision.signal is Signal.HOLD
    assert np.isnan(decision.target_weight)


def test_invariant_6b_hold_never_produces_a_fill():
    from trademind.backtesting import Order, Portfolio, compute_costs, execute_orders
    from trademind.backtesting.config import BacktestConfig, CostConfig

    portfolio = Portfolio(1_000_000.0)
    portfolio.buy("A", 100, 1000.0,
                  compute_costs(100_000.0, CostConfig(0, 0, 0)), date(2023, 6, 15))

    order = Order("A", decision_date=date(2023, 6, 15),
                  target_weight=float("nan"), signal="HOLD")
    fills = execute_orders([order], portfolio, date(2023, 6, 16),
                           {"A": 1000.0}, BacktestConfig())

    assert fills == []
    assert portfolio.position("A").shares == 100.0


def test_invariant_6c_errors_preserve_rather_than_liquidate():
    """An outage must not become a realised loss plus a round trip."""
    from trademind.decision import (
        DecisionEngine, DecisionInput, Position, RejectReason, RiskEngine,
        Signal, Thresholds,
    )

    engine = DecisionEngine(
        Thresholds(buy=0.55, sell=0.45, min_expected_return=0.0039),
        RiskEngine(max_data_age_sessions=3),
    )
    decision = engine.decide(DecisionInput(
        "A", date(2023, 6, 15), calibrated_probability=0.30,
        expected_return=-0.05, data_age_sessions=99,
        position=Position("A", shares=100.0, cost_basis=1000.0),
    ))

    assert decision.rejected
    assert decision.reject_reason is RejectReason.STALE_DATA
    assert decision.signal is Signal.HOLD
    assert np.isnan(decision.target_weight)


# =====================================================================
# 7. SELL exits a long, and never opens a short
# =====================================================================

def test_invariant_7_sell_exits_a_long():
    from trademind.decision import Signal, Thresholds, generate_signal

    held, _ = _decision_inputs()
    decision = generate_signal(
        type(held)(**{**held.__dict__, "calibrated_probability": 0.20}),
        Thresholds(buy=0.55, sell=0.45, min_expected_return=0.0039),
    )
    assert decision.signal is Signal.SELL
    assert decision.target_weight == 0.0


def test_invariant_7b_sell_while_flat_stays_flat():
    from trademind.decision import Signal, Thresholds, generate_signal

    _, flat = _decision_inputs()
    decision = generate_signal(
        type(flat)(**{**flat.__dict__, "calibrated_probability": 0.20}),
        Thresholds(buy=0.55, sell=0.45, min_expected_return=0.0039),
    )
    assert decision.signal is Signal.SELL
    assert decision.target_weight == 0.0


def test_mutation_7_shorting_is_structurally_impossible():
    """Break it: try to sell shares that are not held."""
    from trademind.backtesting import Portfolio, compute_costs
    from trademind.backtesting.config import CostConfig

    with pytest.raises(ValueError, match="long-only"):
        Portfolio(1_000_000.0).sell(
            "A", 100, 1000.0, compute_costs(100_000.0, CostConfig(0, 0, 0)),
            date(2023, 6, 15),
        )


# =====================================================================
# 8. A failed candidate cannot be promoted
# =====================================================================

def _failed_outcome():
    from trademind.monitoring import ValidationOutcome

    return ValidationOutcome(
        passed=False, candidate_version="cand-1", incumbent_version="prod-1",
        checks={"auc_not_worse": False},
    )


def test_invariant_8_a_failed_candidate_keeps_the_incumbent():
    from trademind.monitoring import HealthState, decide_promotion

    decision = decide_promotion(_failed_outcome())
    assert not decision.promote
    assert decision.final_state is HealthState.KEEP_CURRENT_MODEL


def test_mutation_8a_forcing_promotion_raises():
    """Break it: construct the forbidden state directly. There is no override."""
    from trademind.monitoring import PromotionBlocked, PromotionDecision

    with pytest.raises(PromotionBlocked, match="no override"):
        PromotionDecision(promote=True, outcome=_failed_outcome())


def test_mutation_8b_the_registry_blocks_it_independently(tmp_path):
    """The second, independent guard on the same rule."""
    from trademind.experiments import IllegalTransition, ModelLifecycle, Stage
    from trademind.storage import init_db

    conn = init_db(tmp_path / "t.db")
    lifecycle = ModelLifecycle(conn)
    lifecycle.register("cand-1", "hgb")
    lifecycle.transition("cand-1", Stage.VALIDATING)
    lifecycle.transition("cand-1", Stage.FAILED, "did not pass")

    with pytest.raises(IllegalTransition, match="terminal"):
        lifecycle.transition("cand-1", Stage.PRODUCTION)
    conn.close()


# =====================================================================
# Meta: the suite must be complete
# =====================================================================

def test_all_eight_invariants_are_covered():
    """Guards against an invariant being quietly dropped from the suite."""
    from pathlib import Path

    source = Path(__file__).read_text(encoding="utf-8")
    for n in range(1, 9):
        assert f"test_invariant_{n}_" in source, f"invariant {n} has no test"


def test_every_invariant_has_a_mutation_check():
    """Invariants 1-5, 7, 8 pair with a mutation test.

    Invariant 6 (HOLD carries) is verified by three positive assertions across
    the decision and execution layers instead: 'HOLD did nothing' has no
    meaningful mutation — breaking it means returning 0.0, which the NaN
    assertion already catches directly.
    """
    from pathlib import Path

    source = Path(__file__).read_text(encoding="utf-8")
    for n in (1, 2, 3, 4, 5, 7, 8):
        assert f"test_mutation_{n}" in source, f"invariant {n} has no mutation check"
