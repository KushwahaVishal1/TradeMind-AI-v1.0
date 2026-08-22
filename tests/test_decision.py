"""Decision-engine tests.

The locked V1 semantics are protected here one rule at a time, because each is
a behaviour someone could plausibly "improve" later without realising it was
deliberate — particularly "preserve the position on error", which looks like a
missing liquidation until you think about what an outage would cost.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from trademind.decision import (
    CostModel,
    Decision,
    DecisionEngine,
    DecisionInput,
    Position,
    RejectReason,
    RiskBucket,
    RiskEngine,
    Signal,
    Thresholds,
    classify_risk,
    cost_feasibility,
    evaluate_decisions,
    generate_signal,
    minimum_expected_return,
    optimise_thresholds_out_of_fold,
    size_position,
)

TODAY = date(2023, 6, 15)
COSTS = CostModel()
TH = Thresholds(buy=0.55, sell=0.45, min_expected_return=0.0039)


def inp(**kw) -> DecisionInput:
    base = dict(
        symbol="RELIANCE.NS",
        decision_date=TODAY,
        calibrated_probability=0.60,
        expected_return=0.01,
        volatility=0.25,
        position=None,
    )
    base.update(kw)
    return DecisionInput(**base)


LONG = Position("RELIANCE.NS", shares=100.0, cost_basis=1000.0)
FLAT = Position("RELIANCE.NS")


# =====================================================================
# Cost feasibility — the finding that shapes the phase
# =====================================================================


def test_round_trip_cost_is_double_the_per_side():
    assert COSTS.per_side == pytest.approx(0.0013)
    assert COSTS.round_trip == pytest.approx(0.0026)


def test_one_day_holding_is_not_feasible_at_realistic_ic():
    """The structural finding: daily trading cannot clear 26 bps."""
    report = cost_feasibility(COSTS, holding_days=1, observed_ic=0.03)

    assert not report.feasible
    assert report.breakeven_ic > 0.09
    assert "NOT FEASIBLE" in report.render()


def test_longer_holding_amortises_the_cost():
    """The main lever available: spread one round trip over more days of edge."""
    short = cost_feasibility(COSTS, holding_days=1, observed_ic=0.05)
    long_ = cost_feasibility(COSTS, holding_days=20, observed_ic=0.05)

    assert long_.breakeven_ic < short.breakeven_ic
    assert long_.feasible and not short.feasible


def test_zero_cost_makes_anything_feasible():
    free = CostModel(0.0, 0.0, 0.0)
    assert cost_feasibility(free, observed_ic=0.01).feasible


def test_minimum_expected_return_is_derived_not_chosen():
    """Derived from the cost model with a margin, never tuned."""
    floor = minimum_expected_return(COSTS, margin=1.5)
    assert floor == pytest.approx(0.0026 * 1.5)


def test_the_roadmap_floor_was_below_the_cost_floor():
    """Documents why config.yaml left min_expected_return null.

    The roadmap specified 0.0005 — 5 bps against a 26 bps round trip. Every
    trade clearing that gate loses 21 bps on costs alone.
    """
    roadmap_value = 0.0005
    assert roadmap_value < COSTS.round_trip
    assert minimum_expected_return(COSTS) > roadmap_value * 7


def test_margin_below_one_is_rejected():
    with pytest.raises(ValueError, match="lose in expectation"):
        minimum_expected_return(COSTS, margin=0.9)


# =====================================================================
# Locked V1 semantics
# =====================================================================


def test_buy_requires_both_gates():
    d = generate_signal(inp(calibrated_probability=0.70, expected_return=0.01), TH)
    assert d.signal is Signal.BUY and not d.rejected


def test_confident_but_unprofitable_is_rejected():
    """The gate that matters: 0.72 confidence on a 3 bps move is a losing trade."""
    d = generate_signal(inp(calibrated_probability=0.72, expected_return=0.0003), TH)
    assert d.signal is Signal.HOLD
    assert d.reject_reason is RejectReason.BELOW_COST_FLOOR
    assert "cost floor" in d.rationale


def test_sell_exits_a_long():
    d = generate_signal(inp(calibrated_probability=0.30, position=LONG), TH)
    assert d.signal is Signal.SELL
    assert d.target_weight == 0.0


def test_sell_while_flat_stays_flat():
    """Locked: V1 is long-only and never opens a short."""
    d = generate_signal(inp(calibrated_probability=0.30, position=FLAT), TH)
    assert d.signal is Signal.SELL
    assert d.target_weight == 0.0
    assert "long-only" in d.rationale


def test_hold_band_carries_an_existing_position():
    """NaN target weight means 'leave it alone', not 'target zero'."""
    d = generate_signal(inp(calibrated_probability=0.50, position=LONG), TH)
    assert d.signal is Signal.HOLD
    assert np.isnan(d.target_weight)


def test_hold_while_flat_stays_flat():
    d = generate_signal(inp(calibrated_probability=0.50, position=FLAT), TH)
    assert d.signal is Signal.HOLD
    assert d.target_weight == 0.0


def test_missing_probability_preserves_the_position():
    d = generate_signal(inp(calibrated_probability=None, position=LONG), TH)
    assert d.rejected
    assert d.signal is Signal.HOLD
    assert np.isnan(d.target_weight)


def test_hold_is_never_actionable():
    d = generate_signal(inp(calibrated_probability=0.50), TH)
    assert not d.is_actionable


def test_thresholds_must_not_overlap():
    with pytest.raises(ValueError, match="HOLD band"):
        Thresholds(buy=0.40, sell=0.60, min_expected_return=0.001)


# =====================================================================
# Risk gates — errors preserve, never liquidate
# =====================================================================


def test_stale_data_preserves_the_position():
    """An outage must not become a realised loss plus a round-trip cost."""
    engine = DecisionEngine(TH, RiskEngine(max_data_age_sessions=3))
    d = engine.decide(inp(data_age_sessions=10, position=LONG))

    assert d.rejected
    assert d.reject_reason is RejectReason.STALE_DATA
    assert d.signal is Signal.HOLD
    assert np.isnan(d.target_weight)


def test_fresh_data_passes_the_gate():
    engine = DecisionEngine(TH, RiskEngine(max_data_age_sessions=3))
    assert not engine.decide(inp(data_age_sessions=1)).rejected


def test_incomplete_features_preserve_the_position():
    engine = DecisionEngine(TH)
    d = engine.decide(inp(features_complete=False, position=LONG))
    assert d.reject_reason is RejectReason.MISSING_FEATURES


def test_non_finite_probability_is_rejected():
    engine = DecisionEngine(TH)
    d = engine.decide(inp(calibrated_probability=float("nan")))
    assert d.reject_reason is RejectReason.MISSING_PROBABILITY


def test_position_cap_blocks_new_entries():
    engine = DecisionEngine(TH, RiskEngine(max_positions=5))
    d = engine.decide(inp(n_open_positions=5, position=FLAT))
    assert d.reject_reason is RejectReason.POSITION_LIMIT


def test_position_cap_never_force_closes():
    """A limit breach must not become an unplanned liquidation."""
    engine = DecisionEngine(TH, RiskEngine(max_positions=5))
    d = engine.decide(inp(n_open_positions=99, position=LONG, calibrated_probability=0.70))
    assert not d.rejected
    assert d.signal is Signal.BUY


def test_risk_gates_run_before_the_signal():
    """A rejected input must never produce a tradeable number."""
    engine = DecisionEngine(TH, RiskEngine(max_data_age_sessions=1))
    d = engine.decide(
        inp(data_age_sessions=99, calibrated_probability=0.99, expected_return=0.5)
    )
    assert d.signal is Signal.HOLD and d.rejected


# =====================================================================
# Risk classification
# =====================================================================


def test_volatility_buckets():
    assert classify_risk(0.10) is RiskBucket.LOW
    assert classify_risk(0.25) is RiskBucket.MEDIUM
    assert classify_risk(0.45) is RiskBucket.HIGH
    assert classify_risk(0.90) is RiskBucket.EXTREME


def test_unknown_volatility_defaults_to_medium():
    assert classify_risk(None) is RiskBucket.MEDIUM
    assert classify_risk(float("nan")) is RiskBucket.MEDIUM


def test_risk_bucket_is_reported_even_on_rejection():
    engine = DecisionEngine(TH)
    d = engine.decide(inp(data_age_sessions=99, volatility=0.80))
    assert d.risk_bucket is RiskBucket.EXTREME


# =====================================================================
# Sizing
# =====================================================================


def buy_decision(p=0.70):
    return Decision(
        symbol="X",
        decision_date=TODAY,
        signal=Signal.BUY,
        calibrated_probability=p,
        expected_return=0.01,
    )


def test_fixed_sizing_uses_the_cap():
    assert size_position(buy_decision(), inp(), "fixed", max_weight=0.10) == 0.10


def test_volatility_targeting_shrinks_volatile_names():
    calm = size_position(
        buy_decision(), inp(volatility=0.15), "volatility_target", max_weight=0.10
    )
    wild = size_position(
        buy_decision(), inp(volatility=0.80), "volatility_target", max_weight=0.10
    )
    assert wild < calm


def test_volatility_targeting_respects_the_cap():
    """A very calm name would otherwise be sized far above the limit."""
    w = size_position(
        buy_decision(), inp(volatility=0.01), "volatility_target", max_weight=0.10
    )
    assert w == pytest.approx(0.10)


def test_volatility_targeting_equalises_risk_contribution():
    """weight x volatility should be roughly constant below the cap."""
    a = size_position(
        buy_decision(),
        inp(volatility=0.40),
        "volatility_target",
        max_weight=0.10,
        reference_volatility=0.25,
    )
    b = size_position(
        buy_decision(),
        inp(volatility=0.80),
        "volatility_target",
        max_weight=0.10,
        reference_volatility=0.25,
    )
    assert a * 0.40 == pytest.approx(b * 0.80)


def test_missing_volatility_falls_back_rather_than_guessing():
    w = size_position(
        buy_decision(), inp(volatility=None), "volatility_target", max_weight=0.10
    )
    assert 0 < w < 0.10


def test_confidence_sizing_scales_with_probability():
    low = size_position(
        buy_decision(0.56), inp(), "confidence", max_weight=0.10, buy_threshold=0.55
    )
    high = size_position(
        buy_decision(0.95), inp(), "confidence", max_weight=0.10, buy_threshold=0.55
    )
    assert high > low


def test_sell_sizes_to_zero():
    d = Decision(symbol="X", decision_date=TODAY, signal=Signal.SELL)
    assert size_position(d, inp(position=LONG)) == 0.0


def test_hold_on_a_long_returns_nan_not_zero():
    """Returning 0.0 here would liquidate every position on every quiet day."""
    d = Decision(symbol="X", decision_date=TODAY, signal=Signal.HOLD)
    assert np.isnan(size_position(d, inp(position=LONG)))


def test_unknown_sizing_method_raises():
    with pytest.raises(ValueError, match="Unknown sizing"):
        size_position(buy_decision(), inp(), "kelly")


# =====================================================================
# Batch decisions
# =====================================================================


def test_batch_prioritises_the_strongest_signals():
    """When the cap binds, it must bind on the weakest candidates."""
    engine = DecisionEngine(TH, RiskEngine(max_positions=2))
    inputs = [
        DecisionInput(
            f"S{i}",
            TODAY,
            calibrated_probability=p,
            expected_return=0.01,
            position=Position(f"S{i}"),
        )
        for i, p in enumerate([0.58, 0.90, 0.62, 0.75])
    ]
    decisions = {d.symbol: d for d in engine.decide_batch(inputs)}

    assert decisions["S1"].signal is Signal.BUY  # p=0.90
    assert decisions["S3"].signal is Signal.BUY  # p=0.75
    assert decisions["S0"].reject_reason is RejectReason.POSITION_LIMIT


def test_batch_returns_one_decision_per_input():
    engine = DecisionEngine(TH)
    inputs = [
        DecisionInput(f"S{i}", TODAY, calibrated_probability=0.6, expected_return=0.01)
        for i in range(5)
    ]
    assert len(engine.decide_batch(inputs)) == 5


def test_batch_frame_has_the_expected_columns():
    engine = DecisionEngine(TH)
    frame = engine.to_frame(engine.decide_batch([inp()]))
    for col in ("symbol", "signal", "target_weight", "rejected", "rationale"):
        assert col in frame.columns


# =====================================================================
# Threshold derivation
# =====================================================================


def signals_frame(n_days=1200, skill=0.0, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    rows = []
    for s in ("A", "B", "C"):
        latent = rng.normal(size=n_days)
        ret = latent * skill * 0.015 + rng.normal(0, 0.015, n_days)
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": s,
                    "calibrated": 1 / (1 + np.exp(-latent * skill)),
                    "y_true": ret,
                    "expected_return": latent * skill * 0.015,
                }
            )
        )
    return pd.concat(rows).sort_values("date").reset_index(drop=True)


def test_thresholds_are_derived_out_of_fold():
    th, stability = optimise_thresholds_out_of_fold(
        signals_frame(skill=0.5), COSTS, expected_return_col="expected_return"
    )
    assert 0.0 <= th.sell <= th.buy <= 1.0
    assert not stability.empty
    assert {"in_block_net", "out_of_block_net"} <= set(stability.columns)


def test_derived_floor_matches_the_cost_model():
    th, _ = optimise_thresholds_out_of_fold(signals_frame(skill=0.5), COSTS)
    assert th.min_expected_return == pytest.approx(minimum_expected_return(COSTS))


def test_threshold_search_needs_enough_data():
    with pytest.raises(ValueError, match="at least"):
        optimise_thresholds_out_of_fold(signals_frame(n_days=100), COSTS)


def test_out_of_block_performance_is_reported_separately():
    """The honest estimate is the out-of-block column, not the in-block one."""
    _, stability = optimise_thresholds_out_of_fold(signals_frame(skill=0.3), COSTS)
    assert (stability["in_block_net"] >= stability["out_of_block_net"]).any()


# =====================================================================
# Evaluation
# =====================================================================


def test_evaluation_reports_net_of_costs():
    decisions = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "decision_date": [TODAY] * 3,
            "signal": ["BUY", "BUY", "HOLD"],
        }
    )
    outcomes = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "date": [TODAY] * 3,
            "y_true": [0.02, -0.01, 0.03],
        }
    )
    m = evaluate_decisions(decisions, outcomes, round_trip_cost=0.0026)

    assert m["n_buys"] == 2
    assert m["gross_return_per_trade"] == pytest.approx(0.005)
    assert m["net_return_per_trade"] == pytest.approx(0.005 - 0.0026)


def test_cost_drag_above_one_is_flagged():
    from trademind.decision import render_evaluation

    decisions = pd.DataFrame(
        {
            "symbol": ["A"],
            "decision_date": [TODAY],
            "signal": ["BUY"],
        }
    )
    outcomes = pd.DataFrame(
        {
            "symbol": ["A"],
            "date": [TODAY],
            "y_true": [0.0005],
        }
    )
    text = render_evaluation(evaluate_decisions(decisions, outcomes, round_trip_cost=0.0026))
    assert "costs exceed gross edge" in text


def test_empty_decisions_evaluate_to_nothing():
    assert (
        evaluate_decisions(
            pd.DataFrame(columns=["symbol", "decision_date", "signal"]),
            pd.DataFrame(columns=["symbol", "date", "y_true"]),
            0.0026,
        )
        == {}
    )
