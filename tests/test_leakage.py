"""Leakage tests.

**The single most important test file in this project.**

The method is the one the roadmap specifies: take a price history, compute
features, then mutate every price after some cut point and recompute. Any
feature value at or before the cut that changes has seen the future.

This runs as a property test across *every* feature column rather than as a
spot-check on a few, because leakage does not announce itself. The
``rank(pct=True)`` mistake in ``regime.py`` produces a perfectly reasonable
number, no future column appears anywhere, and only a mutation test finds it.

If a new feature is added and this file starts failing, the feature is wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trademind.features import regime
from trademind.features.labels import (
    DIRECTION_LABEL,
    LABEL_COLUMNS,
    RESEARCH_LABEL,
    TRADEABLE_LABEL,
    add_labels,
)
from trademind.features.pipeline import (
    build_panel,
    build_symbol_features,
    drop_unlabelled,
    feature_columns,
    trim_warmup,
)
from trademind.ingestion.corporate_actions import apply_corporate_actions

RNG = np.random.default_rng(42)


def synthetic_bars(n=800, symbol="TEST.NS", seed=42, splits=None, divs=None):
    """A random-walk price history with realistic OHLCV structure."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.015, n)
    close = 500.0 * np.exp(np.cumsum(rets))

    intraday = np.abs(rng.normal(0, 0.008, n))
    df = pd.DataFrame({
        "date": pd.bdate_range("2019-01-01", periods=n),
        "symbol": symbol,
        "open_split": close * (1 + rng.normal(0, 0.003, n)),
        "high_split": close * (1 + intraday),
        "low_split": close * (1 - intraday),
        "close_split": close,
        "volume": rng.integers(50_000, 500_000, n).astype("int64"),
        "dividend": np.asarray(divs if divs is not None else [0.0] * n, dtype=float),
        "split_ratio": np.asarray(splits if splits is not None else [1.0] * n,
                                  dtype=float),
    })
    # Keep the bar internally consistent.
    df["high_split"] = df[["open_split", "high_split", "close_split"]].max(axis=1)
    df["low_split"] = df[["open_split", "low_split", "close_split"]].min(axis=1)
    return apply_corporate_actions(df)


# =======================================================================
# THE leakage test
# =======================================================================

def test_no_feature_sees_the_future():
    """Mutate all prices after the cut; every feature at or before it must hold.

    The core guarantee of the entire system. If this fails, no downstream
    metric means anything.
    """
    bars = synthetic_bars(n=800)
    cut = 600

    original = build_symbol_features(bars)

    # Violently perturb the future: a 50% drop plus fresh noise.
    mutated_bars = bars.copy()
    tail = slice(cut + 1, None)
    for col in ("open_split", "high_split", "low_split", "close_split", "adj_close",
                "open_raw", "high_raw", "low_raw", "close_raw"):
        n_tail = len(mutated_bars.loc[tail, col])
        mutated_bars.loc[tail, col] = (
            mutated_bars.loc[tail, col].to_numpy() * 0.5
            * (1 + RNG.normal(0, 0.1, n_tail))
        )
    mutated_bars.loc[tail, "volume"] = (
        mutated_bars.loc[tail, "volume"].to_numpy() * 7
    )

    mutated = build_symbol_features(mutated_bars)

    offenders = []
    for col in feature_columns(original):
        a = original[col].iloc[: cut + 1].to_numpy(dtype=float)
        b = mutated[col].iloc[: cut + 1].to_numpy(dtype=float)
        if not np.allclose(a, b, equal_nan=True, rtol=1e-9, atol=1e-12):
            n_changed = int((~np.isclose(a, b, equal_nan=True)).sum())
            offenders.append(f"{col} ({n_changed} rows changed)")

    assert not offenders, (
        "LOOK-AHEAD BIAS — these features changed when only future data moved:\n  "
        + "\n  ".join(offenders)
    )


def test_leakage_holds_at_several_cut_points():
    """One cut point could pass by luck. Several cannot."""
    bars = synthetic_bars(n=700, seed=7)
    original = build_symbol_features(bars)
    cols = feature_columns(original)

    for cut in (300, 450, 600):
        mutated_bars = bars.copy()
        tail = slice(cut + 1, None)
        for col in ("open_split", "high_split", "low_split", "close_split",
                    "adj_close", "adj_open"):
            if col in mutated_bars:
                mutated_bars.loc[tail, col] *= 2.5

        mutated = build_symbol_features(mutated_bars)

        for col in cols:
            a = original[col].iloc[: cut + 1].to_numpy(dtype=float)
            b = mutated[col].iloc[: cut + 1].to_numpy(dtype=float)
            assert np.allclose(a, b, equal_nan=True, rtol=1e-9), (
                f"{col} leaked at cut={cut}"
            )


def test_the_test_can_actually_fail():
    """A leaky feature must be caught — otherwise the test above proves nothing.

    Deliberately plants the full-sample ``rank(pct=True)`` mistake and confirms
    the mutation check flags it.
    """
    bars = synthetic_bars(n=400)
    cut = 300

    def leaky(df):
        out = build_symbol_features(df)
        # The classic error: ranked against the entire series, future included.
        out["LEAKY_vol_rank"] = out["volatility_20"].rank(pct=True)
        return out

    original = leaky(bars)
    mutated_bars = bars.copy()
    mutated_bars.loc[cut + 1:, "adj_close"] *= 3.0
    mutated_bars.loc[cut + 1:, "close_split"] *= 3.0
    mutated = leaky(mutated_bars)

    a = original["LEAKY_vol_rank"].iloc[: cut + 1].to_numpy()
    b = mutated["LEAKY_vol_rank"].iloc[: cut + 1].to_numpy()

    assert not np.allclose(a, b, equal_nan=True), (
        "The leakage detector failed to catch a known-leaky feature."
    )


# =======================================================================
# Expanding vs full-sample normalisation
# =======================================================================

def test_expanding_percentile_is_causal():
    s = pd.Series(RNG.normal(size=600))
    original = regime.expanding_percentile(s, min_periods=100)

    mutated = s.copy()
    mutated.iloc[400:] += 50.0
    after = regime.expanding_percentile(mutated, min_periods=100)

    assert np.allclose(original.iloc[:400], after.iloc[:400], equal_nan=True)


def test_full_sample_rank_is_not_causal():
    """Documents precisely what expanding_percentile replaces."""
    s = pd.Series(RNG.normal(size=600))
    original = s.rank(pct=True)

    mutated = s.copy()
    mutated.iloc[400:] += 50.0

    assert not np.allclose(original.iloc[:400], mutated.rank(pct=True).iloc[:400])


def test_expanding_percentile_is_bounded():
    s = pd.Series(RNG.normal(size=500))
    vals = regime.expanding_percentile(s, min_periods=50).dropna()
    assert vals.between(0.0, 1.0).all()


def test_expanding_zscore_is_causal():
    s = pd.Series(RNG.normal(size=600))
    original = regime.expanding_zscore(s, min_periods=100)

    mutated = s.copy()
    mutated.iloc[400:] *= 10.0

    assert np.allclose(
        original.iloc[:400],
        regime.expanding_zscore(mutated, min_periods=100).iloc[:400],
        equal_nan=True,
    )


def test_cumulative_max_is_causal():
    s = pd.Series([1.0, 5.0, 3.0, 2.0, 9.0, 4.0])
    dd = regime.drawdown_from_peak(s)
    # Index 3 sits 60% below the peak of 5 seen so far, not below the later 9.
    assert dd.iloc[3] == pytest.approx(2.0 / 5.0 - 1.0)


# =======================================================================
# Cross-sectional safety
# =======================================================================

def test_cross_sectional_ranks_within_date_only():
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-02"] * 3 + ["2023-01-03"] * 3),
        "symbol": ["A", "B", "C"] * 2,
        "return_5d": [0.01, 0.02, 0.03, 0.50, 0.60, 0.70],
    })
    ranked = regime.add_cross_sectional(panel, ["return_5d"])

    day1 = ranked[ranked["date"] == "2023-01-02"]["xs_rank_return_5d"].tolist()
    day2 = ranked[ranked["date"] == "2023-01-03"]["xs_rank_return_5d"].tolist()

    # Both days rank 1/3, 2/3, 3/3 internally despite wildly different levels.
    assert day1 == day2


def test_cross_sectional_unaffected_by_future_dates():
    base = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-02"] * 3),
        "symbol": ["A", "B", "C"],
        "return_5d": [0.01, 0.02, 0.03],
    })
    future = pd.DataFrame({
        "date": pd.to_datetime(["2023-06-01"] * 3),
        "symbol": ["A", "B", "C"],
        "return_5d": [9.0, 8.0, 7.0],
    })

    alone = regime.add_cross_sectional(base, ["return_5d"])
    together = regime.add_cross_sectional(
        pd.concat([base, future], ignore_index=True), ["return_5d"]
    )

    assert np.allclose(
        alone["xs_rank_return_5d"],
        together[together["date"] == "2023-01-02"]["xs_rank_return_5d"],
    )


# =======================================================================
# Labels
# =======================================================================

def test_tradeable_label_is_open_to_open_after_the_signal():
    bars = synthetic_bars(n=50)
    out = add_labels(bars, horizon=1)

    i = 10
    expected = out["adj_open"].iloc[i + 2] / out["adj_open"].iloc[i + 1] - 1
    assert out[TRADEABLE_LABEL].iloc[i] == pytest.approx(expected)


def test_tradeable_label_never_uses_the_signal_day_price():
    """The entry price must be unreachable from the decision.

    Changing the close of day t must not move the label for day t.
    """
    bars = synthetic_bars(n=60)
    original = add_labels(bars, horizon=1)

    mutated_bars = bars.copy()
    mutated_bars.loc[10, "adj_close"] *= 1.5
    mutated_bars.loc[10, "close_split"] *= 1.5
    mutated = add_labels(mutated_bars, horizon=1)

    assert mutated[TRADEABLE_LABEL].iloc[10] == pytest.approx(
        original[TRADEABLE_LABEL].iloc[10]
    )


def test_research_label_does_use_the_signal_day_close():
    """Which is exactly why it is not the training target."""
    bars = synthetic_bars(n=60)
    original = add_labels(bars, horizon=1)

    mutated_bars = bars.copy()
    mutated_bars.loc[10, "adj_close"] *= 1.5
    mutated_bars.loc[10, "close_split"] *= 1.5
    mutated = add_labels(mutated_bars, horizon=1)

    assert mutated[RESEARCH_LABEL].iloc[10] != pytest.approx(
        original[RESEARCH_LABEL].iloc[10]
    )


def test_last_rows_are_unlabelled_not_dropped():
    """The newest row is what the live job predicts on; it must survive."""
    bars = synthetic_bars(n=50)
    out = add_labels(bars, horizon=1)

    assert len(out) == 50
    assert pd.isna(out[TRADEABLE_LABEL].iloc[-1])
    assert pd.isna(out[TRADEABLE_LABEL].iloc[-2])
    assert not pd.isna(out[TRADEABLE_LABEL].iloc[-3])


def test_direction_matches_the_tradeable_return():
    bars = synthetic_bars(n=100)
    out = add_labels(bars).dropna(subset=[TRADEABLE_LABEL])

    expected = (out[TRADEABLE_LABEL] > 0).astype(float)
    assert (out[DIRECTION_LABEL] == expected).all()


def test_zero_horizon_rejected():
    bars = synthetic_bars(n=30)
    with pytest.raises(ValueError):
        add_labels(bars, horizon=0)


# =======================================================================
# Pipeline hygiene
# =======================================================================

def test_no_label_column_is_a_model_input():
    """A label leaking into the feature matrix is instant, total leakage."""
    panel = build_panel({"A.NS": synthetic_bars(n=600)})
    cols = set(feature_columns(panel))

    for label in LABEL_COLUMNS:
        assert label not in cols
    assert "adj_close" not in cols
    assert "adj_open" not in cols
    assert "close_raw" not in cols


def test_no_feature_correlates_perfectly_with_the_label():
    """A near-1.0 correlation means the label got in through a side door."""
    panel = build_panel({"A.NS": synthetic_bars(n=800)})
    panel = drop_unlabelled(panel)

    label = panel[TRADEABLE_LABEL]
    suspicious = []
    for col in feature_columns(panel):
        series = panel[col]
        if series.notna().sum() < 100 or series.nunique() < 3:
            continue
        corr = series.corr(label)
        if pd.notna(corr) and abs(corr) > 0.95:
            suspicious.append(f"{col} (r={corr:.3f})")

    assert not suspicious, f"Suspiciously label-correlated features: {suspicious}"


def test_warmup_rows_are_trimmed():
    panel = build_panel({"A.NS": synthetic_bars(n=600)})
    assert len(panel) == 600 - 252


def test_short_history_yields_nothing_rather_than_partial_features():
    panel = build_panel({"A.NS": synthetic_bars(n=100)})
    assert panel.empty


def test_features_are_reasonably_populated_after_warmup():
    panel = build_panel({"A.NS": synthetic_bars(n=900)})
    from trademind.features.pipeline import coverage_report

    worst = coverage_report(panel).iloc[0]
    assert worst["missing_rate"] < 0.10, (
        f"{worst['feature']} is {worst['missing_rate']:.0%} missing after warm-up"
    )


def test_panel_survives_a_split():
    """Corporate actions must not produce a fake momentum spike."""
    splits = [1.0] * 800
    splits[500] = 2.0
    panel = build_panel({"A.NS": synthetic_bars(n=800, splits=splits)})

    # No single-day return should look like a 50% crash.
    assert panel["return_1d"].abs().max() < 0.30


def test_drop_unlabelled_only_removes_unlabelled():
    panel = build_panel({"A.NS": synthetic_bars(n=600)})
    trained = drop_unlabelled(panel)

    assert trained[TRADEABLE_LABEL].notna().all()
    assert len(trained) == panel[TRADEABLE_LABEL].notna().sum()


def test_unsorted_bars_rejected():
    bars = synthetic_bars(n=300).iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="date-sorted"):
        build_symbol_features(bars)


def test_multi_symbol_panel_is_date_sorted():
    panel = build_panel({
        "A.NS": synthetic_bars(n=600, symbol="A.NS", seed=1),
        "B.NS": synthetic_bars(n=600, symbol="B.NS", seed=2),
    })
    assert panel["date"].is_monotonic_increasing
    assert panel["symbol"].nunique() == 2
    assert any(c.startswith("xs_rank_") for c in panel.columns)


def test_trim_warmup_on_short_frame_returns_empty():
    assert trim_warmup(pd.DataFrame({"a": [1, 2, 3]}), sessions=252).empty
