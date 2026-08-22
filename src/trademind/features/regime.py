"""Market regime features.

## The trap this module exists to avoid

The natural way to write "is volatility high right now?" is::

    df["vol_percentile"] = df["volatility_20"].rank(pct=True)      # WRONG

That ranks each day against the *entire* series, including years that had not
happened yet. A day in 2016 gets scored against 2020's volatility. Nothing about
it looks like look-ahead — there is no ``shift(-1)``, no future price, no
obvious future column — and the leakage test in ``tests/test_leakage.py`` catches
it precisely because it mutates the tail and re-checks the head.

The same trap swallows any full-sample statistic: z-scores against
``series.mean()``, min-max scaling against ``series.max()``, quantile binning
via ``pd.qcut``. All of them are standard preprocessing and all of them leak.

The fix is an **expanding window**: rank each day against history up to and
including that day only. The cost is that early observations are ranked against
a thin sample, which is why the first year is discarded as warm-up rather than
being fed to the model on a shorter history.

Cross-sectional features have the same hazard along the other axis: ranking a
symbol against its peers *on the same date* is fine, because every one of those
observations is available at *t*. Ranking against peers pooled across all dates
is not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Below this, an expanding rank is too noisy to mean anything.
MIN_EXPANDING_OBS = 252


def expanding_percentile(series: pd.Series, min_periods: int = MIN_EXPANDING_OBS) -> pd.Series:
    """Rank each observation against history up to and including itself.

    Returns a value in [0, 1]. The causal counterpart of ``rank(pct=True)``.
    """

    def _rank(window: np.ndarray) -> float:
        current = window[-1]
        if np.isnan(current):
            return np.nan
        valid = window[~np.isnan(window)]
        if len(valid) < 2:
            return np.nan
        return float((valid <= current).sum() - 1) / (len(valid) - 1)

    return series.expanding(min_periods=min_periods).apply(_rank, raw=True)


def expanding_zscore(series: pd.Series, min_periods: int = MIN_EXPANDING_OBS) -> pd.Series:
    """Z-score against expanding history. Causal alternative to a full-sample one."""
    mean = series.expanding(min_periods=min_periods).mean()
    std = series.expanding(min_periods=min_periods).std(ddof=1)
    return (series - mean) / std.replace(0.0, np.nan)


def trend_regime(close: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    """Discrete trend state: 1 uptrend, -1 downtrend, 0 undetermined."""
    fast_ma = close.rolling(fast, min_periods=fast).mean()
    slow_ma = close.rolling(slow, min_periods=slow).mean()

    state = pd.Series(np.nan, index=close.index)
    both = fast_ma.notna() & slow_ma.notna()
    state[both & (fast_ma > slow_ma) & (close > slow_ma)] = 1.0
    state[both & (fast_ma < slow_ma) & (close < slow_ma)] = -1.0
    state[both & state.isna()] = 0.0
    return state


def drawdown_from_peak(close: pd.Series) -> pd.Series:
    """Fractional distance below the running maximum.

    ``cummax`` is causal — the running maximum at *t* only sees rows up to *t*.
    Contrast with ``close.max()``, which would not be.
    """
    peak = close.cummax()
    return close / peak.replace(0.0, np.nan) - 1.0


def days_since_high(close: pd.Series, window: int = 252) -> pd.Series:
    """Sessions since the highest close within a trailing window."""

    def _since(w: np.ndarray) -> float:
        return float(len(w) - 1 - int(np.nanargmax(w)))

    return close.rolling(window, min_periods=window).apply(_since, raw=True)


def build(df: pd.DataFrame, volatility: pd.Series | None = None) -> pd.DataFrame:
    """Regime features for one symbol."""
    close = df["adj_close"]
    out = pd.DataFrame(index=df.index)

    out["trend_regime"] = trend_regime(close)
    out["drawdown_from_peak"] = drawdown_from_peak(close)
    out["days_since_high_252"] = days_since_high(close, 252)

    if volatility is not None:
        # Expanding, not full-sample. The whole point of this module.
        out["vol_percentile_expanding"] = expanding_percentile(volatility)
        out["vol_zscore_expanding"] = expanding_zscore(volatility)

    return out


def add_cross_sectional(panel: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Rank each symbol against its peers, within each date.

    Safe because every observation being compared shares the same timestamp and
    is therefore available at *t*. Grouping by date is what makes it safe;
    ranking the pooled panel would not be.

    Relative strength is the point: a 3% move means something different when
    every peer also moved 3%.
    """
    out = panel.copy()
    for col in columns:
        if col not in out.columns:
            continue
        out[f"xs_rank_{col}"] = out.groupby("date")[col].rank(pct=True)
    return out
