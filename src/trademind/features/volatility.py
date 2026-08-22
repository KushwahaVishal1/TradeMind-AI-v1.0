"""Volatility features.

Realised volatility from close-to-close returns, plus two range-based
estimators that use the intraday high and low and are therefore far more
efficient per observation.

Everything is annualised with 252 trading days so the numbers read like the
percentages a practitioner expects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def realised_vol(returns: pd.Series, window: int) -> pd.Series:
    """Annualised standard deviation of returns over a trailing window."""
    return returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(TRADING_DAYS)


def parkinson_vol(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """Parkinson estimator from the high-low range.

    Roughly five times more efficient than close-to-close for the same sample,
    because a day's range carries more information than its endpoints. Blind to
    overnight gaps, which is why it is paired with the close-to-close measure
    rather than replacing it.
    """
    ratio = np.log(high / low.replace(0.0, np.nan))
    factor = 1.0 / (4.0 * np.log(2.0))
    var = (ratio ** 2).rolling(window, min_periods=window).mean() * factor
    return np.sqrt(var * TRADING_DAYS)


def garman_klass_vol(open_: pd.Series, high: pd.Series, low: pd.Series,
                     close: pd.Series, window: int) -> pd.Series:
    """Garman-Klass estimator using the full OHLC bar."""
    hl = np.log(high / low.replace(0.0, np.nan)) ** 2
    co = np.log(close / open_.replace(0.0, np.nan)) ** 2
    daily = 0.5 * hl - (2.0 * np.log(2.0) - 1.0) * co
    var = daily.rolling(window, min_periods=window).mean().clip(lower=0.0)
    return np.sqrt(var * TRADING_DAYS)


def build(df: pd.DataFrame) -> pd.DataFrame:
    close = df["adj_close"]
    factor = df["adj_factor"]
    high = df["high_split"] * factor
    low = df["low_split"] * factor
    adj_open = df["adj_open"]

    ret = close.pct_change()
    out = pd.DataFrame(index=df.index)

    for w in (10, 20, 60):
        out[f"volatility_{w}"] = realised_vol(ret, w)

    out["parkinson_vol_20"] = parkinson_vol(high, low, 20)
    out["garman_klass_vol_20"] = garman_klass_vol(adj_open, high, low, close, 20)

    # Volatility term structure. Above 1 means short-term vol is elevated
    # relative to the longer baseline -- a regime change in progress.
    out["vol_ratio_10_60"] = (
        out["volatility_10"] / out["volatility_60"].replace(0.0, np.nan)
    )

    # Downside deviation: only negative returns contribute.
    downside = ret.where(ret < 0, 0.0)
    out["downside_vol_20"] = (
        downside.rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252)
    )

    return out
