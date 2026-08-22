"""Volume features.

Raw share counts are useless to a cross-sectional model: RELIANCE and a
mid-cap trade in different orders of magnitude, and any single symbol's volume
trends over a decade. Everything here is normalised against that symbol's own
recent history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def relative_volume(volume: pd.Series, window: int = 20) -> pd.Series:
    """Today's volume as a multiple of its trailing average.

    The window excludes the current day, so a single huge session does not
    inflate its own baseline and disguise itself as ordinary.
    """
    baseline = volume.shift(1).rolling(window, min_periods=window).mean()
    return volume / baseline.replace(0.0, np.nan)


def volume_zscore(volume: pd.Series, window: int = 60) -> pd.Series:
    """Standardised against a trailing window that excludes today."""
    prior = volume.shift(1)
    mean = prior.rolling(window, min_periods=window).mean()
    std = prior.rolling(window, min_periods=window).std(ddof=1)
    return (volume - mean) / std.replace(0.0, np.nan)


def build(df: pd.DataFrame) -> pd.DataFrame:
    volume = df["volume"].astype(float)
    close = df["adj_close"]
    out = pd.DataFrame(index=df.index)

    out["relative_volume_20"] = relative_volume(volume, 20)
    out["volume_zscore_60"] = volume_zscore(volume, 60)

    # Log turnover: a liquidity level feature, log-scaled because the raw
    # distribution spans orders of magnitude.
    turnover = (volume * close).replace(0.0, np.nan)
    out["log_turnover"] = np.log(turnover)
    out["turnover_trend_20"] = (
        np.log(turnover) - np.log(turnover).rolling(20, min_periods=20).mean()
    )

    # Volume-weighted direction: are up days trading heavier than down days?
    ret = close.pct_change()
    signed = np.sign(ret) * volume
    out["volume_direction_20"] = (
        signed.rolling(20, min_periods=20).sum()
        / volume.rolling(20, min_periods=20).sum().replace(0.0, np.nan)
    )

    return out
