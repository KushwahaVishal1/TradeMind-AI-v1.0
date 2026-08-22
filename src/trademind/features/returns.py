"""Return and momentum features.

All computed on ``adj_close`` so that dividends count as return rather than
appearing as a price gap.

Momentum features deliberately skip the most recent few sessions
(``momentum_20_5`` is the 20-day return excluding the last 5). Short-horizon
reversal and longer-horizon momentum are opposing effects; blending them into
one number lets them cancel. Separating them is standard practice in the
cross-sectional momentum literature and gives the model two distinguishable
signals instead of one muddled one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simple_return(series: pd.Series, periods: int = 1) -> pd.Series:
    return series.pct_change(periods)


def log_return(series: pd.Series, periods: int = 1) -> pd.Series:
    """Log returns: additive across time, which rolling sums rely on."""
    return np.log(series / series.shift(periods))


def skip_momentum(series: pd.Series, window: int, skip: int) -> pd.Series:
    """Return over ``window`` sessions ending ``skip`` sessions ago."""
    if skip >= window:
        raise ValueError(f"skip ({skip}) must be < window ({window})")
    return series.shift(skip) / series.shift(window) - 1.0


def build(df: pd.DataFrame) -> pd.DataFrame:
    close = df["adj_close"]
    out = pd.DataFrame(index=df.index)

    for p in (1, 2, 5, 10, 20, 60):
        out[f"return_{p}d"] = simple_return(close, p)

    out["log_return_1d"] = log_return(close, 1)

    # Momentum with the short-term reversal window excluded.
    out["momentum_20_5"] = skip_momentum(close, 20, 5)
    out["momentum_60_5"] = skip_momentum(close, 60, 5)
    out["momentum_120_20"] = skip_momentum(close, 120, 20)

    # Consistency of direction over the last 20 sessions: 0 = all down, 1 = all up.
    up = (close.diff() > 0).astype(float)
    out["up_day_ratio_20"] = up.rolling(20, min_periods=20).mean()

    # Overnight vs intraday decomposition. A persistent split between the two
    # is informative and is invisible in a close-to-close return alone.
    adj_open = df["adj_open"]
    out["overnight_return"] = adj_open / close.shift(1) - 1.0
    out["intraday_return"] = close / adj_open.replace(0.0, np.nan) - 1.0

    return out
