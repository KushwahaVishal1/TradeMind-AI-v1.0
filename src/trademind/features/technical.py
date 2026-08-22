"""Technical indicators.

Every function here is causal by construction: the value at index *t* depends
only on rows ``0..t``. pandas' ``rolling`` and ``ewm`` are already causal, so
the discipline is mostly about what *not* to do:

- no ``shift(-n)``
- no ``center=True`` on a rolling window
- no ``bfill``, which pulls a future value backwards
- no full-sample statistics (``.mean()``, ``.quantile()`` over the whole
  series), which is the subtlest trap and the one that ruins percentile and
  z-score features. See ``regime.py``.

Using ``close[t]`` in a feature for *t* is legitimate — the decision is made
after the close of *t* and fills at the open of *t+1*. See ``labels.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    # adjust=False gives the recursive form, which is what a live system
    # computes incrementally and therefore what the backtest should match.
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI. Returns 0-100."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder smoothing is an EMA with alpha = 1/window.
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # avg_loss == 0 means an unbroken run of gains: RSI is 100 by definition.
    return out.where(avg_loss != 0.0, 100.0).where(avg_gain.notna())


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": macd_line - signal_line,
        }
    )


def bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger bands, plus normalised width and position.

    Width and position are what the model gets; the raw bands are price-scaled
    and therefore not comparable across symbols.
    """
    mid = sma(series, window)
    std = series.rolling(window, min_periods=window).std(ddof=0)

    upper = mid + num_std * std
    lower = mid - num_std * std
    span = (upper - lower).replace(0.0, np.nan)

    return pd.DataFrame(
        {
            "bb_width": (upper - lower) / mid.replace(0.0, np.nan),
            # 0 at the lower band, 1 at the upper.
            "bb_position": (series - lower) / span,
        }
    )


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range, Wilder-smoothed."""
    return (
        true_range(high, low, close)
        .ewm(alpha=1 / window, adjust=False, min_periods=window)
        .mean()
    )


def atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """ATR as a fraction of price, so it is comparable across symbols."""
    return atr(high, low, close, window) / close.replace(0.0, np.nan)


def distance_from_sma(series: pd.Series, window: int) -> pd.Series:
    """Fractional distance from a moving average. Scale-free."""
    ma = sma(series, window)
    return series / ma.replace(0.0, np.nan) - 1.0


def build(df: pd.DataFrame) -> pd.DataFrame:
    """All technical features for one symbol.

    Computed on ``adj_close`` — the total-return series — so that dividends do
    not appear as spurious gaps in momentum indicators.
    """
    close = df["adj_close"]
    factor = df["adj_factor"]
    high = df["high_split"] * factor
    low = df["low_split"] * factor

    out = pd.DataFrame(index=df.index)

    for w in (20, 50, 200):
        out[f"dist_sma_{w}"] = distance_from_sma(close, w)

    out["rsi_14"] = rsi(close, 14)
    out = out.join(macd(close))
    out = out.join(bollinger(close, 20))
    out["atr_pct_14"] = atr_pct(high, low, close, 14)

    # Trend agreement: is the fast average above the slow one?
    out["sma_50_over_200"] = (sma(close, 50) / sma(close, 200).replace(0.0, np.nan)) - 1.0

    return out
