"""Corporate actions: splits and dividends.

The invariant this module exists to protect:

    100 shares x Rs.1,000 = Rs.100,000
    2:1 split
    200 shares x Rs.500   = Rs.100,000

A split changes the *share count* and the *cost basis*. It does not change
wealth. Any code path where a split moves portfolio value is a bug, and
``tests/test_corporate_actions.py`` asserts it directly.

Two transformations live here, and they run in opposite directions:

``reconstruct_raw``
    Provider gives split-adjusted prices; we reverse the adjustment to recover
    as-traded prices for execution.

``compute_adj_close``
    Provider gives split-adjusted prices; we additionally remove dividends to
    build the total-return series used for features and labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import PRICE_COLUMNS_SPLIT, ProviderError

# Below this the ratio is meaningless noise; above it, a data error.
MIN_SPLIT_RATIO = 0.01
MAX_SPLIT_RATIO = 100.0


def _validate_actions(df: pd.DataFrame) -> None:
    if (df["split_ratio"] <= 0).any():
        bad = df.loc[df["split_ratio"] <= 0, "date"].tolist()
        raise ProviderError(f"Non-positive split_ratio on {bad[:5]}")
    out_of_range = df["split_ratio"].between(MIN_SPLIT_RATIO, MAX_SPLIT_RATIO)
    if not out_of_range.all():
        bad = df.loc[~out_of_range, ["date", "split_ratio"]].head().to_dict("records")
        raise ProviderError(f"split_ratio outside plausible range: {bad}")
    if (df["dividend"] < 0).any():
        raise ProviderError("Negative dividend encountered")


def cumulative_split_factor(split_ratio: pd.Series) -> pd.Series:
    """For each row, the product of every split ratio *strictly after* it.

    A price on day t was quoted in pre-split terms relative to every split that
    happened later, so multiplying the split-adjusted price by this factor
    recovers the as-traded price.

    With a single 2:1 split on day 5, rows 0-4 get 2.0 and rows 5+ get 1.0. The
    split ratio on the ex-date itself is excluded, because on the ex-date the
    stock already trades at the post-split price.
    """
    ratios = split_ratio.to_numpy(dtype=float)
    # Reverse cumulative product, shifted so each row excludes its own ratio.
    rev = np.cumprod(ratios[::-1])[::-1]
    factor = np.empty_like(rev)
    factor[:-1] = rev[1:]
    factor[-1] = 1.0
    return pd.Series(factor, index=split_ratio.index)


def reconstruct_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``*_raw`` as-traded price columns by reversing split adjustment.

    Volume is left untouched. Provider volume is share count as reported and is
    not re-scaled here; relative-volume features normalise it anyway, and
    inventing a raw volume from an approximate split history would add error
    without adding information.
    """
    _validate_actions(df)
    out = df.copy()
    factor = cumulative_split_factor(out["split_ratio"])

    for col in PRICE_COLUMNS_SPLIT:
        out[col.replace("_split", "_raw")] = out[col] * factor

    out["split_factor"] = factor
    return out


def compute_adj_close(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``adj_close``: the total-return series used for features and labels.

    Standard back-adjustment. Walking backwards from the most recent bar, each
    dividend scales all prior prices by ``1 - dividend / prior_close``, so that
    the returns computed off the series include the cash the holder received.

    The most recent bar always equals its split-adjusted close, which means the
    series is revised whenever a new dividend arrives. That is expected and is
    exactly why this series must never be used for execution.
    """
    _validate_actions(df)
    out = df.copy()

    close = out["close_split"].to_numpy(dtype=float)
    div = out["dividend"].to_numpy(dtype=float)
    n = len(close)

    if n == 0:
        out["adj_close"] = pd.Series(dtype=float)
        return out

    factors = np.ones(n, dtype=float)
    cum = 1.0
    # Walk backwards; a dividend on day i discounts every day before i.
    for i in range(n - 1, 0, -1):
        factors[i] = cum
        prior_close = close[i - 1]
        if div[i] > 0 and prior_close > 0:
            cum *= 1.0 - div[i] / prior_close
    factors[0] = cum

    out["adj_close"] = close * factors
    return out


def apply_corporate_actions(df: pd.DataFrame) -> pd.DataFrame:
    """Full corporate-action enrichment: raw reconstruction + total return."""
    if df.empty:
        return df.copy()
    if not df["date"].is_monotonic_increasing:
        raise ProviderError("Corporate actions require date-sorted input")
    return compute_adj_close(reconstruct_raw(df))


def adjust_position_for_split(
    shares: float, cost_basis: float, split_ratio: float
) -> tuple[float, float]:
    """Apply a split to an open position.

    Returns ``(new_shares, new_cost_basis)``. Position value is unchanged by
    construction: shares multiply by the ratio, basis divides by it.

    The backtester calls this on every ex-date for every held position. It is
    the operational half of the invariant at the top of this module.
    """
    if split_ratio <= 0:
        raise ValueError(f"split_ratio must be positive, got {split_ratio}")
    return shares * split_ratio, cost_basis / split_ratio
