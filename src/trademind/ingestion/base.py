"""Market-data schema and the provider interface.

## The three price series, and why there must be three

The roadmap requires that the backtester execute at "raw tradable prices" while
features are computed on an adjusted series. Those are different numbers, and
conflating them is a silent, hard-to-detect bug: adjusted prices are revised
retroactively every time a split or dividend occurs, so a backtest that fills
orders at adjusted prices is filling at a price that did not exist on the day.

This project therefore carries three explicitly named series and never lets
them blur together:

``open_raw, high_raw, low_raw, close_raw``
    As-traded on the exchange that session. The number a human would have paid.
    **Used only by the backtester for execution.** Discontinuous across splits.

``close_split``
    Split-adjusted, dividend-unadjusted. Continuous through splits.
    Useful for charting and for volume normalisation.

``adj_close``
    Total-return series: split- and dividend-adjusted. Continuous.
    **Used only for feature and target computation.** Never for execution.

A naming convention this loud is deliberate. ``close`` alone is banned from the
schema so that no downstream module can reach for "the" close price and get
whichever one happened to be there.

## The yfinance caveat

yfinance does not serve genuine as-traded prices. With ``auto_adjust=False`` its
OHLC is already split-adjusted, and ``Adj Close`` is additionally
dividend-adjusted. The as-traded series must therefore be *reconstructed* by
reversing the split adjustment — see ``corporate_actions.reconstruct_raw``.

The reconstruction is exact when the split history is complete and correct, and
wrong in proportion to any split the provider missed. That is a real limitation
of free data and is disclosed rather than hidden. Phase 1 reconciles what it can
and logs the rest.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Sequence

import pandas as pd

# Columns every provider must return, in order.
RAW_COLUMNS: tuple[str, ...] = (
    "date",
    "symbol",
    "open_split",
    "high_split",
    "low_split",
    "close_split",
    "volume",
    "dividend",
    "split_ratio",
)

# Columns added by the corporate-action layer.
DERIVED_COLUMNS: tuple[str, ...] = (
    "open_raw",
    "high_raw",
    "low_raw",
    "close_raw",
    "adj_close",
)

PRICE_COLUMNS_SPLIT: tuple[str, ...] = (
    "open_split", "high_split", "low_split", "close_split",
)
PRICE_COLUMNS_RAW: tuple[str, ...] = (
    "open_raw", "high_raw", "low_raw", "close_raw",
)


class ProviderError(RuntimeError):
    """The upstream data source failed or returned something unusable."""


class MarketDataProvider(ABC):
    """Interface every data source implements.

    Keeping this abstract is not ceremony: yfinance is a free source with known
    corporate-action gaps, and being able to swap in a paid feed later without
    touching the feature or model layers is the point.
    """

    name: str = "abstract"

    @abstractmethod
    def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Return daily bars for ``symbol`` over ``[start, end]``.

        Must return a DataFrame with exactly ``RAW_COLUMNS``, sorted by date
        ascending, one row per trading session, no duplicates.

        ``split_ratio`` is 1.0 on non-split days; a 2-for-1 split is 2.0 on the
        ex-date. ``dividend`` is 0.0 on non-dividend days, in currency units per
        share, on the ex-date.
        """

    def fetch_many(
        self, symbols: Sequence[str], start: date, end: date
    ) -> dict[str, pd.DataFrame]:
        return {s: self.fetch(s, start, end) for s in symbols}


def empty_frame() -> pd.DataFrame:
    """A correctly-typed, zero-row frame — used when a symbol has no data."""
    return pd.DataFrame({c: pd.Series(dtype=_dtype_for(c)) for c in RAW_COLUMNS})


def _dtype_for(column: str) -> str:
    if column == "date":
        return "datetime64[ns]"
    if column == "symbol":
        return "object"
    if column == "volume":
        return "int64"
    return "float64"


def check_schema(df: pd.DataFrame, columns: Sequence[str] = RAW_COLUMNS) -> None:
    """Raise if required columns are missing. Extra columns are permitted."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ProviderError(f"Missing required columns: {missing}")
