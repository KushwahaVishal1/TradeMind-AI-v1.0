"""yfinance market-data provider.

Free, no API key, adequate coverage of NSE large-caps. Its weaknesses are
documented in ``base.py`` and handled rather than hidden: OHLC arrives
split-adjusted, so as-traded prices are reconstructed downstream, and its
corporate-action history is occasionally incomplete, which the validator's
``ABNORMAL_JUMP`` check is designed to catch.

Deliberately not tested against the live API. Network-dependent tests are flaky
and slow, and an assertion about today's RELIANCE close is not a useful
regression test. The fixture-driven tests cover the transformation logic; this
class only does I/O and column renaming.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import pandas as pd

from .base import RAW_COLUMNS, MarketDataProvider, ProviderError, empty_frame

log = logging.getLogger(__name__)


class YFinanceProvider(MarketDataProvider):
    name = "yfinance"

    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0) -> None:
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        raw = self._download(symbol, start, end)
        if raw is None or raw.empty:
            log.warning("No data returned for %s over %s..%s", symbol, start, end)
            return empty_frame()
        return self._normalise(raw, symbol)

    # -- I/O ---------------------------------------------------------------

    def _download(self, symbol: str, start: date, end: date):
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "yfinance is not installed. Run: pip install yfinance"
            ) from exc

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                ticker = yf.Ticker(symbol)
                # yfinance treats `end` as exclusive.
                df = ticker.history(
                    start=start.isoformat(),
                    end=(end + timedelta(days=1)).isoformat(),
                    interval="1d",
                    auto_adjust=False,  # keep split-adjusted OHLC + actions
                    actions=True,
                )
                return df
            except Exception as exc:  # pragma: no cover - network dependent
                last_exc = exc
                log.warning(
                    "Fetch %s attempt %d/%d failed: %s", symbol, attempt, self.max_retries, exc
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        raise ProviderError(
            f"Failed to fetch {symbol} after {self.max_retries} attempts"
        ) from last_exc

    # -- shaping -----------------------------------------------------------

    @staticmethod
    def _normalise(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Map the yfinance frame onto RAW_COLUMNS."""
        df = df.copy()

        # Multi-symbol downloads return a column MultiIndex; single ones don't.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        date_col = next((c for c in ("Date", "Datetime", "index") if c in df.columns), None)
        if date_col is None:
            raise ProviderError(f"No date column in yfinance response for {symbol}")

        dates = pd.to_datetime(df[date_col])
        # NSE bars arrive tz-aware in Asia/Kolkata; drop to naive dates so that
        # nothing downstream has to reason about timezones.
        if getattr(dates.dt, "tz", None) is not None:
            dates = dates.dt.tz_localize(None)

        required = ["Open", "High", "Low", "Close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ProviderError(f"yfinance response for {symbol} lacks {missing}")

        out = pd.DataFrame(
            {
                "date": dates.dt.normalize(),
                "symbol": symbol,
                "open_split": df["Open"].astype(float),
                "high_split": df["High"].astype(float),
                "low_split": df["Low"].astype(float),
                "close_split": df["Close"].astype(float),
                "volume": df.get("Volume", 0).fillna(0).astype("int64"),
                "dividend": df.get("Dividends", 0.0).fillna(0.0).astype(float),
                # yfinance reports 0.0 on non-split days; the schema wants 1.0.
                "split_ratio": df.get("Stock Splits", 0.0).fillna(0.0).astype(float),
            }
        )
        out.loc[out["split_ratio"] == 0.0, "split_ratio"] = 1.0

        out = out.dropna(subset=["close_split"])
        out = out.sort_values("date").drop_duplicates("date", keep="last")
        return out.reset_index(drop=True)[list(RAW_COLUMNS)]
