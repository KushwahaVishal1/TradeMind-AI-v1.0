"""Separate index candle store; never fed into the daily equity models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INDICES = {"NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK", "SENSEX": "^BSESN"}
INTERVALS = {"1m": 1, "5m": 5, "15m": 15}
TIMEZONE = "Asia/Kolkata"


def normalise_candles(raw, symbol, interval, now=None):
    """Keep valid, completed regular-session candles with aware timestamps."""
    minutes = INTERVALS[interval]
    if raw is None or raw.empty:
        raise ValueError(f"No intraday data returned for {symbol}")
    frame = raw.rename(columns=str.lower).copy()
    prices = ["open", "high", "low", "close"]
    if not set(prices) <= set(frame):
        raise ValueError(f"Missing OHLC columns for {symbol}")
    stamps = pd.DatetimeIndex(frame.index)
    if stamps.tz is None:
        raise ValueError("Provider returned timestamps without a timezone")
    frame["timestamp"] = stamps.tz_convert(TIMEZONE)
    frame = frame.reset_index(drop=True)
    clock = pd.Timestamp.now(tz=TIMEZONE) if now is None else pd.Timestamp(now)
    if clock.tzinfo is None:
        raise ValueError("The observation time must include a timezone")
    clock = clock.tz_convert(TIMEZONE)
    minute = frame.timestamp.dt.hour * 60 + frame.timestamp.dt.minute
    ends = frame.timestamp + pd.Timedelta(minutes=minutes)
    session_end = frame.timestamp.dt.normalize() + pd.Timedelta(hours=15, minutes=30)
    ends = ends.where(ends <= session_end, session_end)
    frame[prices] = frame[prices].apply(pd.to_numeric, errors="coerce")
    valid = (
        np.isfinite(frame[prices]).all(axis=1)
        & frame[prices].gt(0).all(axis=1)
        & frame.high.ge(frame[["open", "close", "low"]].max(axis=1))
        & frame.low.le(frame[["open", "close", "high"]].min(axis=1))
        & minute.ge(9 * 60 + 15) & minute.lt(15 * 60 + 30)
        & frame.timestamp.dt.dayofweek.lt(5) & ends.le(clock)
    )
    frame = frame.loc[valid].copy()
    if frame.empty:
        raise ValueError(f"No valid completed candles for {symbol}")
    frame["volume"] = (
        pd.to_numeric(frame["volume"], errors="coerce").where(lambda s: s > 0)
        if "volume" in frame else np.nan
    )
    frame["symbol"] = symbol
    frame["interval"] = interval
    frame["fetched_at"] = clock
    return frame[[
        "timestamp", "symbol", "interval", *prices, "volume", "fetched_at"
    ]].drop_duplicates("timestamp", keep="last").sort_values("timestamp")


def candle_path(data_root: Path, symbol: str, interval: str) -> Path:
    if symbol not in INDICES.values() or interval not in INTERVALS:
        raise ValueError("Unsupported index or interval")
    return Path(data_root) / "intraday" / interval / f"{symbol[1:]}.parquet"


def load_candles(data_root: Path, symbol: str, interval: str) -> pd.DataFrame:
    path = candle_path(data_root, symbol, interval)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def refresh_candles(data_root: Path, symbol: str, interval: str = "5m") -> pd.DataFrame:
    """Fetch five recent days and merge by candle start, preserving older history."""
    import yfinance as yf

    path = candle_path(data_root, symbol, interval)
    raw = yf.Ticker(symbol).history(
        period="5d", interval=interval, auto_adjust=False,
        actions=False, prepost=False, timeout=15, raise_errors=True,
    )
    fresh = normalise_candles(raw, symbol, interval)
    existing = load_candles(data_root, symbol, interval)
    merged = pd.concat([existing, fresh], ignore_index=True)
    merged = merged.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replace only after a complete successful download and serialization.
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        merged.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return merged
