"""Intraday ingestion retains timestamp precision and excludes unfinished bars."""

import pandas as pd
import pytest

from trademind.ingestion.intraday import (
    candle_path,
    load_candles,
    normalise_candles,
    refresh_candles,
)


def bars():
    return pd.DataFrame(
        {"Open": [100.] * 3, "High": [102.] * 3,
         "Low": [99.] * 3, "Close": [101.] * 3, "Volume": [0] * 3},
        index=pd.date_range("2026-09-21 09:15", periods=3, freq="5min",
                            tz="Asia/Kolkata"),
    )


def test_closed_candles_and_timezone():
    raw = bars()
    raw.index = raw.index.tz_convert("UTC")
    result = normalise_candles(raw, "^NSEI", "5m", "2026-09-21T09:26:00+05:30")
    assert result.timestamp.dt.strftime("%H:%M").tolist() == ["09:15", "09:20"]
    assert str(result.timestamp.dt.tz) == "Asia/Kolkata"
    assert result.volume.isna().all()


def test_unknown_timezone_is_rejected():
    raw = bars()
    raw.index = raw.index.tz_localize(None)
    with pytest.raises(ValueError, match="without a timezone"):
        normalise_candles(raw, "^NSEI", "5m")


def test_bad_ohlc_and_outside_session_removed():
    raw = bars()
    raw.iloc[1, raw.columns.get_loc("High")] = 98
    extra = raw.iloc[:1].copy()
    extra.index = pd.DatetimeIndex([pd.Timestamp("2026-09-21 15:30", tz="Asia/Kolkata")])
    result = normalise_candles(pd.concat([raw, extra]), "^NSEI", "5m",
                               "2026-09-21T16:00:00+05:30")
    assert len(result) == 2


def test_refresh_merges_and_failure_preserves_cache(tmp_path, monkeypatch):
    import yfinance as yf

    raw = bars()
    raw.index -= pd.Timedelta(days=7)

    class Ticker:
        def __init__(self, symbol):
            assert symbol == "^NSEI"

        def history(self, **kwargs):
            assert kwargs["interval"] == "5m"
            return raw

    monkeypatch.setattr(yf, "Ticker", Ticker)
    refresh_candles(tmp_path, "^NSEI")
    raw.loc[raw.index[-1], "Close"] = 100.5
    result = refresh_candles(tmp_path, "^NSEI")
    assert len(result) == 3
    assert result.close.iloc[-1] == 100.5
    saved = candle_path(tmp_path, "^NSEI", "5m").read_bytes()
    raw = pd.DataFrame()
    with pytest.raises(ValueError, match="No intraday"):
        refresh_candles(tmp_path, "^NSEI")
    assert candle_path(tmp_path, "^NSEI", "5m").read_bytes() == saved
    assert len(load_candles(tmp_path, "^NSEI", "5m")) == 3


def test_invalid_path_inputs_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        candle_path(tmp_path, "../../test", "5m")
    with pytest.raises(ValueError):
        candle_path(tmp_path, "^NSEI", "1d")
