"""Ingestion pipeline orchestration.

Uses a stub provider and an in-memory lake so the test covers ordering and
decision logic without touching the network or the filesystem.

The property under test: **bad data must not reach the lake.** Everything
downstream — features, models, backtest — assumes the lake is trustworthy.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from trademind.ingestion.base import MarketDataProvider, ProviderError, empty_frame
from trademind.ingestion.market_data import MarketDataIngestion
from trademind.storage import PredictionStore, init_db


def bars(closes, splits=None, divs=None, symbol="TEST.NS"):
    n = len(closes)
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2023-01-02", periods=n),
            "symbol": symbol,
            "open_split": closes,
            "high_split": closes * 1.01,
            "low_split": closes * 0.99,
            "close_split": closes,
            "volume": np.full(n, 100_000, dtype="int64"),
            "dividend": np.asarray(divs if divs is not None else [0.0] * n, dtype=float),
            "split_ratio": np.asarray(
                splits if splits is not None else [1.0] * n, dtype=float
            ),
        }
    )


class StubProvider(MarketDataProvider):
    name = "stub"

    def __init__(self, frames: dict[str, pd.DataFrame], fail: set[str] | None = None):
        self.frames = frames
        self.fail = fail or set()
        self.calls: list[str] = []

    def fetch(self, symbol, start, end):
        self.calls.append(symbol)
        if symbol in self.fail:
            raise ProviderError(f"simulated outage for {symbol}")
        return self.frames.get(symbol, empty_frame())


class MemoryLake:
    """Same interface as ParquetLake, backed by a dict."""

    def __init__(self):
        self.data: dict[str, pd.DataFrame] = {}

    def append_raw(self, symbol, df):
        prev = self.data.get(symbol)
        merged = pd.concat([prev, df]) if prev is not None else df
        self.data[symbol] = (
            merged.sort_values("date")
            .drop_duplicates("date", keep="last")
            .reset_index(drop=True)
        )
        return symbol

    def read_raw(self, symbol):
        return self.data.get(symbol)

    def last_date(self, symbol):
        df = self.data.get(symbol)
        return None if df is None or df.empty else pd.Timestamp(df["date"].max()).date()

    def symbols(self):
        return sorted(self.data)


def pipeline(frames, fail=None, tmp_path=None):
    lake = MemoryLake()
    conn = init_db((tmp_path or ".") / "t.db")
    store = PredictionStore(conn)
    ing = MarketDataIngestion(StubProvider(frames, fail), lake, store)
    return ing, lake, store, conn


START, END = date(2023, 1, 1), date(2023, 3, 1)


# -- happy path ----------------------------------------------------------


def test_clean_symbol_is_written(tmp_path):
    ing, lake, _, conn = pipeline(
        {"TEST.NS": bars([100 + i for i in range(25)])}, tmp_path=tmp_path
    )
    result = ing.ingest_symbol("TEST.NS", START, END)

    assert result.ok
    assert result.rows == 25
    assert lake.read_raw("TEST.NS") is not None
    conn.close()


def test_derived_price_columns_reach_the_lake(tmp_path):
    """Enrichment happens before persistence, so all three series are stored."""
    ing, lake, _, conn = pipeline(
        {"TEST.NS": bars([500.0] * 6, splits=[1, 1, 1, 2, 1, 1])}, tmp_path=tmp_path
    )
    ing.ingest_symbol("TEST.NS", START, END)

    stored = lake.read_raw("TEST.NS")
    for col in ("close_raw", "close_split", "adj_close"):
        assert col in stored.columns
    assert stored["close_raw"].iloc[0] == 1000.0
    conn.close()


# -- bad data must not be persisted --------------------------------------


def test_error_severity_blocks_the_write(tmp_path):
    bad = bars([100.0, 101.0, -5.0, 102.0])  # negative price
    ing, lake, _, conn = pipeline({"TEST.NS": bad}, tmp_path=tmp_path)

    result = ing.ingest_symbol("TEST.NS", START, END)

    assert not result.ok
    assert not result.written
    assert lake.read_raw("TEST.NS") is None  # nothing written
    conn.close()


def test_warning_severity_does_not_block(tmp_path):
    # +25%: above the WARNING threshold (20%), below the ERROR threshold (50%).
    warn = bars([100.0, 101.0, 102.0, 128.0, 129.0, 130.0])
    ing, lake, _, conn = pipeline({"TEST.NS": warn}, tmp_path=tmp_path)

    result = ing.ingest_symbol("TEST.NS", START, END)

    assert result.written
    assert result.report.warnings
    conn.close()


def test_jump_exactly_at_threshold_does_not_warn():
    """20.0% is not greater than 20%. Boundaries are asserted, not assumed."""
    from trademind.ingestion.calendar import TradingCalendar
    from trademind.ingestion.data_validator import DataValidator

    report = DataValidator(TradingCalendar()).validate(
        bars([100.0, 100.0, 120.0, 121.0, 122.0]), "TEST.NS"
    )
    assert "ABNORMAL_JUMP" not in {i.code for i in report.issues}


def test_provider_outage_is_contained(tmp_path):
    ing, lake, _, conn = pipeline({}, fail={"TEST.NS"}, tmp_path=tmp_path)
    result = ing.ingest_symbol("TEST.NS", START, END)

    assert not result.ok
    assert "simulated outage" in result.error
    conn.close()


def test_empty_response_is_an_error(tmp_path):
    ing, lake, _, conn = pipeline({}, tmp_path=tmp_path)
    result = ing.ingest_symbol("MISSING.NS", START, END)

    assert not result.ok
    assert lake.read_raw("MISSING.NS") is None
    conn.close()


# -- universe-level behaviour --------------------------------------------


def test_one_bad_symbol_does_not_stop_the_others(tmp_path):
    frames = {
        "GOOD1.NS": bars([100 + i for i in range(20)], symbol="GOOD1.NS"),
        "BAD.NS": bars([100.0, -1.0, 102.0], symbol="BAD.NS"),
        "GOOD2.NS": bars([200 + i for i in range(20)], symbol="GOOD2.NS"),
    }
    ing, lake, _, conn = pipeline(frames, tmp_path=tmp_path)

    summary = ing.ingest_universe(list(frames), START, END)

    assert len(summary.succeeded) == 2
    assert len(summary.failed) == 1
    assert sorted(lake.symbols()) == ["GOOD1.NS", "GOOD2.NS"]
    conn.close()


def test_issues_are_persisted_for_the_retraining_gate(tmp_path):
    """Phase 8 reads this table: an unresolved ERROR blocks retraining."""
    ing, _, store, conn = pipeline({"TEST.NS": bars([100.0, -1.0, 102.0])}, tmp_path=tmp_path)
    run_id = store.start_run("test", config_hash="x")
    ing.ingest_symbol("TEST.NS", START, END, run_id=run_id)

    rows = store.conn.execute(
        "SELECT severity, code FROM data_quality_issues WHERE run_id = ?", (run_id,)
    ).fetchall()

    assert any(r["severity"] == "ERROR" for r in rows)
    conn.close()


def test_info_findings_are_not_persisted(tmp_path):
    """INFO is noise in a gating table; only WARNING and ERROR are stored."""
    ing, _, store, conn = pipeline(
        {"TEST.NS": bars([100 + i for i in range(25)])}, tmp_path=tmp_path
    )
    run_id = store.start_run("test", config_hash="x")
    ing.ingest_symbol("TEST.NS", START, END, run_id=run_id)

    rows = store.conn.execute(
        "SELECT severity FROM data_quality_issues WHERE run_id = ?", (run_id,)
    ).fetchall()

    assert all(r["severity"] != "INFO" for r in rows)
    conn.close()


def test_reingesting_the_same_bars_is_idempotent(tmp_path):
    frame = bars([100 + i for i in range(20)])
    ing, lake, _, conn = pipeline({"TEST.NS": frame}, tmp_path=tmp_path)

    ing.ingest_symbol("TEST.NS", START, END)
    ing.ingest_symbol("TEST.NS", START, END)

    assert len(lake.read_raw("TEST.NS")) == 20
    conn.close()


def test_summary_renders_failures(tmp_path):
    frames = {"BAD.NS": bars([100.0, -1.0], symbol="BAD.NS")}
    ing, _, _, conn = pipeline(frames, tmp_path=tmp_path)

    text = ing.ingest_universe(["BAD.NS"], START, END).render()

    assert "FAILED BAD.NS" in text
    conn.close()
