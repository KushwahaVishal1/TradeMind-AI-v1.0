"""Parquet data lake.

Layout::

    data/raw/market_data/symbol=RELIANCE.NS/daily.parquet
    data/processed/features/symbol=RELIANCE.NS/f1.parquet

Partitioned by symbol because every access pattern in this project is
either one symbol's full history or a scan across all symbols — and one file
per symbol makes an incremental daily append cheap without a compaction step.

The ``symbol=`` directory convention is Hive-style, so DuckDB can read the
whole lake with a single glob and recover ``symbol`` as a column::

    SELECT * FROM 'data/raw/market_data/*/daily.parquet'

Feature files carry the feature version in the filename, so bumping
``features.version`` in config produces a new file rather than overwriting the
old one — which keeps a prediction's lineage genuinely reproducible.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def _safe(symbol: str) -> str:
    """Filesystem-safe symbol, reversibly enough for a partition name."""
    return symbol.replace("/", "_").replace("\\", "_")


class ParquetLake:
    """Read/write access to the columnar data lake."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # -- paths ------------------------------------------------------------

    def raw_path(self, symbol: str) -> Path:
        return self.root / "raw" / "market_data" / f"symbol={_safe(symbol)}" / "daily.parquet"

    def feature_path(self, symbol: str, feature_version: str) -> Path:
        return (self.root / "processed" / "features"
                / f"symbol={_safe(symbol)}" / f"{feature_version}.parquet")

    # -- market data ------------------------------------------------------

    def write_raw(self, symbol: str, df: pd.DataFrame) -> Path:
        path = self.raw_path(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        log.debug("Wrote %d rows to %s", len(df), path)
        return path

    def read_raw(self, symbol: str) -> pd.DataFrame | None:
        path = self.raw_path(symbol)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def append_raw(self, symbol: str, new: pd.DataFrame) -> Path:
        """Merge new bars into existing history.

        Later rows win on a date collision, so a re-download that includes
        corrected bars supersedes the old ones. This is the one place in the
        project where overwriting is correct: market data gets legitimately
        restated, unlike predictions, which are historical facts.
        """
        existing = self.read_raw(symbol)
        if existing is None or existing.empty:
            return self.write_raw(symbol, new)

        merged = pd.concat([existing, new], ignore_index=True)
        merged = (
            merged.sort_values("date")
            .drop_duplicates("date", keep="last")
            .reset_index(drop=True)
        )
        return self.write_raw(symbol, merged)

    def last_date(self, symbol: str):
        """Most recent stored bar date, for incremental fetches."""
        df = self.read_raw(symbol)
        if df is None or df.empty:
            return None
        return pd.Timestamp(df["date"].max()).date()

    def symbols(self) -> list[str]:
        base = self.root / "raw" / "market_data"
        if not base.exists():
            return []
        return sorted(
            p.name.split("=", 1)[1] for p in base.iterdir()
            if p.is_dir() and p.name.startswith("symbol=")
        )

    # -- features ---------------------------------------------------------

    def write_features(self, symbol: str, feature_version: str,
                       df: pd.DataFrame) -> Path:
        path = self.feature_path(symbol, feature_version)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return path

    def read_features(self, symbol: str, feature_version: str) -> pd.DataFrame | None:
        path = self.feature_path(symbol, feature_version)
        return pd.read_parquet(path) if path.exists() else None

    def read_all_features(self, feature_version: str) -> pd.DataFrame:
        """Every symbol's features, concatenated. The modelling entry point."""
        frames = [
            f for f in (
                self.read_features(s, feature_version) for s in self.symbols()
            ) if f is not None and not f.empty
        ]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values(
            ["date", "symbol"]
        ).reset_index(drop=True)
