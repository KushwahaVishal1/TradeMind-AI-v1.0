"""Structured logging with a run id attached to every record."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FMT = "%(asctime)s | %(levelname)-7s | %(run_id)s | %(name)s | %(message)s"


class _RunIdFilter(logging.Filter):
    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = getattr(record, "run_id", self.run_id)
        return True


def setup_logging(level: str = "INFO", run_id: str = "-",
                  log_dir: str | Path | None = None) -> None:
    """Configure root logging. Idempotent — safe to call more than once."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    fmt = logging.Formatter(_FMT)
    flt = _RunIdFilter(run_id)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    stream.addFilter(flt)
    root.addHandler(stream)

    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fileh = logging.FileHandler(log_dir / "trademind.log", encoding="utf-8")
        fileh.setFormatter(fmt)
        fileh.addFilter(flt)
        root.addHandler(fileh)
