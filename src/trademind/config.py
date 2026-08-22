"""Configuration loading.

One rule: nothing in this project reads a magic number that isn't in
``config/``. Every run records a hash of the configuration that produced it,
so a stored prediction can always be traced back to the exact settings used.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when configuration is missing, malformed, or self-contradictory."""


_MISSING = object()

# Sessions between the decision (close of t) and the entry fill (open of t+1).
# See features/labels.py for the timing contract this encodes.
EXECUTION_OFFSET_SESSIONS = 1


def _project_root() -> Path:
    # src/trademind/config.py -> src/trademind -> src -> repo root
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    """Parsed configuration plus the hash of the raw content it came from."""

    raw: dict[str, Any]
    config_hash: str
    root: Path
    universe: list[str] = field(default_factory=list)
    benchmark: str = ""

    # -- typed accessors -------------------------------------------------

    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        """Fetch ``a.b.c`` from the config tree.

        Raises if the key is absent and no default was supplied, rather than
        returning None and letting a null propagate into a model.
        """
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is _MISSING:
                    raise ConfigError(f"Missing config key: {dotted}")
                return default
            node = node[part]
        return node

    @property
    def data_root(self) -> Path:
        return self.root / self.get("data.root")

    @property
    def history_start(self) -> date:
        return date.fromisoformat(self.get("data.history_start"))

    @property
    def final_test_start(self) -> date:
        """First date of the locked final-test window.

        Any code that fits, tunes, or selects on data at or after this date is
        a bug. Phase 3 enforces it; this property is the single source of truth.
        """
        return date.fromisoformat(self.get("data.final_test_start"))

    @property
    def feature_version(self) -> str:
        return self.get("features.version")

    @property
    def decision_version(self) -> str:
        return self.get("decision.version")

    @property
    def threshold_version(self) -> str:
        return self.get("decision.threshold_version")


def _hash_content(*texts: str) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def load_config(
    config_path: str | Path | None = None,
    universe_path: str | Path | None = None,
) -> Config:
    """Load ``config.yaml`` and ``universe.yaml`` into a frozen Config."""
    root = _project_root()
    config_path = Path(config_path) if config_path else root / "config" / "config.yaml"
    universe_path = (
        Path(universe_path) if universe_path else root / "config" / "universe.yaml"
    )

    for p in (config_path, universe_path):
        if not p.exists():
            raise ConfigError(f"Config file not found: {p}")

    config_text = config_path.read_text(encoding="utf-8")
    universe_text = universe_path.read_text(encoding="utf-8")

    raw = yaml.safe_load(config_text) or {}
    uni = yaml.safe_load(universe_text) or {}

    symbols = uni.get("symbols") or []
    if not symbols:
        raise ConfigError(f"Universe {universe_path} defines no symbols")
    if len(set(symbols)) != len(symbols):
        raise ConfigError("Universe contains duplicate symbols")

    cfg = Config(
        raw=raw,
        config_hash=_hash_content(config_text, universe_text),
        root=root,
        universe=list(symbols),
        benchmark=uni.get("benchmark", ""),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    """Fail loudly at load time rather than deep inside a training loop."""
    if cfg.history_start >= cfg.final_test_start:
        raise ConfigError(
            "data.history_start must fall before data.final_test_start; "
            "otherwise there is no development data."
        )
    if cfg.get("features.target_horizon_days") < 1:
        raise ConfigError("features.target_horizon_days must be >= 1")
    # Purge must cover the label horizon PLUS the execution offset. Under this
    # project's timing contract the label for t spans open[t+1]..open[t+1+h], so
    # a row at t encodes information through t+1+h. Sizing the purge to the
    # horizon alone leaves exactly one leaking row at every fold boundary.
    horizon = cfg.get("features.target_horizon_days")
    min_purge = horizon + EXECUTION_OFFSET_SESSIONS
    if cfg.get("features.purge_days") < min_purge:
        raise ConfigError(
            f"features.purge_days must be >= {min_purge} for "
            f"target_horizon_days={horizon} (horizon + execution offset of "
            f"{EXECUTION_OFFSET_SESSIONS}). Sizing the purge to the horizon "
            "alone leaves a leaking row at every fold boundary."
        )
    if cfg.get("features.embargo_days") < 0:
        raise ConfigError("features.embargo_days must be >= 0")


def config_fingerprint(cfg: Config) -> str:
    """Stable JSON rendering of config, for debugging hash mismatches."""
    return json.dumps(cfg.raw, sort_keys=True, indent=2)
