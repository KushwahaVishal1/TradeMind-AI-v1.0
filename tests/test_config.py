"""Configuration loading and its fail-fast guards."""

from __future__ import annotations

import pytest
import yaml

from trademind.config import ConfigError, load_config

GOOD_UNIVERSE = {"benchmark": "^NSEI", "symbols": ["A.NS", "B.NS"]}

GOOD_CONFIG = {
    "data": {
        "root": "data",
        "history_start": "2015-01-01",
        "final_test_start": "2024-01-01",
    },
    "features": {
        "version": "f1",
        "target_horizon_days": 1,
        "purge_days": 5,
        "embargo_days": 3,
    },
    "decision": {"version": "d1", "threshold_version": "t1"},
}


def write(tmp_path, config=None, universe=None):
    cfg_p = tmp_path / "config.yaml"
    uni_p = tmp_path / "universe.yaml"
    cfg_p.write_text(yaml.safe_dump(config or GOOD_CONFIG))
    uni_p.write_text(yaml.safe_dump(universe or GOOD_UNIVERSE))
    return cfg_p, uni_p


def test_loads_and_exposes_universe(tmp_path):
    cfg = load_config(*write(tmp_path))
    assert cfg.universe == ["A.NS", "B.NS"]
    assert cfg.benchmark == "^NSEI"
    assert cfg.feature_version == "f1"


def test_config_hash_is_stable(tmp_path):
    paths = write(tmp_path)
    assert load_config(*paths).config_hash == load_config(*paths).config_hash


def test_config_hash_changes_with_content(tmp_path):
    a = load_config(*write(tmp_path))

    changed = {**GOOD_CONFIG, "features": {**GOOD_CONFIG["features"], "version": "f2"}}
    b = load_config(*write(tmp_path, config=changed))

    assert a.config_hash != b.config_hash


def test_missing_key_raises(tmp_path):
    cfg = load_config(*write(tmp_path))
    with pytest.raises(ConfigError, match="Missing config key"):
        cfg.get("data.nonexistent")


def test_default_returned_when_absent(tmp_path):
    cfg = load_config(*write(tmp_path))
    assert cfg.get("data.nonexistent", 42) == 42


def test_empty_universe_rejected(tmp_path):
    with pytest.raises(ConfigError, match="no symbols"):
        load_config(*write(tmp_path, universe={"symbols": []}))


def test_duplicate_symbols_rejected(tmp_path):
    with pytest.raises(ConfigError, match="duplicate"):
        load_config(*write(tmp_path, universe={"symbols": ["A.NS", "A.NS"]}))


def test_final_test_before_history_start_rejected(tmp_path):
    bad = {**GOOD_CONFIG, "data": {**GOOD_CONFIG["data"], "final_test_start": "2014-01-01"}}
    with pytest.raises(ConfigError, match="history_start"):
        load_config(*write(tmp_path, config=bad))


def test_purge_shorter_than_horizon_rejected(tmp_path):
    """Purge must cover the label window or train/val boundaries leak."""
    bad = {
        **GOOD_CONFIG,
        "features": {**GOOD_CONFIG["features"], "target_horizon_days": 10, "purge_days": 2},
    }
    with pytest.raises(ConfigError, match="purge_days"):
        load_config(*write(tmp_path, config=bad))


def test_purge_must_cover_horizon_plus_execution_offset(tmp_path):
    """purge == horizon is not enough; the execution offset adds one session."""
    bad = {
        **GOOD_CONFIG,
        "features": {**GOOD_CONFIG["features"], "target_horizon_days": 5, "purge_days": 5},
    }
    with pytest.raises(ConfigError, match="execution offset"):
        load_config(*write(tmp_path, config=bad))


def test_purge_at_the_tightened_minimum_is_accepted(tmp_path):
    ok = {
        **GOOD_CONFIG,
        "features": {**GOOD_CONFIG["features"], "target_horizon_days": 5, "purge_days": 6},
    }
    assert load_config(*write(tmp_path, config=ok)).get("features.purge_days") == 6
