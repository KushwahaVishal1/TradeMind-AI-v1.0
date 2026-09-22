import json
from types import SimpleNamespace

import pandas as pd
import pytest

from trademind.reporting.forecast_outcomes import (
    forecast_history,
    resolve_forecasts,
    score_outcome,
)


def setup_forecast(tmp_path):
    folder = tmp_path / "intraday" / "evening_forecasts"
    folder.mkdir(parents=True)
    path = folder / "NSEI_2026-09-22.json"
    record = {"index": "NIFTY 50", "symbol": "^NSEI", "direction": "DOWN",
              "expected_return": -.001, "probability_up": .45,
              "forecast_session": "2026-09-22", "as_of": "2026-09-21",
              "created_at": "2026-09-21T19:00:00+05:30"}
    path.write_text(json.dumps(record))
    return SimpleNamespace(data_root=tmp_path), path


def test_correct_wrong_flat_and_error():
    forecast = {"direction": "DOWN", "expected_return": -.01}
    assert score_outcome(forecast, 100, 99)["status"] == "CORRECT"
    assert score_outcome(forecast, 100, 101)["status"] == "WRONG"
    assert score_outcome(forecast, 100, 100)["status"] == "FLAT"
    assert score_outcome(forecast, 100, 101)["return_error"] == pytest.approx(.02)
    with pytest.raises(ValueError):
        score_outcome(forecast, 0, 100)


def test_resolution_waits_preserves_forecast_and_is_idempotent(tmp_path):
    cfg, path = setup_forecast(tmp_path)
    original = path.read_bytes()
    calls = []

    def fetch(*args):
        calls.append(args)
        return pd.DataFrame({"Open": [100.], "Close": [99.]},
                            index=pd.DatetimeIndex(["2026-09-22"], tz="Asia/Kolkata"))

    assert resolve_forecasts(cfg, "2026-09-22T14:00:00+05:30", fetch) == (0, {})
    assert not calls
    assert forecast_history(cfg).Result.tolist() == ["PENDING"]
    assert resolve_forecasts(cfg, "2026-09-22T18:00:00+05:30", fetch) == (1, {})
    assert forecast_history(cfg).Result.tolist() == ["CORRECT"]
    assert resolve_forecasts(cfg, "2026-09-23T18:00:00+05:30", fetch) == (0, {})
    assert len(calls) == 1
    assert path.read_bytes() == original
    path.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="changed"):
        forecast_history(cfg)


def test_missing_or_wrong_session_does_not_score(tmp_path):
    cfg, _ = setup_forecast(tmp_path)
    def fetch(*args):
        return pd.DataFrame({"Open": [100.], "Close": [99.]},
                            index=pd.DatetimeIndex(["2026-09-21"], tz="Asia/Kolkata"))
    count, errors = resolve_forecasts(cfg, "2026-09-22T19:00:00+05:30", fetch)
    assert count == 0 and errors
    assert forecast_history(cfg).Result.tolist() == ["PENDING"]


def test_retrospective_forecast_is_ineligible(tmp_path):
    cfg, path = setup_forecast(tmp_path)
    row = json.loads(path.read_text())
    row["created_at"] = "2026-09-22T19:00:00+05:30"
    path.write_text(json.dumps(row))
    count, errors = resolve_forecasts(cfg, "2026-09-23T19:00:00+05:30",
                                     lambda *args: pytest.fail("Must not fetch"))
    assert count == 0 and errors
    assert forecast_history(cfg).Result.tolist() == ["INELIGIBLE"]
