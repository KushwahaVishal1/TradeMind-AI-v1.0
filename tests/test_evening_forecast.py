import numpy as np
import pandas as pd
import pytest

from trademind.ingestion.evening_forecast import fit_forecast, make_features


def history():
    rng = np.random.default_rng(42)
    opening = 100 * np.exp(np.cumsum(rng.normal(0, .01, 650)))
    closing = opening * (1 + rng.normal(0, .005, 650))
    return pd.DataFrame({
        "Open": opening, "Close": closing,
        "High": np.maximum(opening, closing) * 1.002,
        "Low": np.minimum(opening, closing) * .998,
    }, index=pd.bdate_range("2021-01-01", periods=650, tz="Asia/Kolkata"))


SETTINGS = {"minimum_samples": 500, "validation_splits": 5,
            "validation_sessions": 60, "logistic_c": .1, "ridge_alpha": 100.}


def test_target_is_next_open_to_close_and_latest_is_unlabelled():
    h = history()
    _, y = make_features(h)
    assert y.iloc[0] == pytest.approx(h.Close.iloc[1] / h.Open.iloc[1] - 1)
    assert pd.isna(y.iloc[-1])


def test_future_changes_do_not_change_past_features():
    h = history()
    original, _ = make_features(h)
    h.iloc[500:] *= 2
    changed, _ = make_features(h)
    pd.testing.assert_frame_equal(original.iloc[:500], changed.iloc[:500])


def test_validation_and_final_training_exclude_unknown_target():
    h = history()
    result = fit_forecast(h, SETTINGS)
    assert result["validation_sessions"] == 300
    assert result["training_last_feature_date"] == str(h.index[-2].date())
    assert result["label_data_through"] == str(h.index[-1].date())
    assert 0 <= result["probability_up"] <= 1
    assert np.isfinite(result["expected_return"])
    with pytest.raises(ValueError, match="Insufficient"):
        fit_forecast(h.iloc[:200], SETTINGS)


def test_morning_generation_is_refused(tmp_path):
    from types import SimpleNamespace

    import yaml

    from trademind.ingestion.evening_forecast import generate_evening_forecasts

    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "evening_forecast.yaml").write_text(
        yaml.safe_dump({**SETTINGS, "ready_hour_ist": 18})
    )
    cfg = SimpleNamespace(root=tmp_path, data_root=tmp_path / "data")
    with pytest.raises(ValueError, match="after 18:00"):
        generate_evening_forecasts(cfg, "2026-09-21T10:00:00+05:30")
