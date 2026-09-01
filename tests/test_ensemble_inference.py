import numpy as np
import pandas as pd

from trademind.ensemble import (
    Calibrator,
    ProductionEnsemble,
    load_production_ensemble,
    save_production_ensemble,
)


class _FakeBaseModel:
    def __init__(self, value, version):
        self.value = value
        self.version = version

    def predict(self, frame):
        return np.full(len(frame), self.value, dtype=float)


class _FakeStack:
    def predict_proba(self, frame):
        probability = np.clip(frame.mean(axis=1).to_numpy(), 0.0, 1.0)
        return np.column_stack([1.0 - probability, probability])


def _bundle():
    calibrator = Calibrator("identity").fit([0.2, 0.8], [0, 1])
    return ProductionEnsemble(
        direction_models={
            "logistic": _FakeBaseModel(0.4, "logistic-v1"),
            "hgb": _FakeBaseModel(0.6, "hgb-v1"),
        },
        return_model=_FakeBaseModel(0.01, "return-v1"),
        stack_model=_FakeStack(),
        calibrator=calibrator,
        meta_feature_columns=["pred_logistic", "pred_hgb", "trend_regime"],
        context_columns=["trend_regime"],
        training_start="2020-01-01",
        training_end="2023-12-31",
        feature_version="f1",
    )


def test_production_ensemble_emits_prediction_job_schema():
    panel = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02"],
            "symbol": ["A.NS", "B.NS"],
            "trend_regime": [0.2, 0.8],
        }
    )
    signals = _bundle().predict(panel)

    assert list(signals) == [
        "date", "symbol", "raw", "calibrated", "expected_return"
    ]
    assert len(signals) == 2
    assert np.allclose(signals["calibrated"], signals["raw"])
    assert np.allclose(signals["expected_return"], 0.01)


def test_production_ensemble_round_trip(tmp_path):
    path = save_production_ensemble(_bundle(), tmp_path / "ensemble.joblib")
    loaded = load_production_ensemble(path)

    assert loaded.training_end == "2023-12-31"
    assert loaded.version.startswith("production_ensemble-")
