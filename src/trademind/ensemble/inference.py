"""Persisted production ensemble and leakage-safe live inference."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from ..models.base import BaseModel
from .calibrator import Calibrator


@dataclass
class ProductionEnsemble:
    direction_models: dict[str, BaseModel]
    return_model: BaseModel
    stack_model: object
    calibrator: Calibrator
    meta_feature_columns: list[str]
    context_columns: list[str]
    training_start: str
    training_end: str
    feature_version: str

    @property
    def version(self) -> str:
        payload = {
            "direction": {k: v.version for k, v in self.direction_models.items()},
            "return": self.return_model.version,
            "training_end": self.training_end,
            "feature_version": self.feature_version,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:12]
        return f"production_ensemble-{digest}-{self.training_end}"

    def predict(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Return the signal schema consumed by ``PredictionJob``."""
        if panel.empty:
            return pd.DataFrame(
                columns=["date", "symbol", "raw", "calibrated", "expected_return"]
            )

        meta = panel[["date", "symbol"]].reset_index(drop=True).copy()
        for name, model in self.direction_models.items():
            meta[f"pred_{name}"] = model.predict(panel).astype(float)
        for col in self.context_columns:
            if col not in panel:
                raise ValueError(f"Live panel lacks ensemble context column {col!r}")
            meta[col] = panel[col].to_numpy()

        missing = [c for c in self.meta_feature_columns if c not in meta]
        if missing:
            raise ValueError(f"Live meta-features are missing: {missing}")
        raw = self.stack_model.predict_proba(meta[self.meta_feature_columns])[:, 1]

        return pd.DataFrame(
            {
                "date": meta["date"].to_numpy(),
                "symbol": meta["symbol"].to_numpy(),
                "raw": raw,
                "calibrated": self.calibrator.predict(raw),
                "expected_return": self.return_model.predict(panel).astype(float),
            }
        )


def save_production_ensemble(bundle: ProductionEnsemble, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_production_ensemble(path: str | Path) -> ProductionEnsemble:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No production ensemble found at {path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, ProductionEnsemble):
        raise TypeError(f"Unexpected production ensemble artifact: {type(bundle)!r}")
    return bundle
