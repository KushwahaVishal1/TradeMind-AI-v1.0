"""Model artifact persistence.

Saves the fitted pipeline alongside the metadata needed to reproduce it: the
spec, the feature list in training order, the training window, library versions,
and the environment. A model file without that context is a black box — you can
score with it, but you cannot say what produced it, which is the whole point of
lineage.

Loading checks that the environment still matches and warns when it does not.
A pickle written under a different scikit-learn version may load and behave
differently, silently.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .base import BaseModel

log = logging.getLogger(__name__)


def _environment() -> dict:
    import sklearn

    return {
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": sys.platform,
    }


@dataclass
class ModelArtifact:
    """A saved model and everything needed to interpret it."""

    model: BaseModel
    metadata: dict

    @property
    def version(self) -> str:
        return self.metadata["version"]


class ModelRegistry:
    """Filesystem-backed store of fitted models.

    ::

        models/
          direction_hgb-a1b2c3d4e5f6-2022-12-28/
            model.joblib
            metadata.json
            feature_importance.csv

    The directory name is the model version, so a prediction row carrying
    ``model_version`` points at exactly one directory. Phase 10 layers the
    promotion lifecycle on top; this is the storage underneath it.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, version: str) -> Path:
        return self.root / version

    def save(self, model: BaseModel, extra: dict | None = None) -> Path:
        if model.pipeline is None:
            raise RuntimeError("Refusing to save an unfitted model")

        version = model.version
        path = self.path_for(version)
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(model.pipeline, path / "model.joblib")

        metadata = {
            **model.metadata(),
            "saved_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "environment": _environment(),
            "feature_names": model.feature_names_,
            **(extra or {}),
        }
        (path / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )

        importance = model.feature_importance()
        if importance is not None:
            importance.to_csv(path / "feature_importance.csv", index=False)
        else:
            # HistGradientBoosting has no native importances by design.
            # Record why, rather than leaving a silently absent file.
            (path / "feature_importance.txt").write_text(
                f"{model.spec.estimator} exposes no native feature importances. "
                "Use BaseModel.permutation_importance(X_val, y_val).\n",
                encoding="utf-8",
            )

        log.info("Saved %s to %s", version, path)
        return path

    def load(self, version: str) -> ModelArtifact:
        path = self.path_for(version)
        if not path.exists():
            raise FileNotFoundError(f"No model registered under {version}")

        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        pipeline = joblib.load(path / "model.joblib")

        self._warn_on_environment_drift(metadata.get("environment", {}), version)

        model = _reconstruct(metadata)
        model.pipeline = pipeline
        model.feature_names_ = metadata["feature_names"]
        model.training_start_ = metadata.get("training_start")
        model.training_end_ = metadata.get("training_end")
        model.n_training_rows_ = metadata.get("n_training_rows", 0)

        return ModelArtifact(model, metadata)

    @staticmethod
    def _warn_on_environment_drift(saved: dict, version: str) -> None:
        current = _environment()
        for key in ("sklearn", "numpy"):
            if saved.get(key) and saved[key] != current[key]:
                log.warning(
                    "%s was saved under %s %s but is loading under %s. "
                    "Behaviour may differ silently.",
                    version,
                    key,
                    saved[key],
                    current[key],
                )

    def list_versions(self) -> list[str]:
        return sorted(
            p.name
            for p in self.root.iterdir()
            if p.is_dir() and (p / "metadata.json").exists()
        )

    def summary(self) -> pd.DataFrame:
        rows = []
        for version in self.list_versions():
            meta = json.loads(
                (self.path_for(version) / "metadata.json").read_text(encoding="utf-8")
            )
            rows.append(
                {
                    "version": version,
                    "task": meta.get("task"),
                    "estimator": meta.get("estimator"),
                    "n_features": meta.get("n_features"),
                    "training_end": meta.get("training_end"),
                    "saved_at": meta.get("saved_at"),
                }
            )
        return pd.DataFrame(rows)


def _reconstruct(metadata: dict) -> BaseModel:
    """Rebuild the right model wrapper from saved metadata."""
    from .base import ModelSpec
    from .direction import DirectionModel
    from .return_model import ReturnModel

    spec = ModelSpec(
        name=metadata["name"],
        task=metadata["task"],
        estimator=metadata["estimator"],
        params=metadata.get("params", {}),
        feature_version=metadata.get("feature_version", "f1"),
    )
    cls = DirectionModel if metadata["task"] == "direction" else ReturnModel
    return cls(spec)
