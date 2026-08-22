"""Artifact storage for experiments.

Small artifacts -- OOF predictions, metrics, feature importances, calibration
tables -- stored next to the experiment that produced them. Kept as CSV and JSON
rather than pickle: an artifact you cannot open in five years without the exact
library version is not an artifact, it is a liability.

Model binaries stay in ``ModelRegistry`` (Phase 4), which handles the pickle
concerns and environment checks. This is for the evidence, not the model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


class ArtifactStore:
    """Filesystem store keyed by experiment id."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, experiment_id: str) -> Path:
        return self.root / experiment_id

    def save_frame(self, experiment_id: str, name: str, frame: pd.DataFrame) -> Path:
        directory = self.path_for(experiment_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.csv"
        frame.to_csv(path, index=False)
        return path

    def save_json(self, experiment_id: str, name: str, payload: dict) -> Path:
        directory = self.path_for(experiment_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def load_frame(self, experiment_id: str, name: str) -> pd.DataFrame | None:
        path = self.path_for(experiment_id) / f"{name}.csv"
        return pd.read_csv(path) if path.exists() else None

    def load_json(self, experiment_id: str, name: str) -> dict | None:
        path = self.path_for(experiment_id) / f"{name}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def list_artifacts(self, experiment_id: str) -> list[str]:
        directory = self.path_for(experiment_id)
        if not directory.exists():
            return []
        return sorted(p.name for p in directory.iterdir() if p.is_file())
