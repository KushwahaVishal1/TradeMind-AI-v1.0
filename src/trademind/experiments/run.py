"""Experiment records.

An experiment is only useful later if it can be reproduced, and reproducing it
needs more than the hyperparameters. It needs the code (git commit), the
configuration (hash), the data (dataset and feature versions), the randomness
(seed), and the environment (package versions). Any one of those missing turns
"our best model scored 0.55" into a claim nobody can check.

``ExperimentRun.reproducibility_gaps()`` lists what is absent, so an
under-specified experiment is visibly under-specified rather than silently so.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _git_dirty() -> bool:
    """Are there uncommitted changes?

    A commit hash recorded against a dirty tree points at code that was never
    what ran. Worth knowing.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        return bool(out.stdout.strip())
    except Exception:
        return False


def package_versions() -> dict[str, str]:
    import numpy as np
    import pandas as pd
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }


@dataclass
class ExperimentRun:
    """One experiment, with everything needed to reproduce it."""

    name: str
    task: str                                  # direction | return | ensemble
    model_type: str = ""
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    # reproducibility
    git_commit: str | None = field(default_factory=_git_commit)
    git_dirty: bool = field(default_factory=_git_dirty)
    config_hash: str | None = None
    dataset_version: str | None = None
    feature_version: str | None = None
    random_seed: int | None = 42
    packages: dict[str, str] = field(default_factory=package_versions)

    # setup
    validation_strategy: str = "purged_walk_forward"
    training_start: str | None = None
    training_end: str | None = None
    n_training_rows: int = 0
    n_features: int = 0

    # results
    metrics: dict[str, float] = field(default_factory=dict)
    primary_metric: str = "roc_auc"
    n_prior_experiments: int = 0
    notes: str = ""

    experiment_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    @property
    def primary_value(self) -> float | None:
        return self.metrics.get(self.primary_metric)

    @property
    def fingerprint(self) -> str:
        """Hash of the inputs. Two runs with the same fingerprint should agree."""
        payload = json.dumps({
            "git_commit": self.git_commit,
            "config_hash": self.config_hash,
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "model_type": self.model_type,
            "hyperparameters": self.hyperparameters,
            "random_seed": self.random_seed,
            "validation_strategy": self.validation_strategy,
        }, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def reproducibility_gaps(self) -> list[str]:
        """What is missing that would prevent someone re-running this."""
        gaps = []
        if not self.git_commit:
            gaps.append("no git commit recorded")
        if self.git_dirty:
            gaps.append(
                "working tree was dirty; the recorded commit is not what ran"
            )
        if not self.config_hash:
            gaps.append("no config hash")
        if not self.dataset_version:
            gaps.append("no dataset version")
        if not self.feature_version:
            gaps.append("no feature version")
        if self.random_seed is None:
            gaps.append("no random seed; results will not be identical")
        return gaps

    @property
    def reproducible(self) -> bool:
        return not self.reproducibility_gaps()

    def to_row(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "task": self.task,
            "created_at": self.created_at,
            "git_commit": self.git_commit,
            "config_hash": self.config_hash,
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "random_seed": self.random_seed,
            "python_version": self.packages.get("python"),
            "package_versions": json.dumps(self.packages),
            "model_type": self.model_type,
            "hyperparameters": json.dumps(self.hyperparameters, default=str),
            "validation_strategy": self.validation_strategy,
            "training_start": self.training_start,
            "training_end": self.training_end,
            "n_training_rows": self.n_training_rows,
            "n_features": self.n_features,
            "metrics": json.dumps(self.metrics),
            "primary_metric": self.primary_metric,
            "primary_value": self.primary_value,
            "n_prior_experiments": self.n_prior_experiments,
            "notes": self.notes,
        }

    @classmethod
    def from_row(cls, row) -> ExperimentRun:
        data = dict(row)
        run = cls(
            name=data["name"],
            task=data["task"],
            model_type=data.get("model_type") or "",
            hyperparameters=json.loads(data.get("hyperparameters") or "{}"),
            git_commit=data.get("git_commit"),
            config_hash=data.get("config_hash"),
            dataset_version=data.get("dataset_version"),
            feature_version=data.get("feature_version"),
            random_seed=data.get("random_seed"),
            packages=json.loads(data.get("package_versions") or "{}"),
            validation_strategy=data.get("validation_strategy") or "",
            training_start=data.get("training_start"),
            training_end=data.get("training_end"),
            n_training_rows=data.get("n_training_rows") or 0,
            n_features=data.get("n_features") or 0,
            metrics=json.loads(data.get("metrics") or "{}"),
            primary_metric=data.get("primary_metric") or "roc_auc",
            n_prior_experiments=data.get("n_prior_experiments") or 0,
            notes=data.get("notes") or "",
        )
        run.experiment_id = data["experiment_id"]
        run.created_at = data["created_at"]
        run.git_dirty = False
        return run

    def render(self) -> str:
        lines = [
            f"{self.name} [{self.experiment_id}] {self.task}/{self.model_type}",
            f"  {self.primary_metric} = "
            f"{self.primary_value if self.primary_value is not None else float('nan'):.4f}",
            f"  trained {self.training_start}..{self.training_end} "
            f"({self.n_training_rows:,} rows, {self.n_features} features)",
            f"  commit {self.git_commit or 'n/a'} | config {self.config_hash or 'n/a'}",
        ]
        gaps = self.reproducibility_gaps()
        if gaps:
            lines.append(f"  NOT REPRODUCIBLE: {'; '.join(gaps)}")
        return "\n".join(lines)
