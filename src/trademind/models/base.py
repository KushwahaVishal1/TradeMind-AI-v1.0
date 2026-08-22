"""Base model interface.

## Preprocessing belongs inside the fold

The most common leak left in an otherwise careful pipeline::

    scaler = StandardScaler().fit(X)          # WRONG - sees validation
    X_scaled = scaler.transform(X)
    cross_validate(model, X_scaled, y, cv=purged_splitter)

The purged splitter is doing its job, and it does not matter: the scaler already
computed a mean and standard deviation over the whole panel, including the
validation window and the locked final-test rows. Every fold trains on features
standardised with future information.

The same applies to imputation (a median over the full sample), and to any
feature selection ranked on all the data before splitting.

Every model here is therefore an sklearn ``Pipeline`` whose preprocessing steps
are fitted by ``fit()`` on training data only. There is no separate
preprocessing pass over the panel, because there is no safe way to write one.

## Determinism

Every estimator gets an explicit ``random_state``. Two runs on identical inputs
must produce identical artifacts, or lineage means nothing and a model version
does not identify a model.
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

RANDOM_STATE = 42


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to reconstruct a model, and nothing else.

    The hash of this is the model's identity — two specs that hash the same
    produce the same model given the same data.
    """

    name: str
    task: str  # "direction" | "return"
    estimator: str  # registered factory key
    params: dict[str, Any] = field(default_factory=dict)
    feature_version: str = "f1"

    @property
    def spec_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def version(self, training_end: str | None = None) -> str:
        """Model version string: ``name-spechash[-trainend]``."""
        base = f"{self.name}-{self.spec_hash}"
        return f"{base}-{training_end}" if training_end else base


class BaseModel(ABC):
    """Common interface across direction and return models."""

    task: str = "abstract"

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.pipeline = None
        self.feature_names_: list[str] = []
        self.training_start_: str | None = None
        self.training_end_: str | None = None
        self.n_training_rows_: int = 0

    # -- construction -----------------------------------------------------

    @abstractmethod
    def _build_pipeline(self):
        """Return an unfitted sklearn Pipeline including preprocessing."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Direction models return P(up); return models return expected return."""

    # -- fitting ----------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.Series | None = None,
    ) -> BaseModel:
        """Fit on training data only. Preprocessing is fitted here, not before."""
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows, y has {len(y)}")
        if len(X) == 0:
            raise ValueError("Cannot fit on an empty training set")

        mask = y.notna().to_numpy()
        X, y = X.loc[mask], y.loc[mask]
        if len(X) == 0:
            raise ValueError("No labelled rows in the training set")

        self.feature_names_ = list(X.columns)
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X, y)

        self.n_training_rows_ = len(X)
        if dates is not None:
            d = pd.to_datetime(dates.loc[mask])
            self.training_start_ = str(d.min().date())
            self.training_end_ = str(d.max().date())
        return self

    def _check_fitted(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline is None:
            raise RuntimeError(f"{self.spec.name} is not fitted")
        missing = [c for c in self.feature_names_ if c not in X.columns]
        if missing:
            raise ValueError(f"Missing features at predict time: {missing[:5]}")
        # Reorder to the training column order; a silent reordering would
        # scramble every prediction without raising anything.
        return X[self.feature_names_]

    # -- introspection ----------------------------------------------------

    @property
    def version(self) -> str:
        return self.spec.version(self.training_end_)

    def feature_importance(self) -> pd.DataFrame | None:
        """Native importances, or None where the estimator has none.

        Returns None for ``HistGradientBoosting*``: scikit-learn deliberately
        does not expose ``feature_importances_`` on it, because the split-count
        heuristic used by other tree ensembles is biased toward high-cardinality
        features and it was judged too misleading to ship. Use
        ``permutation_importance`` for those models — slower, but it measures
        the quantity people actually mean.
        """
        if self.pipeline is None:
            return None
        est = self.pipeline.steps[-1][1]

        if hasattr(est, "feature_importances_"):
            values = est.feature_importances_
        elif hasattr(est, "coef_"):
            values = np.abs(np.ravel(est.coef_))
        else:
            return None

        if len(values) != len(self.feature_names_):
            return None
        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names_,
                    "importance": values,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def permutation_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 5,
        scoring: str | None = None,
    ) -> pd.DataFrame:
        """Drop in score when each feature is shuffled.

        The general answer, and the only one available for
        ``HistGradientBoosting``. Measure it on *validation* data: on training
        data it reports what the model memorised, not what generalises.

        Correlated features share credit and each will look individually
        unimportant, because shuffling one leaves its twin intact. This panel
        has several such groups (return horizons, volatility estimators), so
        read the results as families rather than a strict ranking.
        """
        from sklearn.inspection import permutation_importance as sk_perm

        if self.pipeline is None:
            raise RuntimeError(f"{self.spec.name} is not fitted")

        X = self._check_fitted(X)
        mask = y.notna().to_numpy()
        result = sk_perm(
            self.pipeline,
            X.loc[mask],
            y.loc[mask],
            n_repeats=n_repeats,
            random_state=RANDOM_STATE,
            scoring=scoring,
        )
        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names_,
                    "importance": result.importances_mean,
                    "importance_std": result.importances_std,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def metadata(self) -> dict:
        return {
            "name": self.spec.name,
            "task": self.task,
            "estimator": self.spec.estimator,
            "params": self.spec.params,
            "version": self.version,
            "spec_hash": self.spec.spec_hash,
            "feature_version": self.spec.feature_version,
            "n_features": len(self.feature_names_),
            "n_training_rows": self.n_training_rows_,
            "training_start": self.training_start_,
            "training_end": self.training_end_,
        }


def make_imputer():
    """Median imputation, fitted per fold.

    Used only by estimators that cannot handle NaN. Gradient-boosted trees
    route missing values natively and are left alone, which is strictly better
    than imputing — missingness in market data is often informative.
    """
    from sklearn.impute import SimpleImputer

    return SimpleImputer(strategy="median")


def select_features(panel: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    """Extract the feature matrix, failing loudly on absent columns."""
    missing = [c for c in feature_cols if c not in panel.columns]
    if missing:
        raise ValueError(f"Features absent from the panel: {missing}")
    return panel[list(feature_cols)]
