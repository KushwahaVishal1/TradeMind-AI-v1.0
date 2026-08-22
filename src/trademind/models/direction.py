"""Direction models: P(the tradeable forward return is positive).

Two estimators, deliberately different in kind:

``logistic``
    L2 logistic regression on standardised features. Linear, low variance, hard
    to overfit, and it produces reasonably calibrated probabilities out of the
    box. It is the honest baseline -- if the gradient-boosted model cannot beat
    it, the extra capacity is fitting noise.

``hgb``
    HistGradientBoostingClassifier. Captures interactions and non-linearity,
    handles NaN natively, and is fast. Its probabilities are typically
    overconfident near the extremes, which Phase 5 corrects; that is a known
    property of boosted trees, not a defect to tune away here.

Both are wrapped in a Pipeline so preprocessing is fitted per fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import RANDOM_STATE, BaseModel, ModelSpec, make_imputer


class DirectionModel(BaseModel):
    """Binary classifier over the direction label."""

    task = "direction"

    def _build_pipeline(self) -> Pipeline:
        kind = self.spec.estimator
        params = dict(self.spec.params)

        if kind == "logistic":
            return Pipeline(
                [
                    ("impute", make_imputer()),
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=params.get("C", 1.0),
                            # L2 is the default; naming it explicitly triggers a
                            # deprecation warning on scikit-learn >= 1.8.
                            solver="lbfgs",
                            max_iter=params.get("max_iter", 2000),
                            class_weight=params.get("class_weight"),
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            )

        if kind == "hgb":
            # Conservative defaults. Daily equity direction has a tiny
            # signal-to-noise ratio, so shallow trees, heavy regularisation and
            # large leaves are the right prior. Deep trees memorise noise and
            # the validation curve will not always reveal it.
            return Pipeline(
                [
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            max_depth=params.get("max_depth", 3),
                            max_iter=params.get("max_iter", 200),
                            learning_rate=params.get("learning_rate", 0.03),
                            min_samples_leaf=params.get("min_samples_leaf", 200),
                            l2_regularization=params.get("l2_regularization", 1.0),
                            max_leaf_nodes=params.get("max_leaf_nodes", 15),
                            early_stopping=False,  # the outer purged split judges this
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            )

        raise ValueError(f"Unknown direction estimator: {kind}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """P(up). Always the positive-class column, never a hard label."""
        X = self._check_fitted(X)
        return self.pipeline.predict_proba(X)[:, 1]

    def predict_label(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict(X) >= threshold).astype(int)


def logistic_direction(feature_version: str = "f1", **params) -> DirectionModel:
    return DirectionModel(
        ModelSpec(
            name="direction_logistic",
            task="direction",
            estimator="logistic",
            params=params,
            feature_version=feature_version,
        )
    )


def hgb_direction(feature_version: str = "f1", **params) -> DirectionModel:
    return DirectionModel(
        ModelSpec(
            name="direction_hgb",
            task="direction",
            estimator="hgb",
            params=params,
            feature_version=feature_version,
        )
    )
