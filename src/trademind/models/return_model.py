"""Return models: expected value of the tradeable forward return.

Regression on daily returns is a low-ceiling problem. R-squared will be
negative -- the mean beats any individual forecast in squared error -- and that
is the expected outcome, not a failure. The useful output is the *ordering* of
expected returns across the cross-section, measured by the information
coefficient.

The magnitude still matters downstream: Phase 6 applies a
``min_expected_return`` gate so a marginally positive forecast does not fire a
trade whose costs exceed its edge. A ranking alone could not support that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import RANDOM_STATE, BaseModel, ModelSpec, make_imputer


class ReturnModel(BaseModel):
    """Regressor over the tradeable forward return."""

    task = "return"

    def _build_pipeline(self) -> Pipeline:
        kind = self.spec.estimator
        params = dict(self.spec.params)

        if kind == "ridge":
            # Strong default regularisation. Features here are heavily
            # collinear -- several return horizons, several volatility
            # estimators -- and unregularised coefficients would be unstable
            # across folds even where predictions were not.
            return Pipeline(
                [
                    ("impute", make_imputer()),
                    ("scale", StandardScaler()),
                    (
                        "model",
                        Ridge(
                            alpha=params.get("alpha", 10.0),
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            )

        if kind == "hgb":
            return Pipeline(
                [
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            max_depth=params.get("max_depth", 3),
                            max_iter=params.get("max_iter", 200),
                            learning_rate=params.get("learning_rate", 0.03),
                            min_samples_leaf=params.get("min_samples_leaf", 200),
                            l2_regularization=params.get("l2_regularization", 1.0),
                            max_leaf_nodes=params.get("max_leaf_nodes", 15),
                            loss=params.get("loss", "squared_error"),
                            early_stopping=False,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            )

        raise ValueError(f"Unknown return estimator: {kind}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = self._check_fitted(X)
        return np.asarray(self.pipeline.predict(X)).ravel()


def ridge_return(feature_version: str = "f1", **params) -> ReturnModel:
    return ReturnModel(
        ModelSpec(
            name="return_ridge",
            task="return",
            estimator="ridge",
            params=params,
            feature_version=feature_version,
        )
    )


def hgb_return(feature_version: str = "f1", **params) -> ReturnModel:
    return ReturnModel(
        ModelSpec(
            name="return_hgb",
            task="return",
            estimator="hgb",
            params=params,
            feature_version=feature_version,
        )
    )
