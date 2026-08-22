"""Out-of-fold prediction generation, and the baselines everything is judged against.

## Why OOF is mandatory

Phase 5 trains a meta-model on the base models' outputs. If those outputs came
from models that had seen the rows they are predicting, the meta-model learns
that the base predictions are far more reliable than they will ever be at
inference time, and it weights them accordingly. The stack then collapses the
moment it meets genuinely unseen data.

An out-of-fold prediction is one made by a model fitted without that row. Every
row in the OOF frame satisfies that, and the purged splitter additionally
guarantees the fitting data did not overlap the row's label window.

## Baselines

``MajorityBaseline`` and ``MeanBaseline`` exist so no model result is ever read
in isolation. They are trivial by design; the point is that on this task they
are hard to beat, and a model that does not beat them has demonstrated nothing
whatever its AUC looks like.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..validation.purged_split import PurgedWalkForwardSplit, assert_fold_is_clean
from .base import BaseModel
from .metrics import classification_metrics, flag_suspicious, regression_metrics, summarise

log = logging.getLogger(__name__)


# =====================================================================
# Baselines
# =====================================================================


class MajorityBaseline:
    """Predicts the training base rate for every row.

    Constant. Its accuracy is the number any classifier must beat before its
    result means anything.
    """

    task = "direction"
    name = "baseline_majority"

    def __init__(self) -> None:
        self.rate_ = 0.5

    def fit(self, X, y, dates=None):
        self.rate_ = float(pd.Series(y).dropna().mean())
        return self

    def predict(self, X) -> np.ndarray:
        return np.full(len(X), self.rate_)


class MeanBaseline:
    """Predicts the training mean return for every row.

    Whatever R-squared the return model achieves is measured against exactly
    this. Beating it in squared error on daily returns is genuinely hard.
    """

    task = "return"
    name = "baseline_mean"

    def __init__(self) -> None:
        self.mean_ = 0.0

    def fit(self, X, y, dates=None):
        self.mean_ = float(pd.Series(y).dropna().mean())
        return self

    def predict(self, X) -> np.ndarray:
        return np.full(len(X), self.mean_)


class MomentumBaseline:
    """Yesterday's direction, repeated. A naive persistence rule.

    Worth including because it is what a person guesses with no model, and
    because short-horizon reversal means it is often slightly *worse* than
    chance — which is itself informative about the data.
    """

    task = "direction"
    name = "baseline_momentum"

    def __init__(self, feature: str = "return_1d") -> None:
        self.feature = feature

    def fit(self, X, y, dates=None):
        if self.feature not in X.columns:
            raise ValueError(f"MomentumBaseline needs '{self.feature}'")
        return self

    def predict(self, X) -> np.ndarray:
        r = X[self.feature].fillna(0.0).to_numpy()
        return np.where(r > 0, 0.6, 0.4)


# =====================================================================
# OOF generation
# =====================================================================


@dataclass
class OOFResult:
    """Out-of-fold predictions plus per-fold and pooled metrics."""

    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    task: str
    model_name: str

    @property
    def pooled_metrics(self) -> dict[str, float]:
        """Metrics over all OOF rows at once.

        Reported alongside the per-fold mean, not instead of it. Pooling hides
        fold-to-fold instability; the mean and standard deviation reveal it.
        """
        fn = classification_metrics if self.task == "direction" else regression_metrics
        return fn(self.predictions["y_true"], self.predictions["y_pred"])

    def aggregate(self) -> dict[str, float]:
        numeric = self.fold_metrics.select_dtypes("number")
        out = {}
        for col in numeric.columns:
            if col in ("fold", "n_train", "n_val", "n_purged", "n_embargoed"):
                continue
            out[f"{col}_mean"] = float(numeric[col].mean())
            out[f"{col}_std"] = float(numeric[col].std(ddof=1)) if len(numeric) > 1 else 0.0
        return out

    def render(self) -> str:
        kind = "classification" if self.task == "direction" else "regression"
        lines = [f"{self.model_name} ({len(self.fold_metrics)} folds)"]
        lines.append(f"  pooled: {summarise(self.pooled_metrics, kind)}")

        agg = self.aggregate()
        key = "roc_auc" if self.task == "direction" else "information_coefficient"
        if f"{key}_mean" in agg:
            lines.append(
                f"  per-fold {key}: {agg[f'{key}_mean']:.4f} +/- {agg[f'{key}_std']:.4f}"
            )
        for w in flag_suspicious(self.pooled_metrics, kind):
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def generate_oof(
    panel: pd.DataFrame,
    model_factory: Callable[[], BaseModel],
    feature_cols: Sequence[str],
    label_col: str,
    task: str,
    splitter: PurgedWalkForwardSplit | None = None,
    model_name: str | None = None,
) -> OOFResult:
    """Fit across purged folds and collect out-of-fold predictions.

    A fresh model is constructed per fold. Reusing one instance would carry
    fitted state across the boundary — subtle, and exactly the kind of leak that
    survives a code review.
    """
    splitter = splitter or PurgedWalkForwardSplit()
    feature_cols = list(feature_cols)

    labelled = panel[panel[label_col].notna()].reset_index(drop=True)
    if labelled.empty:
        raise ValueError(f"No rows with a non-null '{label_col}'")

    metric_fn = classification_metrics if task == "direction" else regression_metrics
    rows, frames = [], []
    resolved_name = model_name

    for fold in splitter.split(labelled):
        assert_fold_is_clean(fold, labelled, splitter.gaps)

        train = labelled.iloc[fold.train_idx]
        val = labelled.iloc[fold.val_idx]

        model = model_factory()  # fresh instance every fold
        if resolved_name is None:
            resolved_name = getattr(getattr(model, "spec", None), "name", "model")
        model.fit(train[feature_cols], train[label_col], train.get("date"))
        preds = model.predict(val[feature_cols])

        metrics = metric_fn(val[label_col], preds)
        rows.append({**fold.describe(), **metrics})

        frames.append(
            pd.DataFrame(
                {
                    "date": val["date"].to_numpy(),
                    "symbol": val["symbol"].to_numpy() if "symbol" in val else None,
                    "fold": fold.index,
                    "y_true": val[label_col].to_numpy(),
                    "y_pred": np.asarray(preds).ravel(),
                }
            )
        )

        log.info(
            "  fold %d: %s",
            fold.index,
            summarise(metrics, "classification" if task == "direction" else "regression"),
        )

    result = OOFResult(
        predictions=pd.concat(frames, ignore_index=True),
        fold_metrics=pd.DataFrame(rows),
        task=task,
        model_name=resolved_name or "model",
    )

    for w in flag_suspicious(
        result.pooled_metrics, "classification" if task == "direction" else "regression"
    ):
        log.warning("%s: %s", result.model_name, w)

    return result


def compare_models(results: Sequence[OOFResult]) -> pd.DataFrame:
    """Side-by-side comparison table, sorted by the metric that matters."""
    rows = []
    for r in results:
        pooled = r.pooled_metrics
        agg = r.aggregate()
        key = "roc_auc" if r.task == "direction" else "information_coefficient"
        rows.append(
            {
                "model": r.model_name,
                "task": r.task,
                key: pooled.get(key, float("nan")),
                f"{key}_fold_std": agg.get(f"{key}_std", float("nan")),
                **(
                    {
                        "accuracy": pooled.get("accuracy"),
                        "majority": pooled.get("majority_accuracy"),
                        "lift": pooled.get("skill_vs_majority"),
                        "brier": pooled.get("brier"),
                        "bss": pooled.get("brier_skill_score"),
                    }
                    if r.task == "direction"
                    else {
                        "dir_acc": pooled.get("directional_accuracy"),
                        "mae": pooled.get("mae"),
                        "mae_baseline": pooled.get("mae_baseline"),
                        "r2": pooled.get("r2"),
                    }
                ),
            }
        )
    df = pd.DataFrame(rows)
    sort_key = "roc_auc" if "roc_auc" in df.columns else "information_coefficient"
    return df.sort_values(sort_key, ascending=False).reset_index(drop=True)
