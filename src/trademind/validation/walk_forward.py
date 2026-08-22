"""Walk-forward evaluation runner and nested hyperparameter tuning.

Two entry points:

``walk_forward``
    Fit and score across purged folds, verifying each fold's separation before
    it is used.

``nested_walk_forward``
    Hyperparameter selection inside an *inner* purged split of each fold's
    training data, then a single fit on the full training block, then scoring on
    the untouched outer validation block.

The nesting is what makes reported validation scores honest. Tuning on the same
folds you report is selection bias: with enough configurations, the best
validation score is mostly luck, and the gap between it and true performance
grows with the size of the search. Nesting costs an inner loop and buys a number
that means something.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .embargo import GapConfig
from .purged_split import Fold, PurgedWalkForwardSplit, assert_fold_is_clean

log = logging.getLogger(__name__)

# fit(X_train, y_train, params) -> fitted model
FitFn = Callable[[pd.DataFrame, pd.Series, dict], Any]
# predict(model, X) -> array
PredictFn = Callable[[Any, pd.DataFrame], np.ndarray]
# score(y_true, y_pred) -> dict of metrics
ScoreFn = Callable[[pd.Series, np.ndarray], dict[str, float]]


@dataclass
class FoldResult:
    fold: Fold
    metrics: dict[str, float]
    params: dict = field(default_factory=dict)
    predictions: pd.DataFrame | None = None


@dataclass
class WalkForwardResult:
    """Per-fold results plus their aggregate."""

    folds: list[FoldResult]

    @property
    def metrics_frame(self) -> pd.DataFrame:
        rows = []
        for r in self.folds:
            rows.append({**r.fold.describe(), **r.metrics})
        return pd.DataFrame(rows)

    def aggregate(self) -> dict[str, float]:
        """Mean and standard deviation of each metric across folds.

        The standard deviation is the important half. A model averaging 0.54
        AUC with a 0.01 spread is a different proposition from one averaging
        0.54 with a 0.09 spread — the second is noise, and a single-number
        summary would hide it.
        """
        if not self.folds:
            return {}
        keys = self.folds[0].metrics.keys()
        out: dict[str, float] = {}
        for k in keys:
            vals = [r.metrics[k] for r in self.folds if k in r.metrics]
            out[f"{k}_mean"] = float(np.mean(vals))
            out[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        return out

    @property
    def oof_predictions(self) -> pd.DataFrame:
        """Out-of-fold predictions, concatenated.

        The input Phase 5's meta-model trains on. Each row was predicted by a
        model that never saw it, which is the entire basis of honest stacking.
        """
        frames = [r.predictions for r in self.folds if r.predictions is not None]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def render(self) -> str:
        agg = self.aggregate()
        lines = [f"{len(self.folds)} folds"]
        for r in self.folds:
            metrics = " ".join(f"{k}={v:.4f}" for k, v in r.metrics.items())
            lines.append(f"  {r.fold} | {metrics}")
        lines.append("  aggregate: " + " ".join(
            f"{k}={v:.4f}" for k, v in agg.items() if k.endswith("_mean")
        ))
        return "\n".join(lines)


def walk_forward(
    panel: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
    fit_fn: FitFn,
    predict_fn: PredictFn,
    score_fn: ScoreFn,
    splitter: PurgedWalkForwardSplit | None = None,
    params: dict | None = None,
    keep_predictions: bool = True,
) -> WalkForwardResult:
    """Fit and score across purged walk-forward folds."""
    splitter = splitter or PurgedWalkForwardSplit()
    params = params or {}
    results: list[FoldResult] = []

    labelled = panel[panel[label_col].notna()].reset_index(drop=True)
    if labelled.empty:
        raise ValueError(f"No rows with a non-null '{label_col}'")

    for fold in splitter.split(labelled):
        # Never trust the splitter; verify.
        assert_fold_is_clean(fold, labelled, splitter.gaps)

        train = labelled.iloc[fold.train_idx]
        val = labelled.iloc[fold.val_idx]

        model = fit_fn(train[list(feature_cols)], train[label_col], params)
        preds = predict_fn(model, val[list(feature_cols)])
        metrics = score_fn(val[label_col], preds)

        pred_frame = None
        if keep_predictions:
            pred_frame = pd.DataFrame({
                "date": val["date"].to_numpy(),
                "symbol": val["symbol"].to_numpy() if "symbol" in val else None,
                "fold": fold.index,
                "y_true": val[label_col].to_numpy(),
                "y_pred": np.asarray(preds).ravel(),
            })

        results.append(FoldResult(fold, metrics, dict(params), pred_frame))
        log.info("%s | %s", fold,
                 " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    return WalkForwardResult(results)


def nested_walk_forward(
    panel: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
    fit_fn: FitFn,
    predict_fn: PredictFn,
    score_fn: ScoreFn,
    param_grid: Iterable[dict],
    selection_metric: str,
    maximise: bool = True,
    outer_splitter: PurgedWalkForwardSplit | None = None,
    inner_splits: int = 3,
) -> WalkForwardResult:
    """Walk-forward with hyperparameter selection nested inside each fold.

    The inner split is purged with the same gaps as the outer one — tuning on an
    unpurged inner loop would reintroduce exactly the leakage the outer loop
    prevents, and the resulting configuration would be selected for its ability
    to exploit that leakage.
    """
    outer = outer_splitter or PurgedWalkForwardSplit()
    grid = list(param_grid)
    if not grid:
        raise ValueError("param_grid is empty")

    labelled = panel[panel[label_col].notna()].reset_index(drop=True)
    results: list[FoldResult] = []

    for fold in outer.split(labelled):
        assert_fold_is_clean(fold, labelled, outer.gaps)

        train = labelled.iloc[fold.train_idx].reset_index(drop=True)
        val = labelled.iloc[fold.val_idx]

        inner = PurgedWalkForwardSplit(
            n_splits=inner_splits,
            test_sessions=max(21, outer.test_sessions // 2),
            gaps=outer.gaps,
            min_train_sessions=max(126, outer.min_train_sessions // 2),
        )

        best_params, best_score = grid[0], None
        for candidate in grid:
            scores = []
            for inner_fold in inner.split(train):
                itr = train.iloc[inner_fold.train_idx]
                iva = train.iloc[inner_fold.val_idx]
                m = fit_fn(itr[list(feature_cols)], itr[label_col], candidate)
                p = predict_fn(m, iva[list(feature_cols)])
                scores.append(score_fn(iva[label_col], p)[selection_metric])

            mean_score = float(np.mean(scores))
            better = (
                best_score is None
                or (mean_score > best_score if maximise else mean_score < best_score)
            )
            if better:
                best_params, best_score = candidate, mean_score

        log.info("fold %d: selected %s (inner %s=%.4f)",
                 fold.index, best_params, selection_metric, best_score)

        model = fit_fn(train[list(feature_cols)], train[label_col], best_params)
        preds = predict_fn(model, val[list(feature_cols)])
        metrics = score_fn(val[label_col], preds)

        results.append(FoldResult(
            fold, metrics, dict(best_params),
            pd.DataFrame({
                "date": val["date"].to_numpy(),
                "symbol": val["symbol"].to_numpy() if "symbol" in val else None,
                "fold": fold.index,
                "y_true": val[label_col].to_numpy(),
                "y_pred": np.asarray(preds).ravel(),
            }),
        ))

    return WalkForwardResult(results)
