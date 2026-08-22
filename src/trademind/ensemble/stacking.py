"""Stacking: a meta-model over out-of-fold base predictions.

## Why the ensemble is learned, not hand-weighted

The original design proposed fixed weights — 0.35 direction, 0.25 price, 0.20
trend, 0.20 sentiment — with a note to validate them later. The problem is not
that the numbers are wrong; it is that there is no principled way to choose
them, they encode an assumption that the base models are independent and equally
scaled, and "validate later" in practice means tuning them against whatever data
is available, which is the test set by the time anyone gets round to it.

A meta-model learns the weights from the data, under the same purged validation
as everything else, and its coefficients are inspectable — so you still get the
interpretability the fixed weights were meant to provide, but earned.

## The alignment hazard

Base models produce separate OOF frames. Joining them is the dangerous step: a
misaligned join is silent, and produces a meta-model trained on RELIANCE's
direction paired with TCS's expected return. Nothing raises; the metrics just
come out strange, and "strange" is hard to distinguish from "this task is hard".

``build_meta_features`` joins strictly on ``(date, symbol)``, verifies every
frame contributes the same key set, and refuses to proceed on a mismatch.

## The meta-model must also be evaluated out-of-fold

Its inputs are OOF, which makes them honest inputs. That does not make the
meta-model's *own* score honest — fitting it on all the OOF rows and scoring it
on the same rows is ordinary overfitting one level up. So the meta-model runs
through the purged walk-forward splitter as well.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..models.metrics import classification_metrics, flag_suspicious, summarise
from ..validation.purged_split import PurgedWalkForwardSplit, assert_fold_is_clean

log = logging.getLogger(__name__)

KEYS = ["date", "symbol"]


class AlignmentError(ValueError):
    """Base-model OOF frames do not cover the same (date, symbol) keys."""


def build_meta_features(
    oof_frames: Mapping[str, pd.DataFrame],
    label_col: str = "y_true",
    extra: pd.DataFrame | None = None,
    extra_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """Join base-model OOF predictions into one meta-feature frame.

    ``extra`` optionally contributes leakage-safe context features — regime
    indicators, for instance — so the meta-model can learn that a base model is
    more reliable in some states than others. Those columns must come from the
    Phase 2 panel and must already be causal.
    """
    if not oof_frames:
        raise ValueError("No base-model OOF frames supplied")

    key_sets, merged = {}, None
    for name, frame in oof_frames.items():
        missing = [k for k in KEYS if k not in frame.columns]
        if missing:
            raise AlignmentError(f"{name}: OOF frame lacks {missing}")
        if frame.duplicated(subset=KEYS).any():
            raise AlignmentError(
                f"{name}: duplicate (date, symbol) rows. Each row must be "
                "predicted exactly once out-of-fold."
            )

        part = frame[KEYS + ["y_pred"] + ([label_col] if label_col in frame else [])]
        part = part.rename(columns={"y_pred": f"pred_{name}"})
        key_sets[name] = set(map(tuple, part[KEYS].to_numpy()))

        if merged is None:
            merged = part
        else:
            part = part.drop(columns=[label_col], errors="ignore")
            merged = merged.merge(part, on=KEYS, how="inner", validate="one_to_one")

    # A silent inner join would quietly drop rows; report the loss explicitly.
    reference = next(iter(key_sets.values()))
    for name, keys in key_sets.items():
        if keys != reference:
            only_a, only_b = len(reference - keys), len(keys - reference)
            log.warning(
                "%s covers a different key set (%d missing, %d extra); the join "
                "keeps only the intersection.", name, only_a, only_b,
            )

    if merged is None or merged.empty:
        raise AlignmentError(
            "Joining the base-model OOF frames produced no rows. The frames "
            "cover disjoint (date, symbol) keys."
        )

    if extra is not None and len(extra_cols):
        missing = [c for c in extra_cols if c not in extra.columns]
        if missing:
            raise ValueError(f"extra frame lacks {missing}")
        merged = merged.merge(
            extra[KEYS + list(extra_cols)], on=KEYS, how="left", validate="one_to_one"
        )

    merged = merged.sort_values(KEYS).reset_index(drop=True)
    log.info(
        "Meta features: %d rows | %d base models | %d context columns",
        len(merged), len(oof_frames), len(extra_cols),
    )
    return merged


def meta_feature_columns(frame: pd.DataFrame, label_col: str = "y_true") -> list[str]:
    return [c for c in frame.columns if c not in KEYS + [label_col, "fold", "block"]]


@dataclass
class StackResult:
    """Out-of-fold meta-model predictions and the coefficients that produced them."""

    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    coefficients: pd.DataFrame
    base_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def pooled_metrics(self) -> dict[str, float]:
        return classification_metrics(
            self.predictions["y_true"], self.predictions["y_pred"]
        )

    def beats_best_base(self, metric: str = "roc_auc") -> bool | None:
        """Did stacking actually help? Often the answer is no."""
        if self.base_metrics.empty or metric not in self.base_metrics.columns:
            return None
        # Explicit bool(): pandas comparisons yield numpy.bool_, which fails an
        # `is True` / `is False` identity check and confuses JSON serialisation.
        return bool(self.pooled_metrics.get(metric, 0) > self.base_metrics[metric].max())

    def render(self) -> str:
        lines = ["stacked meta-model"]
        lines.append(f"  pooled: {summarise(self.pooled_metrics)}")

        if not self.base_metrics.empty:
            best = self.base_metrics.loc[self.base_metrics["roc_auc"].idxmax()]
            verdict = "improves on" if self.beats_best_base() else "does NOT beat"
            lines.append(
                f"  {verdict} the best base model "
                f"({best['model']}, AUC {best['roc_auc']:.4f})"
            )

        lines.append("  mean coefficients:")
        for _, r in self.coefficients.iterrows():
            lines.append(f"    {r['feature']:<28} {r['mean_coefficient']:+.4f}")

        for w in flag_suspicious(self.pooled_metrics):
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def fit_stack(
    meta: pd.DataFrame,
    label_col: str = "y_true",
    feature_cols: Sequence[str] | None = None,
    splitter: PurgedWalkForwardSplit | None = None,
    C: float = 1.0,
    base_metrics: pd.DataFrame | None = None,
) -> StackResult:
    """Fit and evaluate the meta-model out-of-fold.

    Logistic regression on standardised meta-features. Deliberately the
    simplest thing that works: the meta-model sits on top of a handful of
    correlated inputs derived from a weak signal, and anything more flexible
    will fit the noise in the base models' errors. Its coefficients are also
    directly readable as the learned equivalent of the hand-set weights.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    splitter = splitter or PurgedWalkForwardSplit()
    cols = list(feature_cols) if feature_cols else meta_feature_columns(meta, label_col)
    if not cols:
        raise ValueError("No meta-features to stack")

    data = meta[meta[label_col].notna()].reset_index(drop=True)
    if data.empty:
        raise ValueError("No labelled rows in the meta frame")

    rows, frames, coefs = [], [], []

    for fold in splitter.split(data):
        assert_fold_is_clean(fold, data, splitter.gaps)

        train = data.iloc[fold.train_idx]
        val = data.iloc[fold.val_idx]

        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=C, solver="lbfgs", max_iter=2000,
                                         random_state=42)),
        ])
        X_train = train[cols].fillna(train[cols].median())
        pipe.fit(X_train, (train[label_col] > 0.5).astype(int))

        X_val = val[cols].fillna(train[cols].median())
        preds = pipe.predict_proba(X_val)[:, 1]

        metrics = classification_metrics(val[label_col], preds)
        rows.append({**fold.describe(), **metrics})

        coefs.append(dict(zip(cols, pipe.named_steps["model"].coef_.ravel())))
        frames.append(pd.DataFrame({
            "date": val["date"].to_numpy(),
            "symbol": val["symbol"].to_numpy() if "symbol" in val else None,
            "fold": fold.index,
            "y_true": val[label_col].to_numpy(),
            "y_pred": preds,
        }))

        log.info("  meta fold %d: %s", fold.index, summarise(metrics))

    coef_frame = pd.DataFrame(coefs)
    coefficients = pd.DataFrame({
        "feature": cols,
        "mean_coefficient": coef_frame.mean().reindex(cols).to_numpy(),
        "std_coefficient": (
            coef_frame.std(ddof=1).reindex(cols).to_numpy()
            if len(coef_frame) > 1 else np.zeros(len(cols))
        ),
    }).sort_values("mean_coefficient", key=abs, ascending=False).reset_index(drop=True)

    return StackResult(
        predictions=pd.concat(frames, ignore_index=True),
        fold_metrics=pd.DataFrame(rows),
        coefficients=coefficients,
        base_metrics=base_metrics if base_metrics is not None else pd.DataFrame(),
    )
