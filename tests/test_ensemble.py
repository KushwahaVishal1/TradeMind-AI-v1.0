"""Stacking and calibration tests.

The one that matters most is ``test_in_sample_calibration_is_circular``: it
shows a near-zero ECE produced by fitting and measuring on the same rows, beside
the real out-of-fold number. Everything else in this file protects that
distinction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trademind.ensemble.calibrator import (
    Calibrator,
    calibrate_out_of_fold,
    fit_production_calibrator,
)
from trademind.ensemble.ensemble_metrics import (
    calibration_metrics,
    decompose_brier,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_table,
    sharpness,
)
from trademind.ensemble.stacking import (
    AlignmentError,
    build_meta_features,
    fit_stack,
    meta_feature_columns,
)
from trademind.validation.embargo import GapConfig
from trademind.validation.purged_split import PurgedWalkForwardSplit

RNG = np.random.default_rng(0)


def oof_frame(n_days=1200, symbols=("A", "B", "C"), skill=0.0, bias=0.0, seed=0):
    """Synthetic OOF output with a tunable amount of skill and miscalibration."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    rows = []
    for s in symbols:
        latent = rng.normal(size=n_days)
        y = (latent * skill + rng.normal(size=n_days) > 0).astype(float)
        # Overconfident probabilities: pushed toward the extremes, plus a bias.
        raw = 1 / (1 + np.exp(-(latent * skill * 2.5 + bias)))
        rows.append(pd.DataFrame({
            "date": dates, "symbol": s, "y_true": y, "y_pred": np.clip(raw, 1e-4, 1 - 1e-4),
        }))
    return pd.concat(rows).sort_values(["date", "symbol"]).reset_index(drop=True)


def splitter(n_splits=3):
    return PurgedWalkForwardSplit(
        n_splits=n_splits, test_sessions=120,
        gaps=GapConfig(horizon=1, purge=2, embargo=3),
        min_train_sessions=252,
    )


# =====================================================================
# THE circularity demonstration
# =====================================================================

def test_in_sample_calibration_is_circular():
    """Fitting and measuring on the same rows produces a meaningless ECE.

    Isotonic regression can map any monotone input onto the observed
    frequencies of the rows it was fitted on. The resulting ECE describes the
    calibrator's flexibility, not future performance.
    """
    df = oof_frame(n_days=1400, skill=0.4, bias=0.8)

    # The wrong way: fit and measure on identical rows.
    circular = Calibrator("isotonic").fit(df["y_pred"], df["y_true"])
    ece_circular = expected_calibration_error(
        df["y_true"], circular.predict(df["y_pred"])
    )

    # The honest way: fitted on earlier blocks, measured on later ones.
    honest = calibrate_out_of_fold(df, method="isotonic", n_blocks=4)
    ece_honest = expected_calibration_error(
        honest.predictions["y_true"], honest.predictions["calibrated"]
    )

    assert ece_circular < 0.01, "in-sample ECE should be near zero by construction"
    assert ece_honest > ece_circular * 2, (
        f"out-of-fold ECE ({ece_honest:.4f}) should be materially worse than "
        f"the circular one ({ece_circular:.4f})"
    )


def test_out_of_fold_calibration_never_reuses_a_row():
    """Every calibrated row must come from a calibrator fitted on earlier data."""
    df = oof_frame(n_days=1400, skill=0.3)
    result = calibrate_out_of_fold(df, n_blocks=4)

    # Block 0 has no prior data and must be absent.
    assert result.predictions["block"].min() >= 1
    for block in result.predictions["block"].unique():
        rows = result.predictions[result.predictions["block"] == block]
        earlier = result.predictions[result.predictions["block"] < block]
        if not earlier.empty:
            assert rows["date"].min() > earlier["date"].max()


def test_calibration_needs_enough_data():
    with pytest.raises(ValueError, match="at least"):
        calibrate_out_of_fold(oof_frame(n_days=60, symbols=("A",)))


def test_calibration_reports_before_and_after():
    df = oof_frame(n_days=1400, skill=0.4, bias=1.0)
    cmp = calibrate_out_of_fold(df, n_blocks=4).comparison()
    assert {"raw", "calibrated"} == set(cmp.index)
    assert "ece_quantile" in cmp.columns


def test_calibration_reduces_a_systematic_bias():
    """A pure offset is the case calibration definitely should fix."""
    df = oof_frame(n_days=1600, skill=0.5, bias=1.5)
    result = calibrate_out_of_fold(df, method="sigmoid", n_blocks=4)

    before = abs(calibration_metrics(result.predictions["y_true"],
                                     result.predictions["raw"])["bias"])
    after = abs(calibration_metrics(result.predictions["y_true"],
                                    result.predictions["calibrated"])["bias"])
    assert after < before


# =====================================================================
# Calibrator mechanics
# =====================================================================

def test_auto_picks_sigmoid_on_small_samples():
    c = Calibrator("auto").fit(RNG.random(200), RNG.integers(0, 2, 200))
    assert c.resolved_method_ == "sigmoid"


def test_auto_picks_isotonic_on_large_samples():
    p = RNG.random(2000)
    c = Calibrator("auto").fit(p, (p + RNG.normal(0, 0.3, 2000) > 0.5).astype(int))
    assert c.resolved_method_ == "isotonic"


def test_single_class_falls_back_to_identity():
    """Both calibrators would produce a degenerate constant otherwise."""
    c = Calibrator("isotonic").fit(RNG.random(500), np.ones(500))
    assert c.resolved_method_ == "identity"


def test_calibrated_output_stays_in_range():
    p = RNG.random(1500)
    y = (p + RNG.normal(0, 0.3, 1500) > 0.5).astype(int)
    out = Calibrator("isotonic").fit(p, y).predict(np.r_[-0.5, 0.0, 0.5, 1.0, 1.5])
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_predicting_before_fitting_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        Calibrator().predict([0.5])


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown calibration"):
        Calibrator("magic")


def test_production_calibrator_uses_every_row():
    df = oof_frame(n_days=1000, skill=0.3)
    c = fit_production_calibrator(df)
    assert c.resolved_method_ in ("isotonic", "sigmoid")
    assert np.isfinite(c.predict(df["y_pred"])).all()


# =====================================================================
# Calibration metrics
# =====================================================================

def test_perfect_calibration_scores_zero_ece():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.05, 0.95, 40000)
    y = (rng.random(40000) < p).astype(int)
    assert expected_calibration_error(y, p, n_bins=10) < 0.02


def test_systematically_overconfident_model_scores_high_ece():
    rng = np.random.default_rng(4)
    p = rng.uniform(0.05, 0.95, 5000)
    y = (rng.random(5000) < 0.5).astype(int)     # outcomes ignore the forecast
    assert expected_calibration_error(y, p) > 0.15


def test_constant_forecast_is_calibrated_but_unsharp():
    """Low ECE with low sharpness is a well-calibrated constant, not a model."""
    y = (RNG.random(5000) < 0.52).astype(int)
    p = np.full(5000, 0.52)

    m = calibration_metrics(y, p)
    assert m["ece_quantile"] < 0.02
    assert m["sharpness"] < 1e-9


def test_quantile_and_uniform_binning_are_both_reported():
    df = oof_frame(n_days=800, skill=0.3)
    m = calibration_metrics(df["y_true"], df["y_pred"])
    assert "ece_quantile" in m and "ece_uniform" in m


def test_reliability_table_bins_cover_every_row():
    df = oof_frame(n_days=800, skill=0.3)
    table = reliability_table(df["y_true"], df["y_pred"], n_bins=10)
    assert table["count"].sum() == len(df)


def test_mce_ignores_tiny_bins():
    """Without a population floor, MCE reports whichever bin holds four rows."""
    df = oof_frame(n_days=800, skill=0.3)
    lenient = maximum_calibration_error(df["y_true"], df["y_pred"], min_count=1)
    strict = maximum_calibration_error(df["y_true"], df["y_pred"], min_count=200)
    assert strict <= lenient


def test_brier_decomposition_reconstructs_the_score():
    df = oof_frame(n_days=1200, skill=0.4)
    d = decompose_brier(df["y_true"], df["y_pred"])
    actual = calibration_metrics(df["y_true"], df["y_pred"])["brier"]
    assert d["brier_reconstructed"] == pytest.approx(actual, abs=0.01)


def test_resolution_is_higher_for_a_model_with_signal():
    """Resolution is the part calibration cannot manufacture."""
    weak = decompose_brier(*[oof_frame(n_days=1200, skill=0.05)[c]
                             for c in ("y_true", "y_pred")])
    strong = decompose_brier(*[oof_frame(n_days=1200, skill=1.0)[c]
                               for c in ("y_true", "y_pred")])
    assert strong["resolution"] > weak["resolution"]


def test_sharpness_of_an_empty_series_is_nan():
    assert np.isnan(sharpness([]))


# =====================================================================
# Stacking alignment
# =====================================================================

def test_meta_features_join_on_date_and_symbol():
    a = oof_frame(n_days=400, seed=1)
    b = oof_frame(n_days=400, seed=2)
    meta = build_meta_features({"m1": a, "m2": b})

    assert len(meta) == len(a)
    assert "pred_m1" in meta and "pred_m2" in meta
    assert "y_true" in meta


def test_misaligned_frames_do_not_silently_mispair():
    """The join must never pair one symbol's prediction with another's."""
    a = oof_frame(n_days=400, symbols=("A", "B"), seed=1)
    b = oof_frame(n_days=400, symbols=("A", "B"), seed=2)
    b_shuffled = b.sample(frac=1.0, random_state=0).reset_index(drop=True)

    from_ordered = build_meta_features({"m1": a, "m2": b})
    from_shuffled = build_meta_features({"m1": a, "m2": b_shuffled})

    merged = from_ordered.merge(from_shuffled, on=["date", "symbol"],
                                suffixes=("_o", "_s"))
    assert np.allclose(merged["pred_m2_o"], merged["pred_m2_s"])


def test_duplicate_oof_rows_are_rejected():
    a = oof_frame(n_days=400, symbols=("A",))
    dup = pd.concat([a, a.iloc[:5]], ignore_index=True)
    with pytest.raises(AlignmentError, match="duplicate"):
        build_meta_features({"m1": dup})


def test_missing_key_column_is_rejected():
    a = oof_frame(n_days=400).drop(columns=["symbol"])
    with pytest.raises(AlignmentError, match="lacks"):
        build_meta_features({"m1": a})


def test_disjoint_frames_are_rejected():
    a = oof_frame(n_days=300, symbols=("A",), seed=1)
    b = oof_frame(n_days=300, symbols=("B",), seed=2)
    b["date"] = b["date"] + pd.Timedelta(days=5000)

    with pytest.raises(AlignmentError, match="no rows"):
        build_meta_features({"m1": a, "m2": b})


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="No base-model"):
        build_meta_features({})


def test_context_features_can_be_attached():
    a = oof_frame(n_days=400)
    ctx = a[["date", "symbol"]].copy()
    ctx["trend_regime"] = 1.0

    meta = build_meta_features({"m1": a}, extra=ctx, extra_cols=["trend_regime"])
    assert "trend_regime" in meta_feature_columns(meta)


# =====================================================================
# Stacking behaviour
# =====================================================================

def test_stack_runs_out_of_fold():
    meta = build_meta_features({
        "m1": oof_frame(n_days=1200, skill=0.3, seed=1),
        "m2": oof_frame(n_days=1200, skill=0.2, seed=2),
    })
    result = fit_stack(meta, splitter=splitter())

    assert len(result.fold_metrics) == 3
    assert not result.predictions.duplicated(subset=["date", "symbol"]).any()


def test_stack_coefficients_are_reported_with_spread():
    """Unstable coefficients across folds mean the weights are not learnable."""
    meta = build_meta_features({
        "m1": oof_frame(n_days=1200, skill=0.4, seed=1),
        "m2": oof_frame(n_days=1200, skill=0.1, seed=2),
    })
    coefs = fit_stack(meta, splitter=splitter()).coefficients

    assert set(coefs["feature"]) == {"pred_m1", "pred_m2"}
    assert "std_coefficient" in coefs.columns


def test_stack_weights_the_stronger_base_model_higher():
    """The learned equivalent of the hand-set weights, earned from data."""
    strong = oof_frame(n_days=1600, skill=1.2, seed=1)
    weak = strong[["date", "symbol", "y_true"]].copy()
    weak["y_pred"] = RNG.random(len(weak))     # pure noise

    meta = build_meta_features({"strong": strong, "noise": weak})
    coefs = fit_stack(meta, splitter=splitter()).coefficients.set_index("feature")

    assert abs(coefs.loc["pred_strong", "mean_coefficient"]) > abs(
        coefs.loc["pred_noise", "mean_coefficient"]
    )


def test_stack_reports_whether_it_beat_the_base_models():
    meta = build_meta_features({"m1": oof_frame(n_days=1200, skill=0.3)})
    base = pd.DataFrame([{"model": "m1", "roc_auc": 0.99}])

    result = fit_stack(meta, splitter=splitter(), base_metrics=base)
    assert result.beats_best_base() is False
    assert "does NOT beat" in result.render()


def test_stack_rejects_an_unlabelled_frame():
    meta = build_meta_features({"m1": oof_frame(n_days=1200)})
    meta["y_true"] = np.nan
    with pytest.raises(ValueError, match="No labelled"):
        fit_stack(meta, splitter=splitter())


def test_stack_finds_no_skill_in_noise():
    """Stacking noise must not manufacture signal."""
    meta = build_meta_features({
        "m1": oof_frame(n_days=1200, skill=0.0, seed=1),
        "m2": oof_frame(n_days=1200, skill=0.0, seed=2),
    })
    auc = fit_stack(meta, splitter=splitter()).pooled_metrics["roc_auc"]
    assert 0.44 < auc < 0.56
