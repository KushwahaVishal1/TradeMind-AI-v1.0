"""Base-model tests.

Two carry the most weight:

``test_scaler_fitted_outside_the_fold_leaks`` demonstrates the preprocessing
leak numerically rather than describing it.

``test_no_skill_on_pure_noise`` asserts the models find *nothing* in random
data. A pipeline that scores well on noise is broken, and this catches the
failure that every other metric would flatter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trademind.models.base import ModelSpec
from trademind.models.direction import DirectionModel, hgb_direction, logistic_direction
from trademind.models.metrics import (
    auc_stderr,
    classification_metrics,
    flag_suspicious,
    regression_metrics,
)
from trademind.models.oof import (
    MajorityBaseline,
    MeanBaseline,
    compare_models,
    generate_oof,
)
from trademind.models.registry import ModelRegistry
from trademind.models.return_model import ReturnModel, hgb_return, ridge_return
from trademind.validation.embargo import GapConfig
from trademind.validation.purged_split import PurgedWalkForwardSplit

FEATURES = ["f1", "f2", "f3"]


def noise_panel(n_days=1000, symbols=("A", "B", "C"), seed=0, signal=0.0):
    """Random features and returns, optionally with a planted linear signal."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    out = []
    for s in symbols:
        f = rng.normal(size=(n_days, 3))
        noise = rng.normal(0, 0.015, n_days)
        y = signal * f[:, 0] * 0.015 + noise
        out.append(pd.DataFrame({
            "date": dates, "symbol": s,
            "f1": f[:, 0], "f2": f[:, 1], "f3": f[:, 2],
            "tradeable_return_1d": y,
            "tradeable_direction_1d": (y > 0).astype(float),
        }))
    return pd.concat(out).sort_values(["date", "symbol"]).reset_index(drop=True)


def splitter(n_splits=3):
    return PurgedWalkForwardSplit(
        n_splits=n_splits, test_sessions=120,
        gaps=GapConfig(horizon=1, purge=2, embargo=3),
        min_train_sessions=252,
    )


# =====================================================================
# The preprocessing leak
# =====================================================================

def test_scaler_fitted_outside_the_fold_leaks():
    """Standardising before splitting moves validation-fold feature values.

    Demonstrates the leak numerically. The purged splitter cannot help: the
    scaler already read the whole panel, so every training fold sees features
    normalised with statistics drawn from the validation window.
    """
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(1)
    n = 600
    x = rng.normal(size=n)
    x[400:] += 25.0            # a regime shift in the validation period

    # Wrong: fitted on everything.
    global_scaled = StandardScaler().fit_transform(x.reshape(-1, 1)).ravel()
    # Right: fitted on training only.
    fold_scaler = StandardScaler().fit(x[:400].reshape(-1, 1))
    fold_scaled = fold_scaler.transform(x.reshape(-1, 1)).ravel()

    assert not np.allclose(global_scaled[:400], fold_scaled[:400]), (
        "expected the global scaler to contaminate training-fold values"
    )
    # The training rows differ materially, not marginally.
    assert np.abs(global_scaled[:400] - fold_scaled[:400]).mean() > 0.1


def test_model_pipeline_contains_its_own_preprocessing():
    """Preprocessing must live inside the estimator, so fit() scopes it."""
    model = logistic_direction()
    model.fit(
        pd.DataFrame(np.random.default_rng(0).normal(size=(300, 3)), columns=FEATURES),
        pd.Series(np.random.default_rng(1).integers(0, 2, 300).astype(float)),
    )
    step_names = [name for name, _ in model.pipeline.steps]
    assert "scale" in step_names and "impute" in step_names


def test_gradient_boosting_does_not_impute():
    """Trees route NaN natively; imputing would discard informative missingness."""
    assert "impute" not in [n for n, _ in hgb_direction()._build_pipeline().steps]


# =====================================================================
# No skill on noise
# =====================================================================

def test_no_skill_on_pure_noise():
    """AUC must sit near 0.5 on random data. Anything else means a leak."""
    panel = noise_panel(n_days=1000, signal=0.0)

    result = generate_oof(
        panel, logistic_direction, FEATURES,
        "tradeable_direction_1d", task="direction", splitter=splitter(),
    )
    auc = result.pooled_metrics["roc_auc"]
    assert 0.44 < auc < 0.56, f"AUC {auc:.3f} on pure noise indicates leakage"


def test_return_model_finds_no_signal_in_noise():
    panel = noise_panel(n_days=1000, signal=0.0)

    result = generate_oof(
        panel, ridge_return, FEATURES,
        "tradeable_return_1d", task="return", splitter=splitter(),
    )
    ic = result.pooled_metrics["information_coefficient"]
    assert abs(ic) < 0.10, f"IC {ic:.3f} on pure noise indicates leakage"


def test_model_does_find_a_planted_signal():
    """The converse: if it cannot find a real signal, the harness is broken."""
    panel = noise_panel(n_days=1000, signal=3.0)

    result = generate_oof(
        panel, logistic_direction, FEATURES,
        "tradeable_direction_1d", task="direction", splitter=splitter(),
    )
    assert result.pooled_metrics["roc_auc"] > 0.65


# =====================================================================
# Metrics
# =====================================================================

def test_majority_baseline_is_reported():
    y = np.r_[np.ones(70), np.zeros(30)]
    m = classification_metrics(y, np.full(100, 0.7))
    assert m["majority_accuracy"] == pytest.approx(0.7)
    assert m["base_rate"] == pytest.approx(0.7)


def test_skill_is_measured_against_the_majority_not_zero():
    """A constant 'always up' predictor must score zero lift, not 70% accuracy."""
    y = np.r_[np.ones(70), np.zeros(30)]
    m = classification_metrics(y, np.full(100, 0.99))
    assert m["accuracy"] == pytest.approx(0.7)
    assert m["skill_vs_majority"] == pytest.approx(0.0)


def test_auc_stderr_shrinks_with_sample_size():
    small = auc_stderr(0.55, 100, 100)
    large = auc_stderr(0.55, 5000, 5000)
    assert small > large > 0


def test_auc_z_flags_indistinguishable_from_chance():
    y = np.r_[np.ones(300), np.zeros(300)]
    rng = np.random.default_rng(0)
    m = classification_metrics(y, rng.random(600))
    assert abs(m["auc_z"]) < 3.0


def test_brier_skill_score_is_zero_for_the_base_rate_forecast():
    y = np.r_[np.ones(60), np.zeros(40)]
    m = classification_metrics(y, np.full(100, 0.6))
    assert m["brier_skill_score"] == pytest.approx(0.0, abs=1e-9)


def test_regression_baseline_is_the_mean():
    y = np.array([0.01, -0.02, 0.03, -0.01, 0.005])
    m = regression_metrics(y, np.full(5, y.mean()))
    assert m["mae"] == pytest.approx(m["mae_baseline"])
    assert m["r2"] == pytest.approx(0.0, abs=1e-9)


def test_constant_prediction_reports_zero_ic():
    """A collapsed model must not look merely mediocre."""
    y = np.random.default_rng(0).normal(size=200)
    m = regression_metrics(y, np.full(200, 0.001))
    assert m["information_coefficient"] == 0.0


def test_implausible_auc_is_flagged():
    warnings = flag_suspicious({"roc_auc": 0.82, "skill_vs_majority": 0.02})
    assert warnings and "leakage" in warnings[0].lower()


def test_plausible_auc_is_not_flagged():
    assert not flag_suspicious({"roc_auc": 0.53, "skill_vs_majority": 0.01})


def test_implausible_ic_is_flagged():
    warnings = flag_suspicious({"information_coefficient": 0.4, "r2": 0.0}, "regression")
    assert warnings


# =====================================================================
# Baselines
# =====================================================================

def test_majority_baseline_predicts_a_constant():
    b = MajorityBaseline().fit(pd.DataFrame({"a": range(10)}), pd.Series([1.0] * 7 + [0.0] * 3))
    preds = b.predict(pd.DataFrame({"a": range(5)}))
    assert np.allclose(preds, 0.7)


def test_mean_baseline_predicts_the_training_mean():
    y = pd.Series([0.01, 0.02, 0.03])
    b = MeanBaseline().fit(pd.DataFrame({"a": range(3)}), y)
    assert b.predict(pd.DataFrame({"a": range(2)}))[0] == pytest.approx(0.02)


def test_baseline_auc_is_undefined_not_flattering():
    """A constant predictor has AUC 0.5 exactly — no accidental skill."""
    y = np.r_[np.ones(50), np.zeros(50)]
    m = classification_metrics(y, np.full(100, 0.5))
    assert m["roc_auc"] == pytest.approx(0.5)


# =====================================================================
# OOF mechanics
# =====================================================================

def test_oof_rows_are_unique():
    panel = noise_panel(n_days=1000)
    r = generate_oof(panel, logistic_direction, FEATURES,
                     "tradeable_direction_1d", "direction", splitter())
    assert not r.predictions.duplicated(subset=["date", "symbol"]).any()


def test_oof_predictions_are_out_of_sample():
    """Every predicted date must fall strictly inside its fold's validation window."""
    panel = noise_panel(n_days=1000)
    sp = splitter()
    r = generate_oof(panel, logistic_direction, FEATURES,
                     "tradeable_direction_1d", "direction", sp)

    labelled = panel[panel["tradeable_direction_1d"].notna()].reset_index(drop=True)
    for fold in sp.split(labelled):
        rows = r.predictions[r.predictions["fold"] == fold.index]
        assert rows["date"].min() >= fold.val_start
        assert rows["date"].max() <= fold.val_end


def test_each_fold_gets_a_fresh_model():
    """Reusing an instance would carry fitted state across the boundary."""
    built = []

    def factory():
        m = logistic_direction()
        built.append(m)
        return m

    panel = noise_panel(n_days=1000)
    generate_oof(panel, factory, FEATURES, "tradeable_direction_1d",
                 "direction", splitter(n_splits=3))

    assert len(built) == 3
    assert len({id(m) for m in built}) == 3


def test_fold_metrics_report_spread():
    panel = noise_panel(n_days=1000)
    r = generate_oof(panel, logistic_direction, FEATURES,
                     "tradeable_direction_1d", "direction", splitter())
    agg = r.aggregate()
    assert "roc_auc_mean" in agg and "roc_auc_std" in agg


def test_compare_models_ranks_them():
    panel = noise_panel(n_days=1000, signal=2.0)
    results = [
        generate_oof(panel, logistic_direction, FEATURES,
                     "tradeable_direction_1d", "direction", splitter()),
        generate_oof(panel, hgb_direction, FEATURES,
                     "tradeable_direction_1d", "direction", splitter()),
    ]
    table = compare_models(results)
    assert len(table) == 2
    assert table["roc_auc"].is_monotonic_decreasing
    assert "majority" in table.columns


def test_render_surfaces_warnings():
    panel = noise_panel(n_days=1000, signal=8.0)
    r = generate_oof(panel, logistic_direction, FEATURES,
                     "tradeable_direction_1d", "direction", splitter())
    assert "WARNING" in r.render()


# =====================================================================
# Model mechanics
# =====================================================================

def test_direction_model_returns_probabilities():
    panel = noise_panel(n_days=400)
    m = logistic_direction().fit(panel[FEATURES], panel["tradeable_direction_1d"])
    p = m.predict(panel[FEATURES])
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_column_order_is_restored_at_predict_time():
    """A silent reordering would scramble every prediction without raising."""
    panel = noise_panel(n_days=400)
    m = logistic_direction().fit(panel[FEATURES], panel["tradeable_direction_1d"])

    a = m.predict(panel[FEATURES])
    b = m.predict(panel[["f3", "f1", "f2"]])
    assert np.allclose(a, b)


def test_missing_feature_at_predict_time_raises():
    panel = noise_panel(n_days=400)
    m = logistic_direction().fit(panel[FEATURES], panel["tradeable_direction_1d"])
    with pytest.raises(ValueError, match="Missing features"):
        m.predict(panel[["f1", "f2"]])


def test_predicting_before_fitting_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        logistic_direction().predict(pd.DataFrame({"f1": [1.0]}))


def test_fitting_on_empty_data_raises():
    with pytest.raises(ValueError):
        logistic_direction().fit(pd.DataFrame(columns=FEATURES), pd.Series(dtype=float))


def test_fitting_is_deterministic():
    """Two runs on identical input must agree, or lineage means nothing."""
    panel = noise_panel(n_days=400)
    a = hgb_direction().fit(panel[FEATURES], panel["tradeable_direction_1d"])
    b = hgb_direction().fit(panel[FEATURES], panel["tradeable_direction_1d"])
    assert np.allclose(a.predict(panel[FEATURES]), b.predict(panel[FEATURES]))


def test_unknown_estimator_raises():
    m = DirectionModel(ModelSpec("x", "direction", "not_a_real_estimator"))
    with pytest.raises(ValueError, match="Unknown"):
        m.fit(pd.DataFrame({"f1": [1.0, 2.0]}), pd.Series([0.0, 1.0]))


def test_nan_features_are_tolerated():
    panel = noise_panel(n_days=400).copy()
    panel.loc[panel.index[:50], "f1"] = np.nan

    for factory in (logistic_direction, hgb_direction):
        m = factory().fit(panel[FEATURES], panel["tradeable_direction_1d"])
        assert np.isfinite(m.predict(panel[FEATURES])).all()


# =====================================================================
# Versioning and registry
# =====================================================================

def test_spec_hash_is_stable():
    a = ModelSpec("m", "direction", "logistic", {"C": 1.0})
    b = ModelSpec("m", "direction", "logistic", {"C": 1.0})
    assert a.spec_hash == b.spec_hash


def test_spec_hash_changes_with_params():
    a = ModelSpec("m", "direction", "logistic", {"C": 1.0})
    b = ModelSpec("m", "direction", "logistic", {"C": 2.0})
    assert a.spec_hash != b.spec_hash


def test_version_includes_the_training_end_date():
    panel = noise_panel(n_days=400)
    m = logistic_direction().fit(
        panel[FEATURES], panel["tradeable_direction_1d"], panel["date"]
    )
    assert m.training_end_ in m.version


def test_registry_round_trip(tmp_path):
    panel = noise_panel(n_days=400)
    m = hgb_direction().fit(
        panel[FEATURES], panel["tradeable_direction_1d"], panel["date"]
    )
    reg = ModelRegistry(tmp_path / "models")
    reg.save(m)

    loaded = reg.load(m.version).model
    assert np.allclose(loaded.predict(panel[FEATURES]), m.predict(panel[FEATURES]))
    assert loaded.feature_names_ == m.feature_names_


def test_registry_refuses_an_unfitted_model(tmp_path):
    with pytest.raises(RuntimeError, match="unfitted"):
        ModelRegistry(tmp_path / "m").save(logistic_direction())


def test_registry_records_the_environment(tmp_path):
    panel = noise_panel(n_days=400)
    m = ridge_return().fit(panel[FEATURES], panel["tradeable_return_1d"], panel["date"])
    reg = ModelRegistry(tmp_path / "models")
    reg.save(m)

    meta = reg.load(m.version).metadata
    assert "sklearn" in meta["environment"]
    assert meta["n_training_rows"] == len(panel)


def test_registry_lists_and_summarises(tmp_path):
    panel = noise_panel(n_days=400)
    reg = ModelRegistry(tmp_path / "models")
    for factory, label in ((logistic_direction, "tradeable_direction_1d"),
                           (ridge_return, "tradeable_return_1d")):
        reg.save(factory().fit(panel[FEATURES], panel[label], panel["date"]))

    assert len(reg.list_versions()) == 2
    assert set(reg.summary()["task"]) == {"direction", "return"}


def test_loading_an_unknown_version_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ModelRegistry(tmp_path / "models").load("nope-000000000000")


def test_native_importance_is_saved_for_linear_models(tmp_path):
    panel = noise_panel(n_days=400)
    m = ridge_return().fit(panel[FEATURES], panel["tradeable_return_1d"], panel["date"])
    path = ModelRegistry(tmp_path / "models").save(m)
    assert (path / "feature_importance.csv").exists()


def test_hgb_has_no_native_importance_and_says_so(tmp_path):
    """sklearn deliberately omits feature_importances_ on HistGradientBoosting.

    Record the reason rather than leaving a silently absent file.
    """
    panel = noise_panel(n_days=400)
    m = hgb_return().fit(panel[FEATURES], panel["tradeable_return_1d"], panel["date"])

    assert m.feature_importance() is None
    path = ModelRegistry(tmp_path / "models").save(m)
    assert (path / "feature_importance.txt").exists()
    assert "permutation" in (path / "feature_importance.txt").read_text()


def test_permutation_importance_works_for_hgb():
    """The general fallback, and the only option for boosted trees here."""
    panel = noise_panel(n_days=600, signal=3.0)
    m = hgb_return().fit(panel[FEATURES], panel["tradeable_return_1d"])

    imp = m.permutation_importance(panel[FEATURES], panel["tradeable_return_1d"],
                                   n_repeats=3)
    assert set(imp["feature"]) == set(FEATURES)
    # f1 carries the planted signal, so it should rank first.
    assert imp.iloc[0]["feature"] == "f1"


def test_feature_importance_covers_every_feature():
    panel = noise_panel(n_days=400)
    m = logistic_direction().fit(panel[FEATURES], panel["tradeable_direction_1d"])
    imp = m.feature_importance()
    assert set(imp["feature"]) == set(FEATURES)
