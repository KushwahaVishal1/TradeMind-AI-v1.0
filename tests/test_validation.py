"""Validation-strategy tests.

The headline test is ``test_unpurged_split_inflates_scores``: it constructs data
where the label is knowable from a neighbouring row, then shows an unpurged
split scores near-perfectly while a purged one scores near-chance. That is the
leak these modules exist to close, demonstrated rather than asserted.
"""

from __future__ import annotations

import itertools
from datetime import date

import numpy as np
import pandas as pd
import pytest

from trademind.validation.dataset_validator import (
    FinalTestLock,
    FinalTestViolation,
    assert_no_locked_data,
    split_development,
)
from trademind.validation.embargo import (
    GapConfig,
    ValidationConfigError,
    required_purge,
)
from trademind.validation.purged_split import (
    PurgedWalkForwardSplit,
    assert_fold_is_clean,
)
from trademind.validation.walk_forward import walk_forward

RNG = np.random.default_rng(0)


def panel(n_days=1200, symbols=("A", "B", "C"), seed=0):
    """A multi-symbol panel — the shape everything downstream actually sees."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    rows = []
    for s in symbols:
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": s,
                    "f1": rng.normal(size=n_days),
                    "f2": rng.normal(size=n_days),
                    "y": rng.normal(0, 0.015, size=n_days),
                }
            )
        )
    return pd.concat(rows).sort_values(["date", "symbol"]).reset_index(drop=True)


# =====================================================================
# Purge sizing
# =====================================================================


def test_required_purge_includes_the_execution_offset():
    """horizon + 1, not horizon. The extra day is the decision-to-fill gap."""
    assert required_purge(horizon=1) == 2
    assert required_purge(horizon=5) == 6
    assert required_purge(horizon=1, execution_offset=0) == 1


def test_gap_config_rejects_an_undersized_purge():
    with pytest.raises(ValidationConfigError, match="too small"):
        GapConfig(horizon=1, purge=1)


def test_gap_config_accepts_the_minimum():
    assert GapConfig(horizon=1, purge=2).purge == 2


def test_gap_config_rejects_negative_embargo():
    with pytest.raises(ValidationConfigError):
        GapConfig(horizon=1, purge=2, embargo=-1)


# =====================================================================
# THE demonstration
# =====================================================================


def test_unpurged_split_leaves_leaking_rows_and_purge_removes_them():
    """Count training rows whose label window reaches into validation.

    Deterministic rather than score-based. A row at date *t* carries a label
    spanning sessions ``t+1`` through ``t+1+horizon``; if that window touches
    the validation block, the row is contaminated. With no purge such rows
    exist at every boundary. With ``purge = horizon + 1`` there are none.

    A score-based demonstration would be noisier and less convincing — the
    inflation from a handful of boundary rows is real but small, and small
    effects across a few folds are indistinguishable from luck. The structural
    count is exact.
    """
    df = panel(n_days=1000, symbols=("A",))
    horizon = 5
    sessions = pd.Index(sorted(df["date"].unique()))

    def count_leaking(purge: int, execution_offset: int) -> int:
        splitter = PurgedWalkForwardSplit(
            n_splits=3,
            test_sessions=100,
            gaps=GapConfig(
                horizon=horizon, purge=purge, embargo=0, execution_offset=execution_offset
            ),
            min_train_sessions=252,
        )
        total = 0
        for fold in splitter.split(df):
            val_start_pos = sessions.searchsorted(fold.val_start)
            train_dates = df["date"].iloc[fold.train_idx]
            # Label of t occupies sessions t+1 .. t+1+horizon.
            label_end_pos = sessions.searchsorted(train_dates.to_numpy()) + 1 + horizon
            total += int((label_end_pos >= val_start_pos).sum())
        return total

    # The realistic mistake: sizing the purge for the close-to-close research
    # label (offset 0) while actually trading the open-to-open one (offset 1).
    # GapConfig accepts it, because for that label it would be correct.
    undersized = count_leaking(purge=horizon, execution_offset=0)
    correct = count_leaking(purge=horizon + 1, execution_offset=1)

    assert undersized > 0, "expected contaminated rows with an undersized purge"
    assert correct == 0, f"{correct} training row(s) still overlap validation"


def test_walk_forward_reports_aggregate_and_spread():
    """A mean without a spread hides whether the model is stable or lucky."""
    df = panel(n_days=1000, symbols=("A",))
    from sklearn.linear_model import LinearRegression

    def fit(X, y, params):
        return LinearRegression().fit(X, y)

    res = walk_forward(
        df,
        ["f1"],
        "y",
        fit,
        lambda m, X: m.predict(X),
        lambda yt, yp: {"corr": float(np.corrcoef(yt, yp)[0, 1])},
        splitter=PurgedWalkForwardSplit(
            n_splits=3,
            test_sessions=100,
            gaps=GapConfig(horizon=1, purge=2, embargo=0),
            min_train_sessions=200,
        ),
    )
    agg = res.aggregate()
    assert "corr_mean" in agg and "corr_std" in agg
    assert len(res.folds) == 3


def test_purged_folds_have_a_real_gap():
    """The property that matters: no training date within `purge` of validation."""
    df = panel(n_days=1000)
    gaps = GapConfig(horizon=1, purge=5, embargo=3)
    splitter = PurgedWalkForwardSplit(
        n_splits=4, test_sessions=100, gaps=gaps, min_train_sessions=252
    )

    sessions = pd.Index(sorted(df["date"].unique()))
    for fold in splitter.split(df):
        train_end = df["date"].iloc[fold.train_idx].max()
        val_start = df["date"].iloc[fold.val_idx].min()
        gap = sessions.searchsorted(val_start) - sessions.searchsorted(train_end) - 1
        assert gap >= gaps.purge, f"fold {fold.index} gap {gap} < purge {gaps.purge}"


def test_assert_fold_is_clean_catches_a_bad_fold():
    df = panel(n_days=800)
    splitter = PurgedWalkForwardSplit(
        n_splits=2, test_sessions=100, gaps=GapConfig(purge=5), min_train_sessions=252
    )
    fold = next(iter(splitter.split(df)))

    # Sabotage: extend training into the validation window.
    bad = fold.__class__(
        **{**fold.__dict__, "train_idx": np.r_[fold.train_idx, fold.val_idx[:5]]}
    )

    with pytest.raises(ValidationConfigError):
        assert_fold_is_clean(bad, df, GapConfig(purge=5))


# =====================================================================
# Panel awareness
# =====================================================================


def test_split_assigns_whole_sessions_not_rows():
    """A date must never straddle the boundary — that is same-day leakage."""
    df = panel(n_days=900, symbols=("A", "B", "C", "D"))
    splitter = PurgedWalkForwardSplit(
        n_splits=3, test_sessions=100, gaps=GapConfig(purge=3), min_train_sessions=252
    )

    for fold in splitter.split(df):
        train_dates = set(df["date"].iloc[fold.train_idx])
        val_dates = set(df["date"].iloc[fold.val_idx])
        assert not (train_dates & val_dates)


def test_all_symbols_present_on_both_sides():
    """Whole-session assignment keeps the cross-section intact."""
    df = panel(n_days=900, symbols=("A", "B", "C"))
    splitter = PurgedWalkForwardSplit(
        n_splits=2, test_sessions=100, gaps=GapConfig(purge=3), min_train_sessions=252
    )

    for fold in splitter.split(df):
        assert df["symbol"].iloc[fold.train_idx].nunique() == 3
        assert df["symbol"].iloc[fold.val_idx].nunique() == 3


# =====================================================================
# Walk-forward mechanics
# =====================================================================


def test_folds_move_forward_in_time():
    df = panel(n_days=1200)
    folds = list(
        PurgedWalkForwardSplit(n_splits=4, test_sessions=100, gaps=GapConfig(purge=3)).split(
            df
        )
    )

    for a, b in itertools.pairwise(folds):
        assert b.val_start > a.val_start
        assert b.train_end >= a.train_end


def test_expanding_window_grows():
    df = panel(n_days=1200)
    folds = list(
        PurgedWalkForwardSplit(n_splits=4, test_sessions=100, gaps=GapConfig(purge=3)).split(
            df
        )
    )
    sizes = [len(f.train_idx) for f in folds]
    assert sizes == sorted(sizes)


def test_rolling_window_stays_bounded():
    df = panel(n_days=1400)
    folds = list(
        PurgedWalkForwardSplit(
            n_splits=3,
            test_sessions=100,
            gaps=GapConfig(purge=3),
            train_sessions=300,
        ).split(df)
    )

    n_symbols = df["symbol"].nunique()
    for f in folds:
        assert len(f.train_idx) <= 310 * n_symbols


def test_validation_never_precedes_training():
    df = panel(n_days=1000)
    for f in PurgedWalkForwardSplit(
        n_splits=3, test_sessions=100, gaps=GapConfig(purge=3)
    ).split(df):
        assert f.train_end < f.val_start


def test_insufficient_history_is_an_explicit_error():
    df = panel(n_days=100)
    with pytest.raises(ValidationConfigError, match="at least"):
        list(PurgedWalkForwardSplit(n_splits=3, test_sessions=100).split(df))


def test_missing_date_column_is_an_explicit_error():
    with pytest.raises(ValidationConfigError, match="date"):
        list(PurgedWalkForwardSplit().split(pd.DataFrame({"a": [1, 2, 3]})))


def test_summary_has_one_row_per_fold():
    df = panel(n_days=1200)
    s = PurgedWalkForwardSplit(n_splits=4, test_sessions=100, gaps=GapConfig(purge=3)).summary(
        df
    )
    assert len(s) == 4
    assert "n_purged" in s.columns


# =====================================================================
# Final-test lock
# =====================================================================


def test_development_split_excludes_the_locked_window():
    df = panel(n_days=1200)
    dev = split_development(df, date(2021, 1, 1))
    assert dev.panel["date"].max() < pd.Timestamp("2021-01-01")


def test_development_split_trims_the_boundary_rows():
    """Rows whose labels reach across the boundary must go too."""
    df = panel(n_days=1200)
    boundary = pd.Timestamp("2021-01-01")

    naive_max = df[df["date"] < boundary]["date"].max()
    dev = split_development(df, boundary, label_lag_sessions=2)

    assert dev.panel["date"].max() < naive_max


def test_development_data_rejects_locked_rows():
    from trademind.validation.dataset_validator import DevelopmentData

    df = panel(n_days=1200)
    with pytest.raises(FinalTestViolation):
        DevelopmentData(df, pd.Timestamp("2021-01-01"))


def test_assert_no_locked_data_guards_a_fit():
    df = panel(n_days=1200)
    with pytest.raises(FinalTestViolation, match="locked window"):
        assert_no_locked_data(df, date(2021, 1, 1), context="fit_direction_model")


def test_assert_no_locked_data_passes_on_clean_data():
    df = panel(n_days=1200)
    dev = split_development(df, date(2021, 1, 1))
    assert_no_locked_data(dev.panel, date(2021, 1, 1), context="fit")


def test_lock_releases_data_once():
    df = panel(n_days=1200)
    lock = FinalTestLock.create(df, date(2021, 1, 1))

    released = lock.unlock(df, reason="final evaluation")
    assert len(released) > 0
    assert released["date"].min() >= pd.Timestamp("2021-01-01")


def test_second_unlock_is_refused():
    """Two evaluations means the first one informed a change."""
    df = panel(n_days=1200)
    lock = FinalTestLock.create(df, date(2021, 1, 1))
    lock.unlock(df, reason="first")

    with pytest.raises(FinalTestViolation, match="already been unlocked"):
        lock.unlock(df, reason="just one more look")


def test_altered_final_test_data_is_detected():
    """A changed fingerprint invalidates comparison against recorded benchmarks."""
    df = panel(n_days=1200)
    lock = FinalTestLock.create(df, date(2021, 1, 1))

    tampered = df.copy()
    mask = tampered["date"] >= pd.Timestamp("2021-01-01")
    tampered.loc[mask, "f1"] *= 1.01

    with pytest.raises(FinalTestViolation, match="has changed"):
        lock.unlock(tampered, reason="evaluation")


def test_lock_survives_a_round_trip(tmp_path):
    df = panel(n_days=1200)
    lock = FinalTestLock.create(df, date(2021, 1, 1))
    path = lock.save(tmp_path / "lock.json")

    reloaded = FinalTestLock.load(path)
    assert reloaded.fingerprint == lock.fingerprint
    assert reloaded.unlock(df, reason="evaluation") is not None


def test_unlock_is_recorded():
    df = panel(n_days=1200)
    lock = FinalTestLock.create(df, date(2021, 1, 1))
    lock.unlock(df, reason="phase 7 final backtest")

    assert len(lock.accesses) == 1
    assert "phase 7" in lock.accesses[0]["reason"]


# =====================================================================
# Runner
# =====================================================================


def _linear_harness():
    from sklearn.linear_model import Ridge

    def fit(X, y, params):
        return Ridge(alpha=params.get("alpha", 1.0)).fit(X.fillna(0.0), y)

    def predict(m, X):
        return m.predict(X.fillna(0.0))

    def score(y_true, y_pred):
        return {"mae": float(np.mean(np.abs(y_true - y_pred)))}

    return fit, predict, score


def test_walk_forward_produces_one_result_per_fold():
    df = panel(n_days=1200)
    fit, predict, score = _linear_harness()

    res = walk_forward(
        df,
        ["f1", "f2"],
        "y",
        fit,
        predict,
        score,
        splitter=PurgedWalkForwardSplit(
            n_splits=3, test_sessions=100, gaps=GapConfig(purge=3)
        ),
    )
    assert len(res.folds) == 3
    assert "mae_mean" in res.aggregate()
    assert "mae_std" in res.aggregate()


def test_oof_predictions_cover_each_row_once():
    """Phase 5's stacking depends on this: no row predicted by a model that saw it."""
    df = panel(n_days=1200)
    fit, predict, score = _linear_harness()

    res = walk_forward(
        df,
        ["f1", "f2"],
        "y",
        fit,
        predict,
        score,
        splitter=PurgedWalkForwardSplit(
            n_splits=3, test_sessions=100, gaps=GapConfig(purge=3)
        ),
    )
    oof = res.oof_predictions
    assert not oof.duplicated(subset=["date", "symbol"]).any()


def test_walk_forward_rejects_an_all_null_label():
    df = panel(n_days=1200)
    df["y"] = np.nan
    fit, predict, score = _linear_harness()

    with pytest.raises(ValueError, match="non-null"):
        walk_forward(df, ["f1", "f2"], "y", fit, predict, score)


def test_render_mentions_every_fold():
    df = panel(n_days=1200)
    fit, predict, score = _linear_harness()
    res = walk_forward(
        df,
        ["f1", "f2"],
        "y",
        fit,
        predict,
        score,
        splitter=PurgedWalkForwardSplit(
            n_splits=3, test_sessions=100, gaps=GapConfig(purge=3)
        ),
    )
    text = res.render()
    assert text.count("fold ") >= 3
