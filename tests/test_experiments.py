"""Experiment tracking and registry lifecycle tests.

The headline test is ``test_fifty_experiments_on_noise_produce_a_flaggable_auc``:
it simulates the exact situation an experiment tracker creates and shows the
winning number landing in the range Phase 4 warns about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trademind.experiments import (
    ArtifactStore,
    ExperimentRun,
    ExperimentTracker,
    IllegalTransition,
    ModelLifecycle,
    Stage,
    adjusted_best,
    compare_experiments,
    is_distinguishable,
    render_comparison,
    selection_inflation,
)
from trademind.storage import init_db


@pytest.fixture()
def conn(tmp_path):
    connection = init_db(tmp_path / "t.db")
    yield connection
    connection.close()


def make_run(name="exp", task="direction", auc=0.52, **kw) -> ExperimentRun:
    base = dict(
        name=name, task=task, model_type="hgb",
        hyperparameters={"max_depth": 3},
        config_hash="cfg1", dataset_version="d1", feature_version="f1",
        random_seed=42, training_start="2018-01-01", training_end="2022-12-31",
        n_training_rows=5000, n_features=45,
        metrics={"roc_auc": auc, "brier": 0.25},
    )
    base.update(kw)
    run = ExperimentRun(**base)
    run.git_commit = run.git_commit or "abc1234"
    run.git_dirty = False
    return run


# =====================================================================
# THE selection-inflation finding
# =====================================================================

def test_selection_inflation_grows_with_experiment_count():
    assert selection_inflation(1) == 0.0
    assert selection_inflation(10) > selection_inflation(5)
    assert selection_inflation(100) > selection_inflation(50)


def test_fifty_experiments_on_noise_produce_a_flaggable_auc():
    """Simulates the tracker's own failure mode.

    Fifty worthless models, each with true AUC 0.52 and a 0.024 standard error.
    The best observed lands near 0.57 — inside the range Phase 4's
    flag_suspicious warns about — purely from selection.
    """
    rng = np.random.default_rng(7)
    observed = rng.normal(0.52, 0.024, 50)
    best = observed.max()

    assert best > 0.55, f"expected selection to inflate; got {best:.4f}"

    adjusted, explanation = adjusted_best(best, n_experiments=50)
    assert adjusted < best
    assert abs(adjusted - 0.52) < 0.03
    assert "inflates" in explanation


def test_single_experiment_needs_no_adjustment():
    value, explanation = adjusted_best(0.55, n_experiments=1)
    assert value == 0.55
    assert "no selection adjustment" in explanation


def test_tracker_records_how_many_experiments_preceded(conn):
    tracker = ExperimentTracker(conn)
    for i in range(4):
        tracker.log(make_run(name=f"exp{i}"))

    runs = tracker.all(task="direction")
    assert [r.n_prior_experiments for r in runs] == [0, 1, 2, 3]


def test_leaderboard_carries_the_selection_discount(conn):
    tracker = ExperimentTracker(conn)
    for i, auc in enumerate([0.51, 0.56, 0.53]):
        tracker.log(make_run(name=f"exp{i}", auc=auc))

    board = tracker.leaderboard(task="direction")
    assert board.iloc[0]["observed"] == 0.56
    assert board.iloc[0]["selection_adjusted"] < 0.56


def test_summary_states_the_adjusted_estimate(conn):
    tracker = ExperimentTracker(conn)
    for i, auc in enumerate([0.50, 0.52, 0.57, 0.51]):
        tracker.log(make_run(name=f"e{i}", auc=auc))

    text = tracker.summary(task="direction")
    assert "best observed" in text
    assert "locked final test" in text


# =====================================================================
# Reproducibility
# =====================================================================

def test_a_complete_run_is_reproducible():
    assert make_run().reproducible


def test_a_missing_seed_is_a_gap():
    run = make_run(random_seed=None)
    assert not run.reproducible
    assert any("seed" in g for g in run.reproducibility_gaps())


def test_a_dirty_tree_is_a_gap():
    """A commit hash recorded against a dirty tree points at code that never ran."""
    run = make_run()
    run.git_dirty = True

    assert not run.reproducible
    assert any("dirty" in g for g in run.reproducibility_gaps())


def test_missing_dataset_version_is_a_gap():
    assert not make_run(dataset_version=None).reproducible


def test_render_flags_irreproducibility():
    assert "NOT REPRODUCIBLE" in make_run(config_hash=None).render()


def test_fingerprint_is_stable():
    assert make_run().fingerprint == make_run().fingerprint


def test_fingerprint_changes_with_hyperparameters():
    a = make_run(hyperparameters={"max_depth": 3})
    b = make_run(hyperparameters={"max_depth": 5})
    assert a.fingerprint != b.fingerprint


def test_duplicate_configurations_are_findable(conn):
    """Identical fingerprints should produce identical results, or something is wrong."""
    tracker = ExperimentTracker(conn)
    first = make_run(name="first")
    tracker.log(first)

    duplicate = tracker.find_duplicate(make_run(name="second"))
    assert duplicate is not None
    assert duplicate.name == "first"


def test_round_trip_through_storage(conn):
    tracker = ExperimentTracker(conn)
    run = make_run(notes="baseline sweep")
    tracker.log(run)

    loaded = tracker.get(run.experiment_id)
    assert loaded.name == run.name
    assert loaded.hyperparameters == run.hyperparameters
    assert loaded.metrics["roc_auc"] == pytest.approx(0.52)
    assert loaded.notes == "baseline sweep"


# =====================================================================
# Comparison
# =====================================================================

def test_close_results_are_not_distinguishable():
    """0.53 and 0.54 with a 0.024 stderr are the same experiment."""
    assert not is_distinguishable(0.53, 0.54)


def test_far_apart_results_are_distinguishable():
    assert is_distinguishable(0.50, 0.65)


def test_difference_stderr_is_larger_than_single_stderr():
    """The sqrt(2) factor: omitting it makes differences look significant."""
    stderr = 0.024
    assert not is_distinguishable(0.50, 0.50 + 2 * stderr, stderr)
    assert is_distinguishable(0.50, 0.50 + 3 * stderr, stderr)


def test_comparison_marks_the_indistinguishable_field():
    runs = [make_run(name=f"e{i}", auc=a)
            for i, a in enumerate([0.520, 0.525, 0.530, 0.600])]
    frame = compare_experiments(runs)

    assert frame.iloc[0]["observed"] == 0.600
    tied = frame[~frame["distinguishable_from_best"]]
    assert len(tied) >= 1


def test_render_advises_the_simpler_model_on_a_tie():
    runs = [make_run(name=f"e{i}", auc=a) for i, a in enumerate([0.52, 0.525, 0.53])]
    text = render_comparison(compare_experiments(runs))
    assert "prefer the simpler model" in text


def test_comparison_of_nothing_is_empty():
    assert compare_experiments([]).empty


# =====================================================================
# Registry lifecycle
# =====================================================================

def test_registration_starts_at_candidate(conn):
    lifecycle = ModelLifecycle(conn)
    lifecycle.register("m1", "hgb", task="direction")
    assert lifecycle.stage_of("m1") == Stage.CANDIDATE


def test_model_versions_are_immutable(conn):
    lifecycle = ModelLifecycle(conn)
    lifecycle.register("m1", "hgb")
    with pytest.raises(ValueError, match="immutable"):
        lifecycle.register("m1", "hgb")


def test_candidate_cannot_jump_to_production(conn):
    """The whole point of the lifecycle."""
    lifecycle = ModelLifecycle(conn)
    lifecycle.register("m1", "hgb")

    with pytest.raises(IllegalTransition, match="Cannot move"):
        lifecycle.transition("m1", Stage.PRODUCTION)


def test_failed_is_terminal(conn):
    """Phase 8's rule, enforced a second time at the registry."""
    lifecycle = ModelLifecycle(conn)
    lifecycle.register("m1", "hgb")
    lifecycle.transition("m1", Stage.VALIDATING)
    lifecycle.transition("m1", Stage.FAILED, "did not pass")

    for target in (Stage.VALIDATED, Stage.STAGING, Stage.PRODUCTION):
        with pytest.raises(IllegalTransition, match="terminal"):
            lifecycle.transition("m1", target)


def test_full_promotion_path(conn):
    lifecycle = ModelLifecycle(conn)
    lifecycle.register("m1", "hgb")

    for stage in (Stage.VALIDATING, Stage.VALIDATED, Stage.STAGING,
                  Stage.PRODUCTION):
        lifecycle.transition("m1", stage)

    assert lifecycle.stage_of("m1") == Stage.PRODUCTION
    assert lifecycle.get("m1")["promoted_at"] is not None


def test_promotion_archives_the_incumbent(conn):
    """Two live production models mean ambiguous prediction lineage."""
    lifecycle = ModelLifecycle(conn)
    for version in ("m1", "m2"):
        lifecycle.register(version, "hgb", task="direction")
        for stage in (Stage.VALIDATING, Stage.VALIDATED, Stage.STAGING,
                      Stage.PRODUCTION):
            lifecycle.transition(version, stage)

    assert lifecycle.stage_of("m1") == Stage.ARCHIVED
    assert lifecycle.stage_of("m2") == Stage.PRODUCTION


def test_only_one_production_model_is_returned(conn):
    lifecycle = ModelLifecycle(conn)
    for version in ("m1", "m2"):
        lifecycle.register(version, "hgb", task="direction")
        for stage in (Stage.VALIDATING, Stage.VALIDATED, Stage.STAGING,
                      Stage.PRODUCTION):
            lifecycle.transition(version, stage)

    assert lifecycle.production(task="direction")["model_version"] == "m2"


def test_different_tasks_can_both_be_in_production(conn):
    lifecycle = ModelLifecycle(conn)
    for version, task in (("d1", "direction"), ("r1", "return")):
        lifecycle.register(version, "hgb", task=task)
        for stage in (Stage.VALIDATING, Stage.VALIDATED, Stage.STAGING,
                      Stage.PRODUCTION):
            lifecycle.transition(version, stage)

    assert lifecycle.production(task="direction")["model_version"] == "d1"
    assert lifecycle.production(task="return")["model_version"] == "r1"


def test_transitions_are_audited(conn):
    lifecycle = ModelLifecycle(conn)
    lifecycle.register("m1", "hgb")
    lifecycle.transition("m1", Stage.VALIDATING, "nightly check", actor="ci")

    history = lifecycle.history("m1")
    assert len(history) == 2
    assert history.iloc[-1]["to_stage"] == Stage.VALIDATING
    assert history.iloc[-1]["actor"] == "ci"
    assert history.iloc[-1]["reason"] == "nightly check"


def test_unregistered_model_cannot_transition(conn):
    with pytest.raises(KeyError):
        ModelLifecycle(conn).transition("ghost", Stage.VALIDATING)


def test_no_production_model_returns_none(conn):
    assert ModelLifecycle(conn).production() is None


# =====================================================================
# Guided promotion
# =====================================================================

class FakeOutcome:
    def __init__(self, passed, failed_checks=()):
        self.passed = passed
        self.failed_checks = list(failed_checks)


def test_guided_promotion_records_every_intermediate_stage(conn):
    """Jumping straight to PRODUCTION would lose the audit trail."""
    lifecycle = ModelLifecycle(conn)
    lifecycle.register("m1", "hgb")

    final = lifecycle.promote_through_validation("m1", FakeOutcome(True))

    assert final == Stage.PRODUCTION
    stages = list(lifecycle.history("m1")["to_stage"])
    assert stages == [Stage.CANDIDATE, Stage.VALIDATING, Stage.VALIDATED,
                      Stage.STAGING, Stage.PRODUCTION]


def test_guided_promotion_fails_a_bad_candidate(conn):
    lifecycle = ModelLifecycle(conn)
    lifecycle.register("m1", "hgb")

    final = lifecycle.promote_through_validation(
        "m1", FakeOutcome(False, ["auc_not_worse"])
    )

    assert final == Stage.FAILED
    assert lifecycle.stage_of("m1") == Stage.FAILED
    assert "auc_not_worse" in lifecycle.history("m1").iloc[-1]["reason"]


# =====================================================================
# Artifacts
# =====================================================================

def test_artifacts_round_trip(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    frame = pd.DataFrame({"a": [1, 2, 3]})

    store.save_frame("exp1", "oof", frame)
    store.save_json("exp1", "metrics", {"roc_auc": 0.52})

    assert store.load_frame("exp1", "oof").equals(frame)
    assert store.load_json("exp1", "metrics")["roc_auc"] == 0.52


def test_artifacts_are_listed(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    store.save_frame("exp1", "oof", pd.DataFrame({"a": [1]}))
    store.save_json("exp1", "metrics", {})

    assert set(store.list_artifacts("exp1")) == {"oof.csv", "metrics.json"}


def test_missing_artifact_returns_none(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    assert store.load_frame("nope", "oof") is None
    assert store.list_artifacts("nope") == []
