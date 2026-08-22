"""Storage-layer invariants.

These are protected tests. Phase 12 wires them into CI as merge blockers,
because every one of them guards a property the rest of the system assumes.
"""

from __future__ import annotations

import sqlite3

import pytest

from trademind.storage import Prediction, PredictionConflictError, PredictionStore, init_db


@pytest.fixture()
def store(tmp_path):
    conn = init_db(tmp_path / "test.db")
    yield PredictionStore(conn)
    conn.close()


def make_pred(**overrides) -> Prediction:
    base = dict(
        symbol="RELIANCE.NS",
        prediction_date="2023-06-15",
        execution_date="2023-06-16",
        model_version="m1",
        feature_version="f1",
        decision_version="d1",
        threshold_version="t1",
        signal="BUY",
        predicted_probability=0.62,
        calibrated_probability=0.58,
        predicted_return=0.012,
    )
    base.update(overrides)
    return Prediction(**base)


# -- determinism ---------------------------------------------------------


def test_prediction_id_is_deterministic():
    assert make_pred().prediction_id == make_pred().prediction_id


def test_prediction_id_changes_with_model_version():
    a = make_pred(model_version="m1")
    b = make_pred(model_version="m2")
    assert a.prediction_id != b.prediction_id


# -- idempotency (the Phase 9 acceptance criterion) ----------------------


def test_replaying_identical_prediction_creates_one_row(store):
    p = make_pred()
    first = store.save(p)
    second = store.save(p)

    assert first == second
    count = store.conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    assert count == 1


def test_conflicting_rewrite_raises(store):
    store.save(make_pred(signal="BUY"))
    with pytest.raises(PredictionConflictError):
        store.save(make_pred(signal="SELL"))


def test_conflicting_rewrite_allowed_when_explicit(store):
    store.save(make_pred(signal="BUY"))
    store.save(make_pred(signal="SELL"), allow_overwrite=True)

    rows = store.conn.execute("SELECT signal FROM predictions").fetchall()
    assert len(rows) == 1 and rows[0]["signal"] == "SELL"


def test_new_model_version_coexists_with_old(store):
    store.save(make_pred(model_version="m1"))
    store.save(make_pred(model_version="m2", signal="SELL"))

    count = store.conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    assert count == 2


# -- look-ahead guards ---------------------------------------------------


def test_same_day_execution_is_rejected():
    with pytest.raises(ValueError, match="look-ahead"):
        make_pred(prediction_date="2023-06-15", execution_date="2023-06-15")


def test_backwards_execution_is_rejected():
    with pytest.raises(ValueError, match="look-ahead"):
        make_pred(prediction_date="2023-06-15", execution_date="2023-06-14")


def test_invalid_signal_rejected():
    with pytest.raises(ValueError):
        make_pred(signal="STRONG_BUY")


def test_probability_out_of_range_rejected():
    with pytest.raises(ValueError):
        make_pred(predicted_probability=1.4)


# -- outcomes ------------------------------------------------------------


def test_resolution_scores_direction(store):
    pid = store.save(make_pred(calibrated_probability=0.58))
    store.resolve(pid, actual_return=0.008)

    row = store.get(pid)
    assert row["status"] == "RESOLVED"

    outcome = store.conn.execute(
        "SELECT * FROM outcomes WHERE prediction_id = ?", (pid,)
    ).fetchone()
    assert outcome["actual_direction"] == 1
    assert outcome["direction_correct"] == 1
    assert outcome["return_error"] == pytest.approx(0.012 - 0.008)


def test_resolution_marks_wrong_direction(store):
    pid = store.save(make_pred(calibrated_probability=0.58))
    store.resolve(pid, actual_return=-0.02)

    outcome = store.conn.execute(
        "SELECT direction_correct FROM outcomes WHERE prediction_id = ?", (pid,)
    ).fetchone()
    assert outcome["direction_correct"] == 0


def test_outcomes_are_write_once(store):
    pid = store.save(make_pred())
    store.resolve(pid, 0.01)
    with pytest.raises(RuntimeError, match="already resolved"):
        store.resolve(pid, 0.02)


def test_pending_excludes_resolved(store):
    a = store.save(make_pred(symbol="TCS.NS"))
    store.save(make_pred(symbol="INFY.NS"))
    store.resolve(a, 0.01)

    pending = store.pending()
    assert [r["symbol"] for r in pending] == ["INFY.NS"]


def test_orphan_outcome_rejected_by_foreign_key(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute(
            "INSERT INTO outcomes (prediction_id, resolved_at, actual_return, "
            "actual_direction) VALUES ('nope', '2023-01-01', 0.0, 0)"
        )


# -- runs ----------------------------------------------------------------


def test_run_lifecycle(store):
    run_id = store.start_run("daily", config_hash="abc123")
    store.finish_run(run_id, "SUCCESS")

    row = store.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    assert row["status"] == "SUCCESS" and row["finished_at"] is not None
