"""Orchestration tests.

Covers the Phase 9 acceptance list: daily pipeline, backfill, idempotency,
artifact lineage, prediction persistence, outcome resolution, monitoring
integration, retraining integration, and failure recovery.

The headline test is ``test_backfill_with_the_current_model_is_refused`` — the
contamination trap described in ``base.py``.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from trademind.orchestration.base import (
    Job,
    JobResult,
    JobStatus,
    PipelineContext,
    TemporalValidityError,
    assert_temporally_valid,
    reap_stale_runs,
)
from trademind.orchestration.daily_pipeline import (
    build_jobs,
    run_pipeline,
)
from trademind.orchestration.scheduler import (
    crontab_line,
    should_run_today,
    systemd_timer,
)
from trademind.storage import PredictionStore, init_db
from trademind.storage.predictions import Prediction

# =====================================================================
# THE contamination guard
# =====================================================================


def test_backfill_with_the_current_model_is_refused():
    """A model trained through last month cannot have predicted last year.

    The obvious backfill implementation produces rows no model could have
    produced at the time, and nothing else in the stack would notice.
    """
    with pytest.raises(TemporalValidityError, match="cannot predict"):
        assert_temporally_valid(
            prediction_date=date(2023, 1, 15),
            training_end=date(2024, 6, 30),
            model_version="prod-1",
        )


def test_predicting_the_training_end_date_itself_is_refused():
    """Boundary: the last training day is still inside the training window."""
    with pytest.raises(TemporalValidityError):
        assert_temporally_valid(date(2023, 6, 15), date(2023, 6, 15), "m")


def test_predicting_after_training_is_allowed():
    #assert_temporally_valid(date(2023, 6, 16), date(2023, 6, 15), "m") is None
    assert assert_temporally_valid(date(2023, 6, 16), date(2023, 6, 15), "m") is None

def test_missing_training_end_is_refused():
    """Without a training window, freedom from look-ahead cannot be shown."""
    with pytest.raises(TemporalValidityError, match="no recorded training_end"):
        assert_temporally_valid(date(2023, 6, 15), None, "m")


def test_string_training_end_is_accepted():
    assert_temporally_valid(date(2023, 6, 16), "2023-06-15", "m")


# =====================================================================
# Idempotency (Phase 9 acceptance)
# =====================================================================


def make_prediction(**kw) -> Prediction:
    base = dict(
        symbol="RELIANCE.NS",
        prediction_date="2023-06-15",
        execution_date="2023-06-16",
        model_version="m1",
        feature_version="f1",
        decision_version="d1",
        threshold_version="t1",
        signal="BUY",
        calibrated_probability=0.61,
        predicted_return=0.012,
        training_start="2018-01-01",
        training_end="2023-06-14",
    )
    base.update(kw)
    return Prediction(**base)


def test_rerunning_a_date_creates_no_duplicates(tmp_path):
    conn = init_db(tmp_path / "t.db")
    store = PredictionStore(conn)

    for _ in range(3):
        store.save(make_prediction())

    n = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    assert n == 1
    conn.close()


def test_a_new_model_version_coexists(tmp_path):
    """Idempotency must not prevent recording a genuinely new prediction."""
    conn = init_db(tmp_path / "t.db")
    store = PredictionStore(conn)

    store.save(make_prediction(model_version="m1"))
    store.save(make_prediction(model_version="m2", signal="SELL"))

    assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 2
    conn.close()


# =====================================================================
# Lineage (Phase 9 acceptance)
# =====================================================================

LINEAGE_FIELDS = [
    "prediction_id",
    "symbol",
    "prediction_date",
    "execution_date",
    "model_version",
    "feature_version",
    "decision_version",
    "threshold_version",
    "training_start",
    "training_end",
    "predicted_probability",
    "calibrated_probability",
    "predicted_return",
    "signal",
    "run_id",
]


def test_every_lineage_field_is_persisted(tmp_path):
    conn = init_db(tmp_path / "t.db")
    store = PredictionStore(conn)
    run_id = store.start_run("daily", config_hash="abc")

    pid = store.save(make_prediction(run_id=run_id, predicted_probability=0.58))
    row = store.get(pid)

    for field in LINEAGE_FIELDS:
        assert field in row, f"missing lineage field {field}"
    assert row["run_id"] == run_id
    assert row["training_end"] == "2023-06-14"
    conn.close()


def test_a_prediction_traces_back_to_its_run(tmp_path):
    conn = init_db(tmp_path / "t.db")
    store = PredictionStore(conn)
    run_id = store.start_run("daily", config_hash="cfg-hash-1", git_commit="abc1234")
    store.finish_run(run_id, "SUCCESS")
    pid = store.save(make_prediction(run_id=run_id))

    row = conn.execute(
        "SELECT r.config_hash, r.git_commit, r.mode FROM predictions p "
        "JOIN runs r ON r.run_id = p.run_id WHERE p.prediction_id = ?",
        (pid,),
    ).fetchone()

    assert row["config_hash"] == "cfg-hash-1"
    assert row["git_commit"] == "abc1234"
    conn.close()


# =====================================================================
# Outcome resolution
# =====================================================================


def test_outcomes_resolve_and_score(tmp_path):
    conn = init_db(tmp_path / "t.db")
    store = PredictionStore(conn)
    pid = store.save(make_prediction(calibrated_probability=0.61))

    store.resolve(pid, actual_return=0.008)

    row = store.get(pid)
    assert row["status"] == "RESOLVED"
    outcome = conn.execute("SELECT * FROM outcomes WHERE prediction_id = ?", (pid,)).fetchone()
    assert outcome["direction_correct"] == 1
    conn.close()


def test_pending_shrinks_as_outcomes_arrive(tmp_path):
    conn = init_db(tmp_path / "t.db")
    store = PredictionStore(conn)
    a = store.save(make_prediction(symbol="A.NS"))
    store.save(make_prediction(symbol="B.NS"))

    assert len(store.pending()) == 2
    store.resolve(a, 0.01)
    assert len(store.pending()) == 1
    conn.close()


# =====================================================================
# Failure recovery
# =====================================================================


def test_abandoned_runs_are_reaped(tmp_path):
    """A killed process leaves RUNNING forever, which stacks scheduler runs."""
    conn = init_db(tmp_path / "t.db")
    store = PredictionStore(conn)

    old = (datetime.now(UTC) - timedelta(hours=12)).isoformat()
    conn.execute(
        "INSERT INTO runs (run_id, mode, started_at, status, config_hash) "
        "VALUES ('stale1', 'daily', ?, 'RUNNING', 'x')",
        (old,),
    )
    assert reap_stale_runs(store, hours=6) == 1

    row = conn.execute("SELECT status FROM runs WHERE run_id='stale1'").fetchone()
    assert row["status"] == "FAILED"
    conn.close()


def test_a_recent_run_is_not_reaped(tmp_path):
    conn = init_db(tmp_path / "t.db")
    store = PredictionStore(conn)
    store.start_run("daily", config_hash="x")

    assert reap_stale_runs(store, hours=6) == 0
    conn.close()


def test_a_job_exception_becomes_a_failed_result():
    """Jobs must not raise; the pipeline needs to close out the run record."""

    class Exploding(Job):
        name = "exploding"

        def execute(self, context):
            raise RuntimeError("boom")

    context = PipelineContext(cfg=None, mode="daily", run_id="r", as_of=date.today())
    result = Exploding().run(context)

    assert result.status is JobStatus.FAILED
    assert "boom" in result.error


def test_job_duration_is_recorded():
    class Quick(Job):
        name = "quick"

        def execute(self, context):
            return JobResult(self.name, JobStatus.SUCCESS)

    context = PipelineContext(cfg=None, mode="daily", run_id="r", as_of=date.today())
    assert Quick().run(context).duration_seconds >= 0.0


# =====================================================================
# Pipeline wiring
# =====================================================================


class FakeConfig:
    """Minimal config stand-in."""

    def __init__(self, tmp_path):
        self.root = tmp_path
        self.config_hash = "test-hash"
        self.universe = ["A.NS"]
        self.feature_version = "f1"
        self.decision_version = "d1"
        self.threshold_version = "t1"
        self.history_start = date(2020, 1, 1)
        self.final_test_start = date(2024, 1, 1)
        self._data_root = tmp_path / "data"

    @property
    def data_root(self):
        return self._data_root

    def get(self, key, default=None):
        return {
            "features.target_horizon_days": 1,
            "decision.buy_threshold": None,
        }.get(key, default)


def test_modes_are_validated(tmp_path):
    with pytest.raises(ValueError, match="mode must be"):
        run_pipeline(FakeConfig(tmp_path), mode="whatever")


def test_retrain_mode_omits_the_prediction_stage():
    """Predicting with a model about to be replaced attributes rows wrongly."""
    names = [j.name for j in build_jobs("retrain")]
    assert "predictions" not in names
    assert "monitoring" in names and "retraining" in names


def test_outcomes_resolve_before_predictions_are_made():
    """Otherwise the pending queue is always one cycle stale."""
    names = [j.name for j in build_jobs("daily")]
    assert names.index("outcomes") < names.index("predictions")


def test_monitoring_runs_after_predictions():
    """Today's features must be in the current window when drift is computed."""
    names = [j.name for j in build_jobs("daily")]
    assert names.index("predictions") < names.index("monitoring")


def test_pipeline_records_a_run_even_when_it_fails(tmp_path):
    cfg = FakeConfig(tmp_path)

    class Failing(Job):
        name = "ingestion"  # critical, so the pipeline halts

        def execute(self, context):
            return JobResult(self.name, JobStatus.FAILED, "simulated")

    run = run_pipeline(cfg, mode="daily", jobs=[Failing()])

    assert run.status == "FAILED"
    conn = init_db(cfg.data_root / "trademind.db")
    row = conn.execute(
        "SELECT status, error FROM runs WHERE run_id = ?", (run.run_id,)
    ).fetchone()
    assert row["status"] == "FAILED"
    assert "simulated" in row["error"]
    conn.close()


def test_a_non_critical_failure_does_not_halt_the_pipeline(tmp_path):
    cfg = FakeConfig(tmp_path)
    executed = []

    class Noisy(Job):
        name = "monitoring"  # not critical

        def execute(self, context):
            executed.append(self.name)
            return JobResult(self.name, JobStatus.FAILED, "flaky")

    class After(Job):
        name = "retraining"

        def execute(self, context):
            executed.append(self.name)
            return JobResult(self.name, JobStatus.SUCCESS)

    run = run_pipeline(cfg, mode="daily", jobs=[Noisy(), After()])

    assert executed == ["monitoring", "retraining"]
    assert run.status == "FAILED"  # reported, but everything still ran


def test_partial_ingestion_is_not_fatal(tmp_path):
    """One delisted ticker must not stop the whole system."""
    cfg = FakeConfig(tmp_path)
    ran = []

    class Partial(Job):
        name = "ingestion"

        def execute(self, context):
            ran.append(self.name)
            return JobResult(self.name, JobStatus.PARTIAL, "1 of 15 failed")

    class Next(Job):
        name = "features"

        def execute(self, context):
            ran.append(self.name)
            return JobResult(self.name, JobStatus.SUCCESS)

    run = run_pipeline(cfg, mode="daily", jobs=[Partial(), Next()])

    assert ran == ["ingestion", "features"]
    assert run.status == "SUCCESS"


def test_context_carries_artifacts_between_jobs():
    context = PipelineContext(cfg=None, mode="daily", run_id="r", as_of=date.today())
    context.put("panel", "data")
    assert context.get("panel") == "data"
    assert context.get("absent", "fallback") == "fallback"


def test_pipeline_render_lists_every_job(tmp_path):
    cfg = FakeConfig(tmp_path)

    class Ok(Job):
        name = "features"

        def execute(self, context):
            return JobResult(self.name, JobStatus.SUCCESS, records=10)

    text = run_pipeline(cfg, mode="daily", jobs=[Ok()]).render()
    assert "features" in text and "SUCCESS" in text


# =====================================================================
# Scheduling
# =====================================================================


def test_weekend_is_not_a_run_day():
    ok, reason = should_run_today(date(2023, 6, 17))  # Saturday
    assert not ok and "not a trading session" in reason


def test_weekday_is_a_run_day():
    ok, _ = should_run_today(date(2023, 6, 15))  # Thursday
    assert ok


def test_approximate_calendar_is_disclosed():
    _ok, reason = should_run_today(date(2023, 6, 15))
    from trademind.ingestion.calendar import TradingCalendar

    if TradingCalendar().is_approximate:
        assert "approximation" in reason


def test_crontab_runs_after_the_close():
    """Running before 15:30 IST would compute features from a partial bar."""
    line = crontab_line("/opt/trademind")
    assert line.startswith("0 18 * * 1-5")
    assert "main.py daily" in line


def test_systemd_timer_is_weekdays_only():
    unit = systemd_timer("/opt/trademind")
    assert "Mon..Fri" in unit and "Persistent=true" in unit
