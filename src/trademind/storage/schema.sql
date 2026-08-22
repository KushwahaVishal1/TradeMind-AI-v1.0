-- TradeMind AI operational store.
--
-- Holds every decision the system has ever made and what the market did next.
-- Append-mostly and immutable by intent: a prediction, once written, is a
-- historical fact. Corrections create new rows, they do not edit old ones.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- Every execution of the pipeline, successful or not.
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    mode            TEXT NOT NULL CHECK (mode IN ('backfill', 'daily', 'retrain', 'test')),
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    status          TEXT NOT NULL CHECK (status IN ('RUNNING', 'SUCCESS', 'FAILED')),
    git_commit      TEXT,
    config_hash     TEXT NOT NULL,
    python_version  TEXT,
    error           TEXT
);

-- One row per (symbol, date, model configuration).
--
-- The natural key deliberately includes every version field. Re-running the
-- same day with the same artifacts is a no-op; re-running after bumping a
-- model or threshold version produces a genuinely new prediction that can be
-- compared against the old one. That is what makes the daily job idempotent
-- without making it unable to ever change its mind.
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id         TEXT PRIMARY KEY,
    run_id                TEXT REFERENCES runs(run_id),

    symbol                TEXT NOT NULL,
    prediction_date       TEXT NOT NULL,   -- date features were computed (t)
    execution_date        TEXT,            -- session the decision acts on (t+1)

    -- lineage
    model_version         TEXT NOT NULL,
    feature_version       TEXT NOT NULL,
    decision_version      TEXT NOT NULL,
    threshold_version     TEXT NOT NULL,
    training_start        TEXT,
    training_end          TEXT,

    -- model output
    predicted_probability REAL CHECK (predicted_probability BETWEEN 0.0 AND 1.0),
    calibrated_probability REAL CHECK (calibrated_probability BETWEEN 0.0 AND 1.0),
    predicted_return      REAL,

    -- decision output
    signal                TEXT NOT NULL CHECK (signal IN ('BUY', 'HOLD', 'SELL')),
    target_weight         REAL,
    risk_bucket           TEXT,

    status                TEXT NOT NULL DEFAULT 'PREDICTED'
                            CHECK (status IN ('PREDICTED', 'RESOLVED', 'VOID')),
    created_at            TEXT NOT NULL,

    UNIQUE (symbol, prediction_date, model_version, feature_version,
            decision_version, threshold_version)
);

CREATE INDEX IF NOT EXISTS idx_pred_date   ON predictions(prediction_date);
CREATE INDEX IF NOT EXISTS idx_pred_symbol ON predictions(symbol, prediction_date);
CREATE INDEX IF NOT EXISTS idx_pred_status ON predictions(status);

-- Realized outcome, written once the market has actually moved.
-- Separate table because outcomes arrive later and must never be part of the
-- prediction's own identity.
CREATE TABLE IF NOT EXISTS outcomes (
    prediction_id     TEXT PRIMARY KEY REFERENCES predictions(prediction_id),
    resolved_at       TEXT NOT NULL,
    actual_return     REAL NOT NULL,
    actual_direction  INTEGER NOT NULL CHECK (actual_direction IN (0, 1)),
    direction_correct INTEGER CHECK (direction_correct IN (0, 1)),
    return_error      REAL
);

-- Model lifecycle: CANDIDATE -> VALIDATING -> VALIDATED -> STAGING -> PRODUCTION
--                                          -> FAILED
CREATE TABLE IF NOT EXISTS model_registry (
    model_version   TEXT PRIMARY KEY,
    model_type      TEXT NOT NULL,
    stage           TEXT NOT NULL CHECK (stage IN
                        ('CANDIDATE', 'VALIDATING', 'VALIDATED',
                         'STAGING', 'PRODUCTION', 'FAILED', 'ARCHIVED')),
    created_at      TEXT NOT NULL,
    promoted_at     TEXT,
    training_start  TEXT,
    training_end    TEXT,
    feature_version TEXT,
    artifact_path   TEXT,
    metrics_json    TEXT,
    notes           TEXT
);

-- Data-quality findings from ingestion. Reported, never silently auto-fixed.
CREATE TABLE IF NOT EXISTS data_quality_issues (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT REFERENCES runs(run_id),
    symbol      TEXT,
    issue_date  TEXT,
    severity    TEXT NOT NULL CHECK (severity IN ('INFO', 'WARNING', 'ERROR')),
    code        TEXT NOT NULL,
    detail      TEXT,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dq_severity ON data_quality_issues(severity, created_at);

-- Every experiment ever run, with enough metadata to reproduce it.
--
-- The `n_prior_experiments` column exists because selection inflation is real:
-- picking the best of N experiments by validation metric inflates that metric
-- by roughly sigma * z(1 - 1/(N+1)). Recording N makes the inflation
-- computable after the fact instead of invisible.
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id       TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    task                TEXT NOT NULL,
    created_at          TEXT NOT NULL,

    -- reproducibility
    git_commit          TEXT,
    config_hash         TEXT,
    dataset_version     TEXT,
    feature_version     TEXT,
    random_seed         INTEGER,
    python_version      TEXT,
    package_versions    TEXT,

    -- what was run
    model_type          TEXT,
    hyperparameters     TEXT,
    validation_strategy TEXT,
    training_start      TEXT,
    training_end        TEXT,
    n_training_rows     INTEGER,
    n_features          INTEGER,

    -- what happened
    metrics             TEXT,
    primary_metric      TEXT,
    primary_value       REAL,
    n_prior_experiments INTEGER DEFAULT 0,
    notes               TEXT
);

CREATE INDEX IF NOT EXISTS idx_exp_task ON experiments(task, primary_value);
CREATE INDEX IF NOT EXISTS idx_exp_created ON experiments(created_at);

-- Audit trail of every registry stage transition. Append-only.
CREATE TABLE IF NOT EXISTS registry_transitions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version  TEXT NOT NULL,
    from_stage     TEXT,
    to_stage       TEXT NOT NULL,
    occurred_at    TEXT NOT NULL,
    actor          TEXT,
    reason         TEXT
);

CREATE INDEX IF NOT EXISTS idx_transitions_model
    ON registry_transitions(model_version, occurred_at);

-- Joined view used by every monitoring routine in Phase 8.
CREATE VIEW IF NOT EXISTS resolved_predictions AS
SELECT
    p.*,
    o.actual_return,
    o.actual_direction,
    o.direction_correct,
    o.return_error,
    o.resolved_at
FROM predictions p
JOIN outcomes o ON o.prediction_id = p.prediction_id;
