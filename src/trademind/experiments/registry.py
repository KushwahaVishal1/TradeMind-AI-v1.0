"""Model registry lifecycle.

::

    CANDIDATE -> VALIDATING -> VALIDATED -> STAGING -> PRODUCTION -> ARCHIVED
                     |
                     +------> FAILED    (terminal)

Two invariants, both structural rather than procedural:

**Illegal transitions raise.** A candidate cannot jump straight to PRODUCTION,
and FAILED is terminal — there is no path out of it. Phase 8's rule that a
failed candidate cannot reach production is enforced twice: once at the
promotion decision, once here. Two independent guards on the same rule is
deliberate; that rule is the one most likely to be worked around under pressure.

**One PRODUCTION model per task.** Promoting a new one archives the incumbent
in the same transaction. Two live production models mean predictions carry
ambiguous lineage, and the monitoring layer would silently average the two.

Every transition is appended to ``registry_transitions`` with a timestamp,
actor, and reason. The table is append-only: the history of what was promoted
and when is itself a record worth not being able to edit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

log = logging.getLogger(__name__)


class Stage:
    CANDIDATE = "CANDIDATE"
    VALIDATING = "VALIDATING"
    VALIDATED = "VALIDATED"
    STAGING = "STAGING"
    PRODUCTION = "PRODUCTION"
    FAILED = "FAILED"
    ARCHIVED = "ARCHIVED"


LEGAL_TRANSITIONS: dict[str, set[str]] = {
    Stage.CANDIDATE: {Stage.VALIDATING, Stage.FAILED},
    Stage.VALIDATING: {Stage.VALIDATED, Stage.FAILED},
    Stage.VALIDATED: {Stage.STAGING, Stage.FAILED, Stage.ARCHIVED},
    Stage.STAGING: {Stage.PRODUCTION, Stage.FAILED, Stage.ARCHIVED},
    Stage.PRODUCTION: {Stage.ARCHIVED},
    Stage.FAILED: set(),  # terminal, by design
    Stage.ARCHIVED: set(),
}


class IllegalTransition(RuntimeError):
    """An attempt to move a model between stages that are not connected."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class ModelLifecycle:
    """Stage management over the ``model_registry`` table."""

    conn: object

    # -- registration -----------------------------------------------------

    def register(
        self,
        model_version: str,
        model_type: str,
        task: str = "direction",
        training_start: str | None = None,
        training_end: str | None = None,
        feature_version: str | None = None,
        artifact_path: str | None = None,
        metrics: dict | None = None,
        notes: str = "",
    ) -> str:
        """Register a new model as CANDIDATE."""
        import json

        existing = self.get(model_version)
        if existing is not None:
            raise ValueError(
                f"{model_version} is already registered at stage "
                f"{existing['stage']}. Model versions are immutable."
            )

        self.conn.execute(
            "INSERT INTO model_registry (model_version, model_type, stage, "
            "created_at, training_start, training_end, feature_version, "
            "artifact_path, metrics_json, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                model_version,
                f"{task}:{model_type}",
                Stage.CANDIDATE,
                _now(),
                training_start,
                training_end,
                feature_version,
                artifact_path,
                json.dumps(metrics or {}),
                notes,
            ),
        )
        self._log_transition(model_version, None, Stage.CANDIDATE, "registered")
        return model_version

    def get(self, model_version: str):
        return self.conn.execute(
            "SELECT * FROM model_registry WHERE model_version = ?", (model_version,)
        ).fetchone()

    def stage_of(self, model_version: str) -> str | None:
        row = self.get(model_version)
        return row["stage"] if row else None

    # -- transitions ------------------------------------------------------

    def transition(
        self,
        model_version: str,
        to_stage: str,
        reason: str = "",
        actor: str = "pipeline",
    ) -> str:
        """Move a model to a new stage, or raise if the move is illegal."""
        row = self.get(model_version)
        if row is None:
            raise KeyError(f"{model_version} is not registered")

        current = row["stage"]
        allowed = LEGAL_TRANSITIONS.get(current, set())

        if to_stage not in allowed:
            detail = (
                f"{current} is terminal"
                if not allowed
                else f"legal moves from {current} are {sorted(allowed)}"
            )
            raise IllegalTransition(
                f"Cannot move {model_version} from {current} to {to_stage}: {detail}."
            )

        # One PRODUCTION model per task, enforced with the promotion itself.
        if to_stage == Stage.PRODUCTION:
            self._archive_incumbent(row["model_type"], model_version)

        self.conn.execute(
            "UPDATE model_registry SET stage = ?, promoted_at = ? WHERE model_version = ?",
            (
                to_stage,
                _now() if to_stage == Stage.PRODUCTION else row["promoted_at"],
                model_version,
            ),
        )
        self._log_transition(model_version, current, to_stage, reason, actor)
        log.info("%s: %s -> %s (%s)", model_version, current, to_stage, reason)
        return to_stage

    def _archive_incumbent(self, model_type: str, incoming: str) -> None:
        rows = self.conn.execute(
            "SELECT model_version FROM model_registry "
            "WHERE stage = ? AND model_type = ? AND model_version != ?",
            (Stage.PRODUCTION, model_type, incoming),
        ).fetchall()

        for row in rows:
            version = row["model_version"]
            self.conn.execute(
                "UPDATE model_registry SET stage = ? WHERE model_version = ?",
                (Stage.ARCHIVED, version),
            )
            self._log_transition(
                version,
                Stage.PRODUCTION,
                Stage.ARCHIVED,
                f"superseded by {incoming}",
            )
            log.info("Archived incumbent %s", version)

    def _log_transition(
        self,
        model_version: str,
        from_stage: str | None,
        to_stage: str,
        reason: str = "",
        actor: str = "pipeline",
    ) -> None:
        self.conn.execute(
            "INSERT INTO registry_transitions (model_version, from_stage, "
            "to_stage, occurred_at, actor, reason) VALUES (?, ?, ?, ?, ?, ?)",
            (model_version, from_stage, to_stage, _now(), actor, reason),
        )

    # -- convenience ------------------------------------------------------

    def production(self, task: str | None = None):
        """The single live model, optionally filtered by task."""
        sql = "SELECT * FROM model_registry WHERE stage = ?"
        params = [Stage.PRODUCTION]
        if task:
            sql += " AND model_type LIKE ?"
            params.append(f"{task}:%")
        rows = self.conn.execute(sql, params).fetchall()

        if len(rows) > 1:
            raise RuntimeError(
                f"{len(rows)} models are in PRODUCTION for task {task!r}. "
                "Predictions written under this state have ambiguous lineage."
            )
        return rows[0] if rows else None

    def history(self, model_version: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                dict(r)
                for r in self.conn.execute(
                    "SELECT * FROM registry_transitions WHERE model_version = ? "
                    "ORDER BY occurred_at",
                    (model_version,),
                )
            ]
        )

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                dict(r)
                for r in self.conn.execute(
                    "SELECT model_version, model_type, stage, created_at, promoted_at, "
                    "training_end FROM model_registry ORDER BY created_at DESC"
                )
            ]
        )

    def promote_through_validation(
        self, model_version: str, validation_outcome, actor: str = "pipeline"
    ) -> str:
        """Walk a candidate through validation to production, or to FAILED.

        Wraps the whole sequence so the intermediate stages are always recorded.
        Skipping straight to PRODUCTION would lose the audit trail of *what*
        was validated.
        """
        self.transition(model_version, Stage.VALIDATING, "validation started", actor)

        if not validation_outcome.passed:
            self.transition(
                model_version,
                Stage.FAILED,
                f"failed: {', '.join(validation_outcome.failed_checks)}",
                actor,
            )
            return Stage.FAILED

        self.transition(model_version, Stage.VALIDATED, "all checks passed", actor)
        self.transition(model_version, Stage.STAGING, "staged for promotion", actor)
        self.transition(model_version, Stage.PRODUCTION, "promoted", actor)
        return Stage.PRODUCTION
