"""Prediction lifecycle: PREDICTED -> RESOLVED.

Resolution is where a stored prediction meets what the market did. Two rules,
both enforced in the storage layer already and re-asserted here:

- A prediction is resolved exactly once. Outcomes are historical facts.
- The outcome comes from the *tradeable* return -- open[t+2]/open[t+1] - 1 --
  the same label the model was trained on. Resolving against a close-to-close
  return would score the model on a target it never optimised for, and the
  measured degradation would be an artifact of that mismatch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class ResolutionSummary:
    n_pending: int
    n_resolved: int
    n_unresolvable: int
    oldest_pending: date | None = None

    def render(self) -> str:
        parts = [f"resolved {self.n_resolved}", f"pending {self.n_pending}"]
        if self.n_unresolvable:
            parts.append(f"unresolvable {self.n_unresolvable}")
        if self.oldest_pending:
            parts.append(f"oldest pending {self.oldest_pending}")
        return " | ".join(parts)


def resolve_pending(
    store,
    outcomes: pd.DataFrame,
    as_of: date | None = None,
    max_pending_sessions: int = 10,
) -> ResolutionSummary:
    """Attach realised outcomes to predictions whose windows have closed.

    ``outcomes`` needs ``symbol``, ``date`` (the prediction date) and
    ``actual_return`` measured on the tradeable label.
    """
    pending = store.pending(as_of.isoformat() if as_of else None)
    if not pending:
        return ResolutionSummary(0, 0, 0)

    lookup = {
        (r.symbol, pd.Timestamp(r.date).date()): float(r.actual_return)
        for r in outcomes.itertuples()
        if np.isfinite(r.actual_return)
    }

    resolved = unresolvable = 0
    still_pending = []

    for row in pending:
        key = (row["symbol"], date.fromisoformat(row["prediction_date"]))
        actual = lookup.get(key)

        if actual is None:
            still_pending.append(key[1])
            continue
        store.resolve(row["prediction_id"], actual)
        resolved += 1

    # A prediction whose outcome never arrives is a data-quality problem, not a
    # model problem. Surface it rather than letting the pending queue grow.
    if still_pending and as_of:
        stale = [d for d in still_pending if (as_of - d).days > max_pending_sessions * 2]
        unresolvable = len(stale)
        if stale:
            log.warning(
                "%d prediction(s) have been pending for over %d sessions; their "
                "outcomes may never arrive.",
                len(stale),
                max_pending_sessions,
            )

    return ResolutionSummary(
        n_pending=len(still_pending),
        n_resolved=resolved,
        n_unresolvable=unresolvable,
        oldest_pending=min(still_pending) if still_pending else None,
    )


def prediction_health(store, as_of: date | None = None) -> dict:
    """Coverage statistics over the stored prediction history."""
    rows = store.conn.execute(
        "SELECT status, COUNT(*) AS n FROM predictions GROUP BY status"
    ).fetchall()
    counts = {r["status"]: r["n"] for r in rows}

    total = sum(counts.values())
    return {
        "n_total": float(total),
        "n_predicted": float(counts.get("PREDICTED", 0)),
        "n_resolved": float(counts.get("RESOLVED", 0)),
        "n_void": float(counts.get("VOID", 0)),
        "resolution_rate": counts.get("RESOLVED", 0) / total if total else 0.0,
    }
