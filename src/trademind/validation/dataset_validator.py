"""The final-test lock.

The roadmap says the final test period "must never be used for optimization."
Written as prose, that is a promise. Promises decay: it is three months later,
the numbers are disappointing, one threshold gets nudged, and nothing in the
codebase objects.

This module turns the promise into a mechanism.

``split_development`` is the *only* supported way to obtain data for fitting,
tuning, or model selection, and it structurally cannot return a row from the
locked window. Everything downstream — the splitter, the tuner, the threshold
optimiser — consumes its output. There is no ergonomic path to the final test
set that does not go through ``FinalTestLock.unlock``, which logs loudly, is
one-shot, and records the access.

The lock cannot stop a determined person from reading the parquet files
directly. It is not a security boundary. It makes the wrong thing require
deliberate effort and leave a trace, which is the realistic goal.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


class FinalTestViolation(RuntimeError):
    """Something attempted to use locked final-test data for optimisation."""


@dataclass
class DevelopmentData:
    """Data cleared for fitting, tuning, and selection.

    Carries its own boundary so any consumer can re-assert the invariant rather
    than assuming the caller got it right.
    """

    panel: pd.DataFrame
    final_test_start: pd.Timestamp
    date_col: str = "date"

    def __post_init__(self) -> None:
        if self.panel.empty:
            return
        latest = pd.to_datetime(self.panel[self.date_col]).max()
        if latest >= self.final_test_start:
            raise FinalTestViolation(
                f"Development data extends to {latest.date()}, at or beyond the "
                f"locked boundary {self.final_test_start.date()}."
            )

    def __len__(self) -> int:
        return len(self.panel)

    @property
    def date_range(self) -> tuple[date, date]:
        dates = pd.to_datetime(self.panel[self.date_col])
        return dates.min().date(), dates.max().date()


def split_development(
    panel: pd.DataFrame,
    final_test_start: date | pd.Timestamp,
    date_col: str = "date",
    label_lag_sessions: int = 2,
) -> DevelopmentData:
    """Return only the rows that may be optimised against.

    ``label_lag_sessions`` trims an extra gap at the boundary. A development row
    close to the cutoff carries a label computed from prices inside the final
    test window — so keeping it would leak the locked period into training via
    the label, exactly the mechanism ``embargo.py`` describes at fold
    boundaries. The same fix applies here and matters more, because this
    boundary is crossed once and never checked again.
    """
    boundary = pd.Timestamp(final_test_start)
    dates = pd.to_datetime(panel[date_col])

    sessions = pd.Index(sorted(dates.unique()))
    pos = sessions.searchsorted(boundary, side="left")
    cut_pos = max(0, pos - label_lag_sessions)
    cutoff = sessions[cut_pos] if len(sessions) else boundary

    dev = panel[dates < cutoff].reset_index(drop=True)

    n_trimmed = int(((dates >= cutoff) & (dates < boundary)).sum())
    if n_trimmed:
        log.info(
            "Trimmed %d row(s) before the locked boundary; their labels reach "
            "into the final-test window.",
            n_trimmed,
        )

    log.info(
        "Development set: %d rows through %s (locked window opens %s)",
        len(dev),
        dev[date_col].max().date() if len(dev) else "n/a",
        boundary.date(),
    )
    return DevelopmentData(dev, boundary, date_col)


@dataclass
class FinalTestLock:
    """Custodian of the locked evaluation window.

    Records a fingerprint of the locked data at creation. If the fingerprint
    changes between the lock being created and the final evaluation running,
    something re-ingested or re-engineered the test period, and any comparison
    against a previously recorded benchmark is invalid.
    """

    final_test_start: pd.Timestamp
    fingerprint: str
    n_rows: int
    created_at: str
    accesses: list[dict] = field(default_factory=list)
    _unlocked: bool = False

    @classmethod
    def create(
        cls,
        panel: pd.DataFrame,
        final_test_start: date | pd.Timestamp,
        date_col: str = "date",
    ) -> FinalTestLock:
        boundary = pd.Timestamp(final_test_start)
        locked = panel[pd.to_datetime(panel[date_col]) >= boundary]
        return cls(
            final_test_start=boundary,
            fingerprint=_fingerprint(locked, date_col),
            n_rows=len(locked),
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )

    def unlock(
        self,
        panel: pd.DataFrame,
        reason: str,
        date_col: str = "date",
    ) -> pd.DataFrame:
        """Release the final-test data. **Once, for evaluation only.**

        Verifies the data has not changed since the lock was created, refuses a
        second unlock in the same session, and records the access.
        """
        if self._unlocked:
            raise FinalTestViolation(
                "The final test has already been unlocked in this session. A "
                "second evaluation means the first one informed a change, which "
                "is optimisation against the test set."
            )

        boundary = self.final_test_start
        locked = panel[pd.to_datetime(panel[date_col]) >= boundary].reset_index(drop=True)

        current = _fingerprint(locked, date_col)
        if current != self.fingerprint:
            raise FinalTestViolation(
                "The final-test data has changed since the lock was created "
                f"(fingerprint {self.fingerprint} -> {current}). Any comparison "
                "against previously recorded benchmark results is invalid."
            )

        self._unlocked = True
        self.accesses.append(
            {
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
                "reason": reason,
                "n_rows": len(locked),
            }
        )
        log.warning(
            "FINAL TEST UNLOCKED (%d rows, from %s). Reason: %s. "
            "Nothing may be tuned after this point.",
            len(locked),
            boundary.date(),
            reason,
        )
        return locked

    # -- persistence ------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(
            {
                "final_test_start": self.final_test_start.isoformat(),
                "fingerprint": self.fingerprint,
                "n_rows": self.n_rows,
                "created_at": self.created_at,
                "accesses": self.accesses,
            },
            indent=2,
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> FinalTestLock:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            final_test_start=pd.Timestamp(data["final_test_start"]),
            fingerprint=data["fingerprint"],
            n_rows=data["n_rows"],
            created_at=data["created_at"],
            accesses=data.get("accesses", []),
        )


def _fingerprint(df: pd.DataFrame, date_col: str) -> str:
    """Content hash of the locked window.

    Built from shape, date bounds, and the checksum of the numeric columns —
    enough to detect re-ingestion or a feature-definition change, cheap enough
    to compute on every run.
    """
    if df.empty:
        return "empty"

    numeric = df.select_dtypes("number")
    checksum = float(numeric.to_numpy(dtype="float64", na_value=0.0).sum())
    payload = "|".join(
        [
            str(len(df)),
            str(len(df.columns)),
            str(pd.to_datetime(df[date_col]).min()),
            str(pd.to_datetime(df[date_col]).max()),
            f"{checksum:.6f}",
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def assert_no_locked_data(
    df: pd.DataFrame,
    final_test_start: date | pd.Timestamp,
    context: str,
    date_col: str = "date",
) -> None:
    """Guard for any function that fits, tunes, or selects.

    Cheap enough to call unconditionally. Call it.
    """
    boundary = pd.Timestamp(final_test_start)
    offending = df[pd.to_datetime(df[date_col]) >= boundary]
    if not offending.empty:
        raise FinalTestViolation(
            f"{context}: {len(offending)} row(s) from the locked window "
            f"(>= {boundary.date()}) reached code that fits or tunes. "
            "Use split_development()."
        )
