"""Provenance: everything needed to reproduce a run.

Captured on every pipeline execution and stamped onto artifacts. A result
without provenance is an anecdote — you can report it, but nobody, including
you in six months, can check it.

``environment_hash`` covers the pieces that change results silently: package
versions, the Python build, the platform. Two runs with the same environment
hash and the same config hash should produce identical output; when they do
not, the pipeline is non-deterministic and every reproducibility claim in the
report is false.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

TRACKED_PACKAGES = (
    "numpy", "pandas", "scikit-learn", "scipy", "pyarrow", "duckdb", "yfinance",
)


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def git_state() -> dict:
    """Commit, branch, and whether the tree was clean.

    ``dirty`` matters more than the commit. A hash recorded against
    uncommitted changes points at code that was never what ran, which is worse
    than recording nothing — it looks authoritative.
    """
    status = _run(["git", "status", "--porcelain"])
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "short_commit": _run(["git", "rev-parse", "--short", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status),
        "dirty_files": len(status.splitlines()) if status else 0,
    }


def package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not installed"
    return out


@dataclass
class Provenance:
    """A complete record of the conditions a run executed under."""

    captured_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    python_version: str = field(default_factory=platform.python_version)
    python_implementation: str = field(default_factory=platform.python_implementation)
    platform_name: str = field(default_factory=platform.platform)
    processor: str = field(default_factory=platform.machine)
    packages: dict[str, str] = field(default_factory=package_versions)
    git: dict = field(default_factory=git_state)
    config_hash: str | None = None
    random_seed: int = 42
    environment_variables: dict[str, str] = field(default_factory=dict)

    @property
    def environment_hash(self) -> str:
        """Hash of everything that could silently change results."""
        payload = json.dumps({
            "python": self.python_version,
            "implementation": self.python_implementation,
            "packages": self.packages,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @property
    def reproducible(self) -> bool:
        return bool(self.git.get("commit")) and not self.git.get("dirty")

    def warnings(self) -> list[str]:
        out = []
        if not self.git.get("commit"):
            out.append("No git commit — this run cannot be traced to code.")
        if self.git.get("dirty"):
            out.append(
                f"Working tree had {self.git['dirty_files']} uncommitted "
                f"change(s); commit {self.git.get('short_commit')} is not what ran."
            )
        missing = [k for k, v in self.packages.items() if v == "not installed"]
        if missing:
            out.append(f"Not installed: {', '.join(missing)}")
        return out

    def to_dict(self) -> dict:
        data = asdict(self)
        data["environment_hash"] = self.environment_hash
        data["reproducible"] = self.reproducible
        data["warnings"] = self.warnings()
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def render(self) -> str:
        lines = [
            f"python {self.python_version} ({self.python_implementation}) "
            f"on {self.platform_name}",
            f"git {self.git.get('short_commit') or 'n/a'}"
            f"{' [DIRTY]' if self.git.get('dirty') else ''} "
            f"branch {self.git.get('branch') or 'n/a'}",
            f"environment hash {self.environment_hash}",
            "packages: " + ", ".join(
                f"{k}=={v}" for k, v in self.packages.items()
                if v != "not installed"
            ),
        ]
        for w in self.warnings():
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def capture(config_hash: str | None = None, seed: int = 42) -> Provenance:
    """Capture provenance for the current process."""
    return Provenance(
        config_hash=config_hash,
        random_seed=seed,
        environment_variables={
            k: v for k, v in os.environ.items()
            if k.startswith(("TRADEMIND_", "PYTHONHASHSEED"))
        },
    )


def set_deterministic_seeds(seed: int = 42) -> None:
    """Seed every source of randomness the pipeline touches.

    ``PYTHONHASHSEED`` cannot be set from inside a running process — it is read
    at interpreter startup. Set it in the environment (the Dockerfile and CI
    workflow both do) or set-dependent iteration order can still vary.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    if os.environ.get("PYTHONHASHSEED") not in (str(seed), "0"):
        import warnings

        warnings.warn(
            f"PYTHONHASHSEED is {os.environ.get('PYTHONHASHSEED')!r}; set it to "
            f"{seed} before starting Python for full determinism.",
            RuntimeWarning, stacklevel=2,
        )
