#!/usr/bin/env python3
"""Static checks for constructs this project has decided are never correct.

Runs in CI as a merge gate.

Uses the AST rather than grep. The grep version false-positives on every
docstring that *warns* about a pattern — and this codebase deliberately
documents each trap next to the guard against it, so a naive grep flags the
documentation and the gate gets disabled. Parsing means only real code counts.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "src" / "trademind" / "features"
SRC = ROOT / "src"


class Violation(NamedTuple):
    path: str
    line: int
    rule: str
    detail: str


def _display(path: Path) -> str:
    """Path relative to the repo when possible, absolute otherwise.

    `relative_to` raises for anything outside ROOT, which made the checker
    unusable on a path from a test fixture or an ad-hoc file.
    """
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _iter_python(root: Path):
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


def check_full_sample_statistics(path: Path, tree: ast.AST) -> list[Violation]:
    """`.rank(pct=True)` and friends outside a groupby see the whole series."""
    out = []
    banned = {"rank", "qcut"}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in banned:
            continue

        # A groupby anywhere in the receiver chain makes it within-date, safe.
        source = ast.unparse(func.value)
        if "groupby" in source:
            continue

        out.append(Violation(
            _display(path), node.lineno, "full-sample-statistic",
            f".{func.attr}() on '{source}' ranks against the entire series, "
            "future rows included. Use regime.expanding_percentile().",
        ))
    return out


def check_backward_fill(path: Path, tree: ast.AST) -> list[Violation]:
    """`bfill` pulls a future value backwards into the past."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("bfill", "backfill"):
                out.append(Violation(
                    _display(path), node.lineno, "backward-fill",
                    "bfill copies a future value into an earlier row.",
                ))
        # method="bfill" as a fillna keyword
        if isinstance(node, ast.keyword) and node.arg == "method":
            if isinstance(node.value, ast.Constant) and node.value.value in (
                "bfill", "backfill"
            ):
                out.append(Violation(
                    _display(path), node.value.lineno,
                    "backward-fill", "fillna(method='bfill') looks forward.",
                ))
    return out


def check_centered_windows(path: Path, tree: ast.AST) -> list[Violation]:
    """`rolling(center=True)` puts future observations in the current row."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "center":
            if isinstance(node.value, ast.Constant) and node.value.value is True:
                out.append(Violation(
                    _display(path), node.value.lineno,
                    "centered-window",
                    "center=True includes future observations in the window.",
                ))
    return out


def main() -> int:
    violations: list[Violation] = []

    for path in _iter_python(FEATURES):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        violations += check_full_sample_statistics(path, tree)
        violations += check_backward_fill(path, tree)
        violations += check_centered_windows(path, tree)

    if not violations:
        print("No forbidden patterns found.")
        return 0

    for v in violations:
        print(f"::error file={v.path},line={v.line}::[{v.rule}] {v.detail}")
    print(f"\n{len(violations)} violation(s).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
