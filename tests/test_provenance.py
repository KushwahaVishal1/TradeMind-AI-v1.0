"""Provenance capture tests."""

from __future__ import annotations

from trademind.provenance import Provenance, capture, package_versions


def test_capture_records_the_environment():
    p = capture(config_hash="abc123")
    assert p.python_version
    assert p.config_hash == "abc123"
    assert p.environment_hash


def test_environment_hash_is_stable_within_a_process():
    assert capture().environment_hash == capture().environment_hash


def test_environment_hash_changes_with_package_versions():
    a = Provenance(packages={"numpy": "1.0"})
    b = Provenance(packages={"numpy": "2.0"})
    assert a.environment_hash != b.environment_hash


def test_a_dirty_tree_is_not_reproducible():
    """A commit recorded against uncommitted changes points at code that never ran."""
    p = Provenance(git={"commit": "abc", "short_commit": "abc",
                        "dirty": True, "dirty_files": 3})
    assert not p.reproducible
    assert any("uncommitted" in w for w in p.warnings())


def test_a_clean_tree_is_reproducible():
    p = Provenance(git={"commit": "abc", "short_commit": "abc", "dirty": False})
    assert p.reproducible
    assert not any("uncommitted" in w for w in p.warnings())


def test_missing_commit_is_a_warning():
    p = Provenance(git={"commit": None, "dirty": False})
    assert not p.reproducible
    assert any("cannot be traced" in w for w in p.warnings())


def test_tracked_packages_are_reported():
    versions = package_versions()
    assert "numpy" in versions and "pandas" in versions


def test_missing_packages_are_flagged_not_hidden():
    p = Provenance(packages={"numpy": "1.0", "duckdb": "not installed"},
                   git={"commit": "a", "dirty": False})
    assert any("duckdb" in w for w in p.warnings())


def test_serialises_to_json():
    import json

    data = json.loads(capture().to_json())
    assert "environment_hash" in data and "warnings" in data


def test_render_includes_the_environment_hash():
    assert "environment hash" in capture().render()


# =====================================================================
# The static gate must itself work
# =====================================================================

def test_codebase_has_no_forbidden_patterns():
    """Runs the CI gate in-process, so it fails locally before it fails in CI."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "check_forbidden_patterns.py")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout


def test_the_forbidden_pattern_checker_can_fail(tmp_path):
    """A gate that cannot fire is not a gate."""
    import ast
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import check_forbidden_patterns as checker

    source = "def f(s):\n    return s.rank(pct=True)\n"
    violations = checker.check_full_sample_statistics(
        tmp_path / "planted.py", ast.parse(source)
    )
    assert violations and violations[0].rule == "full-sample-statistic"


def test_the_checker_permits_grouped_ranks():
    """Ranking within a date is safe; the gate must not block it."""
    import ast
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import check_forbidden_patterns as checker

    source = 'def f(df):\n    return df.groupby("date")["x"].rank(pct=True)\n'
    assert not checker.check_full_sample_statistics(
        Path("x.py"), ast.parse(source)
    )


# =====================================================================
# The final evaluation must be single-use
# =====================================================================

def test_final_evaluation_refuses_without_confirmation():
    """Unlocking the test set must be a deliberate act, not a default."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "final_evaluation.py")],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "--confirm" in result.stderr + result.stdout


def test_no_fabricated_final_test_result_exists():
    """The report's Results section must stay empty until the run happens.

    Guards against the failure this project was built to prevent: the original
    planning document recorded a 'locked final-test result' of +0.48% at Sharpe
    1.45 for code that had never been written.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    evidence = root / "reports" / "final" / "FINAL_TEST_EXECUTED.json"
    report = root / "reports" / "final" / "TradeMind_AI_Final_Report.md"

    if evidence.exists():
        return   # the evaluation has legitimately been run

    text = report.read_text(encoding="utf-8")
    assert "LOCKED. Not evaluated." in text, (
        "The final report claims a result without an evidence file."
    )
    # The discarded placeholder numbers may appear only as a cautionary tale.
    # Checked against a surrounding window rather than a single line, because
    # prose wraps and the disclaimer often sits on the previous line.
    lines = text.splitlines()
    markers = ("placeholder", "no such code", "recorded a", "never been written",
               "fabricat")

    for fabricated in ("+0.48%", "Sharpe 1.45"):
        for i, line in enumerate(lines):
            if fabricated not in line:
                continue
            context = " ".join(lines[max(0, i - 3): i + 4]).lower()
            assert any(m in context for m in markers), (
                f"unexplained fabricated figure near line {i + 1}: {line.strip()}"
            )
