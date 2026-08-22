"""Loading dashboard state, with graceful degradation.

Every loader returns None or an empty frame when its source is missing, rather
than raising. A dashboard that crashes because the backtest has not been run
yet is useless precisely when it would be most useful -- during setup, when you
want to see how far the pipeline got.

The panels then render "not yet computed" instead of a number, which is also
more honest than a zero.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def _safe(fn, what: str, default=None):
    try:
        return fn()
    except Exception as exc:
        log.debug("Could not load %s: %s", what, exc)
        return default


def load_equity_curve(root: Path) -> pd.DataFrame:
    path = Path(root) / "reports" / "equity_curve.csv"
    return _safe(lambda: pd.read_csv(path), "equity curve", pd.DataFrame())


def load_trades(root: Path) -> pd.DataFrame:
    path = Path(root) / "reports" / "trades.csv"
    return _safe(lambda: pd.read_csv(path), "trades", pd.DataFrame())


def load_signals(data_root: Path) -> pd.DataFrame:
    path = Path(data_root) / "predictions" / "calibrated_oof.parquet"
    return _safe(lambda: pd.read_parquet(path), "signals", pd.DataFrame())


def load_predictions(store, limit: int = 500) -> pd.DataFrame:
    def _read():
        rows = store.conn.execute(
            "SELECT * FROM predictions ORDER BY prediction_date DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    return _safe(_read, "predictions", pd.DataFrame())


def load_resolved(store) -> pd.DataFrame:
    return _safe(
        lambda: pd.DataFrame([dict(r) for r in store.resolved()]),
        "resolved predictions",
        pd.DataFrame(),
    )


def load_runs(store, limit: int = 50) -> pd.DataFrame:
    def _read():
        rows = store.conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    return _safe(_read, "runs", pd.DataFrame())


def count_data_quality(store) -> tuple[int, int]:
    def _read():
        rows = store.conn.execute(
            "SELECT severity, COUNT(*) AS n FROM data_quality_issues GROUP BY severity"
        ).fetchall()
        counts = {r["severity"]: r["n"] for r in rows}
        return counts.get("ERROR", 0), counts.get("WARNING", 0)

    return _safe(_read, "data quality", (0, 0))


def build_dashboard_state(cfg, store=None) -> dict:
    """Assemble everything the overview needs, tolerating missing pieces."""
    from ..backtesting.report import performance_metrics
    from ..decision import CostModel, cost_feasibility
    from ..models.metrics import classification_metrics

    state: dict = {}

    costs = _safe(lambda: CostModel.from_config(cfg), "cost model")
    if costs is not None:
        holding = _safe(lambda: cfg.get("features.target_horizon_days"), "horizon", 1) or 1
        state["round_trip_cost"] = costs.round_trip
        state["holding_days"] = holding

    signals = load_signals(cfg.data_root)
    if not signals.empty and {"calibrated", "y_true"} <= set(signals.columns):
        ic = _safe(
            lambda: float(signals["calibrated"].corr(signals["y_true"], method="spearman")),
            "observed IC",
        )
        state["observed_ic"] = ic

        metrics = _safe(
            lambda: classification_metrics(
                (signals["y_true"] > 0).astype(float), signals["calibrated"]
            ),
            "classification metrics",
            {},
        )
        state.update(
            {
                "auc": metrics.get("roc_auc"),
                "auc_stderr": metrics.get("auc_stderr"),
                "accuracy": metrics.get("accuracy"),
                "majority_accuracy": metrics.get("majority_accuracy"),
            }
        )

    if costs is not None:
        report = _safe(
            lambda: cost_feasibility(
                costs,
                holding_days=state.get("holding_days", 1),
                observed_ic=state.get("observed_ic"),
            ),
            "feasibility",
        )
        if report is not None:
            state["breakeven_ic"] = report.breakeven_ic

    curve = load_equity_curve(cfg.root)
    trades = load_trades(cfg.root)
    if not curve.empty:
        state["performance"] = _safe(
            lambda: performance_metrics(
                curve,
                trades if not trades.empty else None,
                _safe(lambda: cfg.get("backtest.initial_capital"), "capital"),
            ),
            "performance metrics",
            {},
        )
        state["equity_curve"] = curve
        state["trades"] = trades

    if store is not None:
        errors, warnings = count_data_quality(store)
        state["n_dq_errors"] = errors
        state["n_dq_warnings"] = warnings
        state["predictions"] = load_predictions(store)
        state["resolved"] = load_resolved(store)
        state["runs"] = load_runs(store)

    return state
