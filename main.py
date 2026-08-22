"""TradeMind AI command-line entrypoint.

    python main.py init       create the operational store
    python main.py ingest     fetch + validate market data  (Phase 1)
    python main.py features   build the feature panel       (Phase 2)
    python main.py validate   inspect folds, lock final test (Phase 3)
    python main.py train      fit base models + baselines   (Phase 4)
    python main.py ensemble   stack + calibrate             (Phase 5)
    python main.py thresholds cost feasibility + thresholds (Phase 6)
    python main.py backtest   event-based backtest          (Phase 7)
    python main.py monitor    drift + health + retrain gate (Phase 8)
    python main.py experiments leaderboard + registry state  (Phase 10)

    python main.py backfill   build history from scratch    (Phase 9)
    python main.py daily      run one trading day           (Phase 9)
    python main.py retrain    monitoring + retraining gate  (Phase 9)
"""

from __future__ import annotations

import argparse
import logging
import platform
import subprocess
import sys

from trademind.config import load_config
from trademind.logging_setup import setup_logging
from trademind.storage import PredictionStore, init_db

log = logging.getLogger("trademind.main")


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def cmd_init(cfg) -> int:
    db_path = cfg.data_root / "trademind.db"
    conn = init_db(db_path)
    tables = [
        r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    ]
    log.info("Store ready at %s", db_path)
    log.info("Tables: %s", ", ".join(tables))
    log.info("Universe: %d symbols | config hash: %s",
             len(cfg.universe), cfg.config_hash)
    conn.close()
    return 0


def cmd_ingest(cfg, args) -> int:
    """Fetch, validate, and persist market data for the configured universe."""
    from datetime import date

    from trademind.ingestion import MarketDataIngestion, YFinanceProvider
    from trademind.storage.lake import ParquetLake

    conn = init_db(cfg.data_root / "trademind.db")
    store = PredictionStore(conn)
    run_id = store.start_run(
        mode="backfill" if not args.incremental else "daily",
        config_hash=cfg.config_hash,
        git_commit=git_commit(),
        python_version=platform.python_version(),
    )

    try:
        ingestion = MarketDataIngestion(
            provider=YFinanceProvider(),
            lake=ParquetLake(cfg.data_root),
            store=store,
        )
        if ingestion.calendar.is_approximate:
            log.warning(
                "Using the approximate weekday calendar. Install "
                "pandas-market-calendars before trusting missing-session findings."
            )

        summary = ingestion.ingest_universe(
            symbols=cfg.universe,
            start=cfg.history_start,
            end=date.today(),
            run_id=run_id,
            incremental=args.incremental,
        )
        status = "SUCCESS" if not summary.failed else "FAILED"
        store.finish_run(run_id, status)
        return 0 if status == "SUCCESS" else 1

    except Exception as exc:
        store.finish_run(run_id, "FAILED", error=str(exc))
        log.exception("Ingestion run %s failed", run_id)
        return 1
    finally:
        conn.close()


def cmd_features(cfg, args) -> int:
    """Build the feature panel from stored market data."""
    from trademind.features import build_panel, coverage_report, feature_columns
    from trademind.storage.lake import ParquetLake

    lake = ParquetLake(cfg.data_root)
    symbols = lake.symbols()
    if not symbols:
        log.error("No market data found. Run: python main.py ingest")
        return 1

    frames = {s: lake.read_raw(s) for s in symbols}
    frames = {k: v for k, v in frames.items() if v is not None and not v.empty}

    panel = build_panel(
        frames, horizon=cfg.get("features.target_horizon_days"), cross_sectional=True
    )
    if panel.empty:
        log.error("Feature build produced an empty panel")
        return 1

    version = cfg.feature_version
    for symbol, group in panel.groupby("symbol"):
        lake.write_features(symbol, version, group.reset_index(drop=True))

    log.info("Wrote features version %s for %d symbols", version, panel["symbol"].nunique())

    worst = coverage_report(panel).head(5)
    log.info("Feature count: %d", len(feature_columns(panel)))
    log.info("Worst coverage:\n%s", worst.to_string(index=False))
    return 0


def cmd_validate(cfg, args) -> int:
    """Show the fold structure and lock the final-test window."""
    from trademind.features import feature_columns
    from trademind.storage.lake import ParquetLake
    from trademind.validation import (
        FinalTestLock, GapConfig, PurgedWalkForwardSplit, split_development,
    )

    lake = ParquetLake(cfg.data_root)
    panel = lake.read_all_features(cfg.feature_version)
    if panel.empty:
        log.error("No features found. Run: python main.py features")
        return 1

    lock_path = cfg.data_root / "final_test_lock.json"
    if lock_path.exists():
        lock = FinalTestLock.load(lock_path)
        log.info("Existing final-test lock: %s (%d rows, created %s)",
                 lock.fingerprint, lock.n_rows, lock.created_at)
    else:
        lock = FinalTestLock.create(panel, cfg.final_test_start)
        lock.save(lock_path)
        log.info("Locked final-test window from %s: %d rows, fingerprint %s",
                 cfg.final_test_start, lock.n_rows, lock.fingerprint)

    dev = split_development(panel, cfg.final_test_start)
    log.info("Development set: %d rows | %s..%s | %d features",
             len(dev), *dev.date_range, len(feature_columns(dev.panel)))

    splitter = PurgedWalkForwardSplit(
        n_splits=args.folds,
        test_sessions=args.test_sessions,
        gaps=GapConfig.from_config(cfg),
    )
    log.info("Fold structure:\n%s", splitter.summary(dev.panel).to_string(index=False))
    return 0


def cmd_train(cfg, args) -> int:
    """Fit base models across purged folds and report against baselines."""
    import numpy as np
    import pandas as pd

    from trademind.features import TRADEABLE_LABEL, drop_unlabelled, feature_columns
    from trademind.models import (
        ModelRegistry, compare_models, generate_oof,
        hgb_direction, hgb_return, logistic_direction, ridge_return,
    )
    from trademind.storage.lake import ParquetLake
    from trademind.validation import GapConfig, PurgedWalkForwardSplit, split_development

    lake = ParquetLake(cfg.data_root)
    panel = lake.read_all_features(cfg.feature_version)
    if panel.empty:
        log.error("No features found. Run: python main.py features")
        return 1

    dev = split_development(panel, cfg.final_test_start)
    data = drop_unlabelled(dev.panel)
    feats = feature_columns(data)

    splitter = PurgedWalkForwardSplit(
        n_splits=args.folds, test_sessions=args.test_sessions,
        gaps=GapConfig.from_config(cfg),
    )

    class _Constant:
        """Baseline: predicts the training mean of the label."""
        def __init__(self):
            self.spec = type("S", (), {"name": "baseline"})()
        def fit(self, X, y, dates=None):
            self.v = float(pd.Series(y).dropna().mean()); return self
        def predict(self, X):
            return np.full(len(X), self.v)

    registry = ModelRegistry(cfg.root / "models")

    for task, label, factories in (
        ("direction", "tradeable_direction_1d",
         [("baseline_majority", _Constant),
          ("direction_logistic", logistic_direction),
          ("direction_hgb", hgb_direction)]),
        ("return", TRADEABLE_LABEL,
         [("baseline_mean", _Constant),
          ("return_ridge", ridge_return),
          ("return_hgb", hgb_return)]),
    ):
        log.info("=== %s models ===", task)
        results = []
        for name, factory in factories:
            log.info("%s", name)
            results.append(generate_oof(
                data, factory, feats, label, task, splitter, model_name=name
            ))
            if not name.startswith("baseline"):
                model = factory().fit(data[feats], data[label], data["date"])
                registry.save(model, extra={"oof_metrics": results[-1].pooled_metrics})

        log.info("\n%s", compare_models(results).to_string(index=False))

    log.info("Registered models:\n%s", registry.summary().to_string(index=False))
    return 0


def cmd_ensemble(cfg, args) -> int:
    """Stack the base models and calibrate the result, both out-of-fold."""
    from trademind.ensemble import build_ensemble
    from trademind.features import drop_unlabelled, feature_columns
    from trademind.models import generate_oof, hgb_direction, logistic_direction
    from trademind.storage.lake import ParquetLake
    from trademind.validation import GapConfig, PurgedWalkForwardSplit, split_development

    lake = ParquetLake(cfg.data_root)
    panel = lake.read_all_features(cfg.feature_version)
    if panel.empty:
        log.error("No features found. Run: python main.py features")
        return 1

    data = drop_unlabelled(split_development(panel, cfg.final_test_start).panel)
    feats = feature_columns(data)
    splitter = PurgedWalkForwardSplit(
        n_splits=args.folds, test_sessions=args.test_sessions,
        gaps=GapConfig.from_config(cfg),
    )

    base = {
        name: generate_oof(data, factory, feats, "tradeable_direction_1d",
                           "direction", splitter, name)
        for name, factory in (("logistic", logistic_direction),
                              ("hgb", hgb_direction))
    }

    context_cols = [c for c in ("trend_regime", "vol_percentile_expanding")
                    if c in data.columns]
    ensemble = build_ensemble(
        base, context=data[["date", "symbol"] + context_cols],
        context_cols=context_cols, splitter=splitter,
    )

    log.info("\n%s", ensemble.render())

    out = cfg.data_root / "predictions" / "calibrated_oof.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    ensemble.signals.to_parquet(out, index=False)
    log.info("Calibrated OOF signals -> %s (%d rows)", out, len(ensemble.signals))
    return 0


def cmd_thresholds(cfg, args) -> int:
    """Check cost feasibility, then derive decision thresholds out-of-fold."""
    import pandas as pd

    from trademind.decision import (
        CostModel, cost_feasibility, optimise_thresholds_out_of_fold,
    )

    costs = CostModel.from_config(cfg)

    signals_path = cfg.data_root / "predictions" / "calibrated_oof.parquet"
    if not signals_path.exists():
        log.error("No calibrated signals. Run: python main.py ensemble")
        return 1
    signals = pd.read_parquet(signals_path)

    observed_ic = None
    if {"calibrated", "y_true"} <= set(signals.columns):
        observed_ic = float(
            signals["calibrated"].corr(signals["y_true"], method="spearman")
        )

    horizon = cfg.get("features.target_horizon_days")
    report = cost_feasibility(costs, holding_days=horizon, observed_ic=observed_ic)
    log.info("\n%s", report.render())

    if not report.feasible:
        log.warning(
            "Deriving thresholds anyway so the numbers are on record, but no "
            "threshold makes an infeasible strategy profitable."
        )

    thresholds, stability = optimise_thresholds_out_of_fold(signals, costs)
    log.info("\nthresholds: %s", thresholds.as_dict())
    log.info("\nblock stability:\n%s", stability.to_string(index=False))
    log.info(
        "\nWrite these into config/config.yaml under `decision:` and treat "
        "them as locked."
    )
    return 0


def cmd_backtest(cfg, args) -> int:
    """Run the event-based backtest on development data, with benchmarks."""
    import pandas as pd

    from trademind.backtesting import (
        BacktestConfig, buy_and_hold, equal_weight_rebalanced,
        performance_metrics, render_report, run_backtest,
    )
    from trademind.decision import DecisionEngine, RiskEngine, Thresholds
    from trademind.storage.lake import ParquetLake
    from trademind.validation import split_development

    buy = cfg.get("decision.buy_threshold", None)
    if buy is None:
        log.error(
            "decision.buy_threshold is null. Run `python main.py thresholds` "
            "and write the derived values into config/config.yaml first."
        )
        return 1

    lake = ParquetLake(cfg.data_root)
    panel = lake.read_all_features(cfg.feature_version)
    signals_path = cfg.data_root / "predictions" / "calibrated_oof.parquet"
    if panel.empty or not signals_path.exists():
        log.error("Need features and calibrated signals first.")
        return 1

    dev = split_development(panel, cfg.final_test_start).panel
    signals = pd.read_parquet(signals_path)
    bt_config = BacktestConfig.from_config(cfg)

    engine = DecisionEngine(
        Thresholds(
            buy=buy,
            sell=cfg.get("decision.sell_threshold"),
            min_expected_return=cfg.get("decision.min_expected_return"),
        ),
        RiskEngine(max_positions=cfg.get("decision.max_positions", 10)),
        sizing_method=cfg.get("decision.sizing_method", "fixed"),
        max_weight=bt_config.max_position_weight,
        reference_volatility=cfg.get("decision.reference_volatility", 0.25),
    )

    result = run_backtest(dev, signals, engine, bt_config)
    metrics = performance_metrics(
        result.equity_curve, result.trades, bt_config.initial_capital
    )

    bars = {s: g.reset_index(drop=True) for s, g in dev.groupby("symbol")}
    sessions = sorted(pd.Timestamp(d).date() for d in dev["date"].unique())
    benchmarks = {
        "buy_and_hold": performance_metrics(
            buy_and_hold(bars, sessions, bt_config),
            initial_capital=bt_config.initial_capital),
        "equal_weight_monthly": performance_metrics(
            equal_weight_rebalanced(bars, sessions, bt_config),
            initial_capital=bt_config.initial_capital),
    }

    log.info("\n%s", render_report(metrics, benchmarks))
    log.info("Max reconciliation error: %.8f", result.max_reconciliation_error)

    out = cfg.root / "reports"
    out.mkdir(parents=True, exist_ok=True)
    result.equity_curve.to_csv(out / "equity_curve.csv", index=False)
    result.trades.to_csv(out / "trades.csv", index=False)
    log.info("Wrote equity curve and trades to %s", out)
    return 0


def cmd_monitor(cfg, args) -> int:
    """Run the monitoring pipeline and the retraining governance policy."""
    import pandas as pd

    from trademind.features import feature_columns
    from trademind.monitoring import run_monitoring
    from trademind.storage import PredictionStore, init_db
    from trademind.storage.lake import ParquetLake

    conn = init_db(cfg.data_root / "trademind.db")
    store = PredictionStore(conn)

    resolved = pd.DataFrame([dict(r) for r in store.resolved()])
    lake = ParquetLake(cfg.data_root)
    panel = lake.read_all_features(cfg.feature_version)

    if panel.empty:
        log.error("No features found. Run: python main.py features")
        conn.close()
        return 1

    feats = feature_columns(panel)
    dates = pd.to_datetime(panel["date"])
    split_at = dates.quantile(0.75)
    reference = panel[dates < split_at]
    current = panel[dates >= split_at]

    report = run_monitoring(
        resolved=resolved,
        reference_features=reference,
        current_features=current,
        feature_columns=feats,
        baseline_auc=args.baseline_auc,
        baseline_ece=args.baseline_ece,
        store=store,
    )
    log.info("\n%s", report.render())
    conn.close()
    return 0


def cmd_experiments(cfg, args) -> int:
    """Show the experiment leaderboard and registry state."""
    from trademind.experiments import (
        ExperimentTracker, ModelLifecycle, compare_experiments, render_comparison,
    )
    from trademind.storage import init_db

    conn = init_db(cfg.data_root / "trademind.db")
    tracker = ExperimentTracker(conn)
    lifecycle = ModelLifecycle(conn)

    for task in ("direction", "return", "ensemble"):
        runs = tracker.all(task=task)
        if not runs:
            continue
        log.info("\n=== %s ===", task)
        log.info("%s", tracker.summary(task=task))
        log.info("\n%s", render_comparison(compare_experiments(runs)))

    if tracker.count() == 0:
        log.info("No experiments logged yet.")

    registry = lifecycle.summary()
    if not registry.empty:
        log.info("\n=== model registry ===\n%s", registry.to_string(index=False))
    else:
        log.info("No models registered yet.")

    conn.close()
    return 0


def cmd_pipeline(cfg, args) -> int:
    """Run the orchestrated pipeline: backfill | daily | retrain."""
    from datetime import date

    from trademind.orchestration import run_pipeline, should_run_today

    as_of = date.fromisoformat(args.date) if args.date else date.today()

    if args.command == "daily" and not args.force:
        ok, reason = should_run_today(as_of)
        log.info("%s", reason)
        if not ok:
            return 0

    run = run_pipeline(cfg, mode=args.command, as_of=as_of)
    log.info("\n%s", run.render())
    return 0 if run.ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trademind")
    parser.add_argument(
        "command",
        choices=["init", "ingest", "features", "validate", "train",
                 "ensemble", "thresholds", "backtest", "monitor", "experiments",
                 "backfill", "daily", "retrain"],
    )
    parser.add_argument(
        "--incremental", action="store_true",
        help="fetch only new bars since the last stored date",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-sessions", type=int, default=126)
    parser.add_argument("--date", default=None,
                        help="run as of this ISO date instead of today")
    parser.add_argument("--force", action="store_true",
                        help="run even on a non-trading day")
    parser.add_argument("--baseline-auc", type=float, default=0.52)
    parser.add_argument("--baseline-ece", type=float, default=0.05)
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    cfg = load_config()
    setup_logging(
        level=args.log_level or cfg.get("logging.level", "INFO"),
        log_dir=cfg.root / "logs",
    )

    log.info("TradeMind AI | python %s | git %s",
             platform.python_version(), git_commit() or "n/a")

    if args.command == "init":
        return cmd_init(cfg)
    if args.command == "ingest":
        return cmd_ingest(cfg, args)
    if args.command == "features":
        return cmd_features(cfg, args)
    if args.command == "validate":
        return cmd_validate(cfg, args)
    if args.command == "train":
        return cmd_train(cfg, args)
    if args.command == "ensemble":
        return cmd_ensemble(cfg, args)
    if args.command == "thresholds":
        return cmd_thresholds(cfg, args)
    if args.command == "backtest":
        return cmd_backtest(cfg, args)
    if args.command == "monitor":
        return cmd_monitor(cfg, args)
    if args.command == "experiments":
        return cmd_experiments(cfg, args)
    return cmd_pipeline(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
