#!/usr/bin/env python3
"""The final evaluation. Runs once, ever.

    python scripts/final_evaluation.py --confirm

## What this does

Unlocks the final-test window that has been sealed since Phase 3, evaluates the
production model and the full decision-and-backtest stack on it, writes the
results to ``reports/final/``, and records that the unlock happened.

## Why it is a separate script with a confirmation flag

Because the temptation it guards against is real and arrives late. Twelve
phases of work produce a number, the number is disappointing, and the honest
next step — publishing it — competes with a dozen plausible-sounding
alternatives: a different threshold, one more feature, a longer training
window. Each is defensible in isolation. Together they are optimising against
the test set, one small step at a time.

Making the unlock a deliberate, logged, single-use act is what stops that. Not
because a script cannot be run twice — it can, and the guard would not stop
someone determined — but because doing so requires deleting an evidence file
whose entire purpose is to say this already happened.

## After running this

Nothing may be tuned. Not a threshold, not a hyperparameter, not the universe,
not the cost model. If the result is bad, the result is bad, and the report
says so. A system that reports a poor honest number is more credible than one
reporting an excellent number of unknown provenance — the entire value of the
twelve phases before this is that this number means something.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

log = logging.getLogger("final_evaluation")

# Written after a successful run. Its presence blocks a second one.
EVIDENCE_FILE = ROOT / "reports" / "final" / "FINAL_TEST_EXECUTED.json"


def already_run() -> dict | None:
    if EVIDENCE_FILE.exists():
        return json.loads(EVIDENCE_FILE.read_text(encoding="utf-8"))
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm", action="store_true",
        help="required; this is a one-time, irreversible evaluation",
    )
    parser.add_argument("--reason", default="Phase 13 final evaluation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    prior = already_run()
    if prior:
        log.error(
            "The final test was already evaluated on %s (reason: %s).\n"
            "Running it again means the first result informed a change, which "
            "is optimisation against the test set.\n"
            "Results are in reports/final/. Evidence: %s",
            prior["executed_at"], prior.get("reason"), EVIDENCE_FILE,
        )
        return 1

    if not args.confirm:
        log.error(
            "Refusing to run without --confirm.\n\n"
            "This unlocks the final-test window permanently. After it runs, "
            "nothing may be tuned: not a threshold, not a hyperparameter, not "
            "the universe, not the cost model. Whatever the number is, it is "
            "the number that goes in the report.\n\n"
            "  python scripts/final_evaluation.py --confirm"
        )
        return 1

    import pandas as pd

    from trademind.backtesting import (
        BacktestConfig,
        buy_and_hold,
        equal_weight_rebalanced,
        performance_metrics,
        render_report,
        run_backtest,
    )
    from trademind.config import load_config
    from trademind.decision import DecisionEngine, RiskEngine, Thresholds
    from trademind.ensemble import load_production_ensemble
    from trademind.models.metrics import classification_metrics, summarise
    from trademind.provenance import capture
    from trademind.storage.lake import ParquetLake
    from trademind.validation import FinalTestLock

    cfg = load_config()
    provenance = capture(config_hash=cfg.config_hash)
    log.info("Provenance:\n%s", provenance.render())

    if not provenance.reproducible:
        log.error(
            "Refusing to run a final evaluation from an unreproducible state: "
            "%s", "; ".join(provenance.warnings()),
        )
        return 1

    # --- preconditions ---------------------------------------------------
    lake = ParquetLake(cfg.data_root)
    panel = lake.read_all_features(cfg.feature_version)
    if panel.empty:
        log.error("No feature panel. Run the pipeline before evaluating.")
        return 1

    lock_path = cfg.data_root / "final_test_lock.json"
    if not lock_path.exists():
        log.error(
            "No final-test lock at %s. The lock must have been created before "
            "any development work, or there is nothing to prove was untouched.",
            lock_path,
        )
        return 1
    lock = FinalTestLock.load(lock_path)

    for key in ("buy_threshold", "sell_threshold", "min_expected_return"):
        if cfg.get(f"decision.{key}", None) is None:
            log.error(
                "decision.%s is null. Thresholds must be derived and locked on "
                "development data before the final test is opened.", key,
            )
            return 1

    production_path = cfg.root / "models" / "production_ensemble.joblib"
    if not production_path.exists():
        log.error("No production ensemble. Run `main.py ensemble` first.")
        return 1
    production = load_production_ensemble(production_path)
    if production.feature_version != cfg.feature_version:
        log.error(
            "Production ensemble feature version %s does not match config %s.",
            production.feature_version,
            cfg.feature_version,
        )
        return 1
    if pd.Timestamp(production.training_end) >= lock.final_test_start:
        log.error(
            "Production ensemble was trained through %s, which reaches the "
            "locked final-test window beginning %s.",
            production.training_end,
            lock.final_test_start.date(),
        )
        return 1

    required_features = {
        feature
        for model in [*production.direction_models.values(), production.return_model]
        for feature in model.feature_names_
    }
    required_features.update(production.context_columns)
    missing_features = sorted(required_features - set(panel.columns))
    if missing_features:
        log.error("Feature panel cannot score the production ensemble: %s", missing_features)
        return 1

    # --- the unlock ------------------------------------------------------
    log.warning("=" * 70)
    log.warning("UNLOCKING THE FINAL TEST. This happens once.")
    log.warning("=" * 70)

    final_panel = lock.unlock(panel, reason=args.reason)
    log.info("Final-test window: %d rows, %s..%s",
             len(final_panel),
             final_panel["date"].min().date(), final_panel["date"].max().date())

    final_signals = production.predict(final_panel)
    outcomes = final_panel[["date", "symbol", "tradeable_direction_1d"]].rename(
        columns={"tradeable_direction_1d": "y_true"}
    )
    final_signals = final_signals.merge(
        outcomes, on=["date", "symbol"], how="inner", validate="one_to_one"
    ).dropna(subset=["calibrated", "y_true"])
    if final_signals.empty:
        log.error(
            "The unlocked final-test window produced no resolved, scoreable signals."
        )
        return 1

    # --- model quality ---------------------------------------------------
    model_metrics = classification_metrics(
        (final_signals["y_true"] > 0).astype(float), final_signals["calibrated"]
    )
    log.info("Final-test model: %s", summarise(model_metrics))

    # --- strategy --------------------------------------------------------
    bt_config = BacktestConfig.from_config(cfg)
    engine = DecisionEngine(
        Thresholds(
            buy=cfg.get("decision.buy_threshold"),
            sell=cfg.get("decision.sell_threshold"),
            min_expected_return=cfg.get("decision.min_expected_return"),
        ),
        RiskEngine(max_positions=cfg.get("decision.max_positions", 10)),
        sizing_method=cfg.get("decision.sizing_method", "fixed"),
        max_weight=bt_config.max_position_weight,
    )

    result = run_backtest(final_panel, final_signals, engine, bt_config)
    metrics = performance_metrics(
        result.equity_curve, result.trades, bt_config.initial_capital
    )

    bars = {s: g.reset_index(drop=True) for s, g in final_panel.groupby("symbol")}
    sessions = sorted(pd.Timestamp(d).date() for d in final_panel["date"].unique())
    benchmarks = {
        "buy_and_hold": performance_metrics(
            buy_and_hold(bars, sessions, bt_config),
            initial_capital=bt_config.initial_capital),
        "equal_weight_monthly": performance_metrics(
            equal_weight_rebalanced(bars, sessions, bt_config),
            initial_capital=bt_config.initial_capital),
    }

    report = render_report(metrics, benchmarks)
    log.info("\n%s", report)
    log.info("Max reconciliation error: %.8f", result.max_reconciliation_error)

    # --- persist ---------------------------------------------------------
    out = ROOT / "reports" / "final"
    out.mkdir(parents=True, exist_ok=True)

    result.equity_curve.to_csv(out / "final_test_equity_curve.csv", index=False)
    result.trades.to_csv(out / "final_test_trades.csv", index=False)
    (out / "final_test_report.txt").write_text(report, encoding="utf-8")
    (out / "final_test_metrics.json").write_text(
        json.dumps({
            "model": model_metrics,
            "strategy": metrics,
            "benchmarks": benchmarks,
            "max_reconciliation_error": result.max_reconciliation_error,
        }, indent=2, default=str),
        encoding="utf-8",
    )

    EVIDENCE_FILE.write_text(json.dumps({
        "executed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "reason": args.reason,
        "lock_fingerprint": lock.fingerprint,
        "final_test_start": str(lock.final_test_start.date()),
        "n_rows": len(final_panel),
        "provenance": provenance.to_dict(),
        "headline": {
            "total_return": metrics.get("total_return"),
            "sharpe": metrics.get("sharpe"),
            "roc_auc": model_metrics.get("roc_auc"),
        },
    }, indent=2, default=str), encoding="utf-8")

    lock.save(lock_path)   # persists the access record

    log.warning("=" * 70)
    log.warning("Final test complete. Nothing may be tuned from this point.")
    log.warning("Results: %s", out)
    log.warning("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
