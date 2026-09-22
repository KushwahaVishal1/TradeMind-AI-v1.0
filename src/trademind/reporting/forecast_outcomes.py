"""Resolve prospective evening forecasts without changing issued predictions."""

import hashlib
import json
from datetime import timedelta

import numpy as np
import pandas as pd

from ..ingestion.intraday import INDICES, TIMEZONE


def _files(cfg):
    return sorted((cfg.data_root / "intraday" / "evening_forecasts").glob("*.json"))


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_outcome(forecast, opening, closing):
    """Direction correctness is distinct from numerical return error."""
    if not np.isfinite([opening, closing]).all() or min(opening, closing) <= 0:
        raise ValueError("Outcome prices must be positive and finite")
    actual = closing / opening - 1
    direction = "UP" if actual > 0 else "DOWN" if actual < 0 else "FLAT"
    status = "FLAT" if direction == "FLAT" else (
        "CORRECT" if forecast["direction"] == direction else "WRONG"
    )
    return {
        "actual_open": float(opening), "actual_close": float(closing),
        "actual_return": actual, "actual_direction": direction, "status": status,
        "return_error": actual - forecast["expected_return"],
    }


def eligible(forecast):
    created = pd.Timestamp(forecast["created_at"])
    return (created.tzinfo is not None
            and str(created.tz_convert(TIMEZONE).date()) < forecast["forecast_session"]
            and forecast["as_of"] < forecast["forecast_session"]
            and forecast["symbol"] in INDICES.values()
            and forecast["direction"] in {"UP", "DOWN"})


def resolve_forecasts(cfg, now=None, fetch=None):
    """Resolve due sessions after 18:00 IST; unavailable data remains pending."""
    clock = pd.Timestamp.now(tz=TIMEZONE) if now is None else pd.Timestamp(now)
    if clock.tzinfo is None:
        raise ValueError("Resolution time must include a timezone")
    clock = clock.tz_convert(TIMEZONE)
    if fetch is None:
        import yfinance as yf

        def fetch(symbol, start, end):
            return yf.Ticker(symbol).history(
                start=start, end=end, interval="1d", auto_adjust=False,
                actions=False, timeout=15,
            )

    errors, written = {}, 0
    for path in _files(cfg):
        try:
            forecast = json.loads(path.read_text(encoding="utf-8"))
            result_path = path.parent / "outcomes" / path.name
            if result_path.exists():
                continue
            if not eligible(forecast):
                raise ValueError("Forecast was not issued before its target session")
            day = pd.Timestamp(forecast["forecast_session"]).date()
            if day > clock.date() or (day == clock.date() and clock.hour < 18):
                continue
            raw = fetch(forecast["symbol"], str(day), str(day + timedelta(days=1)))
            if raw.empty:
                raise ValueError("Final daily bar unavailable; result remains pending")
            stamps = pd.DatetimeIndex(raw.index)
            if stamps.tz is None:
                raise ValueError("Provider timestamps lack a timezone")
            bar = raw.loc[stamps.tz_convert(TIMEZONE).date == day]
            if len(bar) != 1:
                raise ValueError("Expected exactly one bar for the forecast session")
            outcome = score_outcome(forecast, float(bar.Open.iloc[0]),
                                    float(bar.Close.iloc[0]))
            outcome.update({"resolved_at": clock.isoformat(), "forecast_hash": _hash(path),
                            "source": "Yahoo Finance daily index OHLC"})
            result_path.parent.mkdir(parents=True, exist_ok=True)
            with result_path.open("x", encoding="utf-8") as handle:
                json.dump(outcome, handle, indent=2, allow_nan=False)
            written += 1
        except Exception as exc:
            errors[path.stem] = str(exc)
    return written, errors


def forecast_history(cfg):
    rows = []
    for path in _files(cfg):
        forecast = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "Session": forecast["forecast_session"], "Index": forecast["index"],
            "Generated": forecast["created_at"], "Predicted direction": forecast["direction"],
            "Probability up (%)": 100 * forecast["probability_up"],
            "Predicted return (%)": 100 * forecast["expected_return"],
            "Actual direction": None, "Actual return (%)": None,
            "Actual open": None, "Actual close": None, "Error (pp)": None,
            "Result": "PENDING" if eligible(forecast) else "INELIGIBLE",
        }
        outcome_path = path.parent / "outcomes" / path.name
        if outcome_path.exists() and row["Result"] != "INELIGIBLE":
            outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
            if outcome["forecast_hash"] != _hash(path):
                raise ValueError(f"Saved forecast changed after resolution: {path.name}")
            row.update({
                "Actual direction": outcome["actual_direction"],
                "Actual return (%)": 100 * outcome["actual_return"],
                "Actual open": outcome["actual_open"], "Actual close": outcome["actual_close"],
                "Error (pp)": 100 * outcome["return_error"], "Result": outcome["status"],
            })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["Session", "Index"], ascending=[False, True]) \
        if rows else pd.DataFrame()
