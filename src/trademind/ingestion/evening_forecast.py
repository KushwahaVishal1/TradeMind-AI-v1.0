"""Experimental evening forecast of next-session close / open - 1.

Uses only completed daily index bars. It cannot predict the intraday path,
entry price, or opening gap. Evaluation uses expanding chronological folds.
"""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .calendar import TradingCalendar
from .intraday import INDICES, TIMEZONE


def make_features(history):
    """Past-only predictors and separately shifted next-session labels."""
    h = history.sort_index()
    c, o = h.Close, h.Open
    f = pd.DataFrame(index=h.index)
    f["return_1"] = c.pct_change(fill_method=None)
    f["return_5"] = c.pct_change(5, fill_method=None)
    f["return_20"] = c.pct_change(20, fill_method=None)
    f["session_return"] = c / o - 1
    f["range"] = (h.High - h.Low) / o
    f["gap"] = o / c.shift(1) - 1
    f["volatility"] = f.return_1.rolling(20).std()
    f["close_position"] = (c - h.Low) / (h.High - h.Low).replace(0, np.nan)
    target = c.shift(-1) / o.shift(-1) - 1
    return f.replace([np.inf, -np.inf], np.nan), target


def fit_forecast(history, settings):
    x, target = make_features(history)
    usable = x.notna().all(axis=1) & target.notna()
    train, y = x.loc[usable], target.loc[usable]
    if len(train) < settings["minimum_samples"]:
        raise ValueError("Insufficient complete historical sessions for validation")
    if x.iloc[-1].isna().any():
        raise ValueError("Latest session has incomplete features")

    def models():
        return (
            make_pipeline(StandardScaler(), LogisticRegression(
                C=settings["logistic_c"], max_iter=1000, random_state=42)),
            make_pipeline(StandardScaler(), Ridge(alpha=settings["ridge_alpha"])),
        )

    observed, probs, returns, baseline_probs, baseline_returns = [], [], [], [], []
    split = TimeSeriesSplit(n_splits=settings["validation_splits"],
                            test_size=settings["validation_sessions"], gap=1)
    for fit, test in split.split(train):
        classifier, regressor = models()
        direction = (y.iloc[fit] > 0).astype(int)
        if direction.nunique() < 2:
            raise ValueError("Training fold lacks both directions")
        classifier.fit(train.iloc[fit], direction)
        regressor.fit(train.iloc[fit], y.iloc[fit])
        observed.extend(y.iloc[test])
        probs.extend(classifier.predict_proba(train.iloc[test])[:, 1])
        returns.extend(regressor.predict(train.iloc[test]))
        baseline_probs.extend([direction.mean()] * len(test))
        baseline_returns.extend([y.iloc[fit].mean()] * len(test))
    actual = np.asarray(observed) > 0
    accuracy = accuracy_score(actual, np.asarray(probs) >= .5)
    baseline = accuracy_score(actual, np.asarray(baseline_probs) >= .5)
    mae = mean_absolute_error(observed, returns)
    baseline_mae = mean_absolute_error(observed, baseline_returns)
    classifier, regressor = models()
    classifier.fit(train, (y > 0).astype(int))
    regressor.fit(train, y)
    probability = float(classifier.predict_proba(x.iloc[[-1]])[0, 1])
    estimate = float(regressor.predict(x.iloc[[-1]])[0])
    return {
        "probability_up": probability, "expected_return": estimate,
        "direction": "UP" if probability >= .5 else "DOWN",
        "validation_accuracy": float(accuracy), "baseline_accuracy": float(baseline),
        "validation_auc": float(roc_auc_score(actual, probs))
        if len(set(actual)) == 2 else None,
        "return_mae": float(mae), "baseline_return_mae": float(baseline_mae),
        "validation_sessions": len(observed), "training_samples": len(train),
        "training_first_feature_date": str(train.index[0].date()),
        "training_last_feature_date": str(train.index[-1].date()),
        "label_data_through": str(history.index[-1].date()),
        "validation_note": "Experimental; historical validation is not proof of an edge",
    }


def generate_evening_forecasts(cfg, now=None):
    """Fetch fresh daily index bars and save an immutable forecast per session."""
    import yfinance as yf

    clock = pd.Timestamp.now(tz=TIMEZONE) if now is None else pd.Timestamp(now)
    if clock.tzinfo is None:
        raise ValueError("Observation time must include a timezone")
    clock = clock.tz_convert(TIMEZONE)
    settings_text = (cfg.root / "config" / "evening_forecast.yaml").read_text()
    settings = yaml.safe_load(settings_text)
    if clock.hour < settings["ready_hour_ist"]:
        raise ValueError("Generate evening forecasts after 18:00 IST")
    output = cfg.data_root / "intraday" / "evening_forecasts"
    output.mkdir(parents=True, exist_ok=True)
    results, errors = [], {}
    for name, symbol in INDICES.items():
        try:
            calendar = TradingCalendar("BSE" if symbol == "^BSESN" else "NSE")
            if calendar.backend != "exchange":
                raise ValueError("An exchange calendar is required to date the forecast")
            recent = calendar.sessions(clock.date() - timedelta(days=15), clock.date())
            if not recent:
                raise ValueError("No recent exchange session found")
            as_of = recent[-1]
            next_session = calendar.next_session(as_of)
            if next_session is None or next_session <= clock.date():
                raise ValueError("No future trading session available")
            path = output / f"{symbol[1:]}_{next_session}.json"
            if path.exists():
                results.append(json.loads(path.read_text()))
                continue
            history = yf.Ticker(symbol).history(
                period=settings["history_period"], interval="1d", auto_adjust=False,
                actions=False, timeout=15,
            )
            if history.empty:
                raise ValueError("Provider returned no daily history")
            if history.index.tz is None:
                raise ValueError("Daily history has no timezone")
            history.index = history.index.tz_convert(TIMEZONE)
            history = history.loc[history.index.date <= as_of].sort_index()
            prices = history[["Open", "High", "Low", "Close"]]
            if (history.empty or history.index[-1].date() != as_of
                    or history.index.has_duplicates or not np.isfinite(prices).all().all()
                    or not prices.gt(0).all().all()):
                raise ValueError("Daily history is stale, incomplete, or invalid")
            forecast = fit_forecast(history, settings)
            forecast.update({
                "index": name, "symbol": symbol, "as_of": str(as_of),
                "forecast_session": str(next_session), "created_at": clock.isoformat(),
                "target": "next-session close / open - 1",
                "source": "Yahoo Finance daily index OHLC",
                "model": "experimental-evening-v1-logistic-ridge",
                "settings_hash": hashlib.sha256(settings_text.encode()).hexdigest(),
                "data_hash": hashlib.sha256(history.to_csv().encode()).hexdigest(),
            })
            # Preserve the exact training input alongside the first issued forecast.
            history.to_parquet(output / f"{symbol[1:]}_{next_session}_input.parquet")
            with path.open("x", encoding="utf-8") as handle:
                json.dump(forecast, handle, indent=2, allow_nan=False)
            results.append(forecast)
        except Exception as exc:
            errors[name] = str(exc)
    return results, errors
