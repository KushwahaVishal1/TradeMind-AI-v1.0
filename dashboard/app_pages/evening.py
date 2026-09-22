"""Evening next-session forecasts, with validation displayed beside estimates."""

import json

import pandas as pd
import streamlit as st

from trademind.ingestion.evening_forecast import generate_evening_forecasts
from trademind.ingestion.intraday import TIMEZONE
from trademind.reporting.forecast_outcomes import forecast_history, resolve_forecasts


def render(state, cfg, store):
    st.title("Tomorrow's session forecast")
    st.caption(
        "Generate after 18:00 IST. Forecast target: the next trading session's "
        "open-to-close direction and return, using completed daily index history."
    )
    st.info(
        "Experimental forecast, not a BUY/SELL instruction. Tomorrow's opening "
        "price, opening gap and intraday highs/lows are not predicted. "
        "Probabilities are model estimates, not calibrated confidence scores."
    )
    if st.button("Generate evening forecasts"):
        try:
            with st.spinner("Downloading daily history and validating index models..."):
                _, errors = generate_evening_forecasts(cfg)
                _, outcome_errors = resolve_forecasts(cfg)
                errors.update(outcome_errors)
            for name, error in errors.items():
                st.error(f"{name}: {error}")
            if not errors:
                st.success("Forecasts saved. Existing forecasts are preserved on reruns.")
        except Exception as exc:
            st.error(str(exc))
    st.subheader("Daily forecast vs actual")
    st.caption(
        "One row per index per session. Correct/wrong compares predicted and actual "
        "open-to-close direction. Update results after 18:00 IST."
    )
    if st.button("Update actual results"):
        with st.spinner("Checking completed sessions..."):
            written, errors = resolve_forecasts(cfg)
        st.success(f"Added {written} outcomes.")
        for name, error in errors.items():
            st.warning(f"{name}: {error}")
    try:
        history = forecast_history(cfg)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Cannot read forecast results: {exc}")
        history = pd.DataFrame()
    if not history.empty:
        counts = history.Result.value_counts()
        correct, wrong = int(counts.get("CORRECT", 0)), int(counts.get("WRONG", 0))
        columns = st.columns(5)
        for col, label, value in zip(columns, ["Total", "Correct", "Wrong", "Pending",
                                               "Direction accuracy"],
                                     [len(history), correct, wrong,
                                      int(counts.get("PENDING", 0)),
                                      f"{correct / (correct + wrong):.1%}"
                                      if correct + wrong else "N/A"], strict=True):
            col.metric(label, value)
        st.caption(
            f"Flat outcomes: {counts.get('FLAT', 0)}; ineligible forecasts: "
            f"{counts.get('INELIGIBLE', 0)}. Both and pending rows are excluded from "
            "accuracy. Error is actual minus predicted return in percentage points. "
            "Counts cover all saved sessions, not historical model validation."
        )
        st.dataframe(history, hide_index=True, width="stretch")
        daily = pd.crosstab(history.Session, history.Result).reindex(
            columns=["CORRECT", "WRONG", "PENDING", "FLAT", "INELIGIBLE"], fill_value=0
        ).sort_index(ascending=False)
        st.caption("Counts by forecast session")
        st.dataframe(daily, width="stretch")
        st.download_button("Download forecast vs actual (CSV)",
                           history.to_csv(index=False), file_name="forecast_vs_actual.csv",
                           mime="text/csv")
    directory = cfg.data_root / "intraday" / "evening_forecasts"
    records = []
    for path in sorted(directory.glob("*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            st.error(f"Cannot read {path.name}: {exc}")
    if not records:
        st.info("No evening forecasts saved yet. Generate them after 18:00 IST.")
        return
    dates = sorted({r["forecast_session"] for r in records}, reverse=True)
    session = st.selectbox("Forecast trading session", dates)
    today = str(pd.Timestamp.now(tz=TIMEZONE).date())
    if session <= today:
        st.warning("This is a forecast for today or an earlier session, not tomorrow.")
    for row in [r for r in records if r["forecast_session"] == session]:
        st.subheader(row["index"])
        left, middle, right = st.columns(3)
        left.metric("Estimated direction", row["direction"])
        middle.metric("Estimated probability of closing above open",
                      f"{row['probability_up']:.1%}")
        right.metric("Expected open-to-close return", f"{row['expected_return']:+.3%}")
        st.caption(
            f"Data through {row['as_of']} | Generated {row['created_at']} | "
            f"Training samples: {row['training_samples']:,}"
        )
        st.write(
            f"Historical validation ({row['validation_sessions']} sessions): "
            f"direction accuracy {row['validation_accuracy']:.1%}, "
            f"baseline accuracy {row['baseline_accuracy']:.1%}. "
            f"Mean absolute return error {row['return_mae']:.3%}, "
            f"baseline error {row['baseline_return_mae']:.3%}."
        )
        if (row["validation_accuracy"] <= row["baseline_accuracy"]
                or row["return_mae"] >= row["baseline_return_mae"]):
            st.warning("The model did not beat the baseline on both validation measures.")
        st.caption(row["validation_note"])
    st.download_button("Download forecast history (CSV)",
                       pd.DataFrame(records).to_csv(index=False),
                       file_name="evening_forecasts.csv", mime="text/csv")
