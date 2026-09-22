"""Inspect recent completed index candles, separately from daily signals."""

import altair as alt
import pandas as pd
import streamlit as st

from trademind.ingestion.intraday import (
    INDICES,
    INTERVALS,
    TIMEZONE,
    load_candles,
    refresh_candles,
)


def render(state, cfg, store):
    st.title("Intraday indices")
    st.caption(
        "NIFTY 50, BANK NIFTY and SENSEX spot indices · Yahoo Finance · "
        "Completed candles only · All timestamps in India time (IST)"
    )
    st.info(
        "Research data that may be delayed. These are index levels, not futures or "
        "options prices. The daily stock models do not generate intraday signals here."
    )
    name = st.selectbox("Index", list(INDICES))
    interval = st.selectbox("Candle interval", list(INTERVALS), index=1)
    symbol = INDICES[name]
    if st.button("Fetch latest candles"):
        try:
            with st.spinner(f"Fetching {name} {interval} candles..."):
                refresh_candles(cfg.data_root, symbol, interval)
            st.success("Download completed.")
        except Exception as exc:
            st.error(f"Download failed: {exc}. Previously saved candles are retained.")
    try:
        frame = load_candles(cfg.data_root, symbol, interval)
    except Exception as exc:
        st.error(f"Cannot read saved candles: {exc}")
        return
    if frame.empty:
        st.info("No candles saved for this selection. Click Fetch latest candles.")
        return
    last = frame.iloc[-1]
    st.caption(
        f"Newest saved candle starts: {last.timestamp:%d %b %Y %H:%M} IST | "
        f"Last successful fetch: {frame.fetched_at.max():%d %b %Y %H:%M} IST"
    )
    age = pd.Timestamp.now(tz=TIMEZONE) - last.timestamp
    if age > pd.Timedelta(minutes=2 * INTERVALS[interval]):
        st.warning(
            "Saved candles are older than two intervals. The market may be closed "
            "or the feed delayed. Check the timestamp before using this data."
        )
    session = st.selectbox(
        "Trading date", sorted(frame.timestamp.dt.date.unique(), reverse=True)
    )
    selected = frame.loc[frame.timestamp.dt.date == session].copy()
    st.metric("Last saved close (index points)", f"{selected.close.iloc[-1]:,.2f}")
    # Explicit clock strings prevent the browser from shifting IST labels.
    selected = selected.sort_values("timestamp")
    selected["Time (IST)"] = selected.timestamp.dt.tz_convert(TIMEZONE).dt.strftime("%H:%M")
    times = selected["Time (IST)"].tolist()
    chart = alt.Chart(selected).mark_line(point=len(selected) == 1).encode(
        x=alt.X("Time (IST):O", sort=times,
                axis=alt.Axis(values=times[::max(1, len(times) // 10)], labelAngle=0)),
        y=alt.Y("close:Q", title="Index points",
                scale=alt.Scale(zero=False, nice=False), axis=alt.Axis(format=",.2f")),
        tooltip=[alt.Tooltip("Time (IST):O"),
                 alt.Tooltip("close:Q", title="Close", format=",.2f")],
    ).properties(height=350)
    st.altair_chart(chart, width="stretch")
    st.caption(
        f"Saved close range: {selected.close.min():,.2f} to "
        f"{selected.close.max():,.2f} points. Price axis zoomed to the selected session."
    )
    st.dataframe(
        selected[["Time (IST)", "open", "high", "low", "close", "volume"]],
        width="stretch", hide_index=True,
    )
    st.caption("Index volume may be unavailable; blank volume is not a trading-volume signal.")
    st.download_button(
        "Download saved history (CSV)", frame.to_csv(index=False),
        file_name=f"{symbol[1:]}_{interval}.csv", mime="text/csv",
    )
