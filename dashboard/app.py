"""TradeMind AI dashboard.

    streamlit run dashboard/app.py

Deliberately thin. Every judgement -- what counts as a concern, how a number
should be read, what order things appear in -- lives in
``src/trademind/reporting/panels.py`` where it is unit-tested. This file only
draws.

That split matters more than it looks. Presentation logic buried in Streamlit
callbacks cannot be tested, and untested presentation logic is where
inconvenient findings quietly stop being displayed.
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402
from app_pages import (  # noqa: E402
    drift,
    evening,
    intraday,
    model,
    overview,
    performance,
    retraining,
    signals,
)

from trademind.config import load_config  # noqa: E402
from trademind.reporting import build_dashboard_state  # noqa: E402
from trademind.storage import PredictionStore, init_db  # noqa: E402

PAGES = {
    "Overview": overview,
    "Signals": signals,
    "Intraday indices": intraday,
    "Tomorrow's forecast": evening,
    "Performance": performance,
    "Model": model,
    "Drift": drift,
    "Retraining": retraining,
}


def _load():
    cfg = load_config()
    conn = init_db(cfg.data_root / "trademind.db")
    return cfg, PredictionStore(conn)


def main() -> None:
    st.set_page_config(page_title="TradeMind AI", layout="wide")

    cfg, store = _load()
    # SQLite connections belong to the thread that created them. Streamlit
    # reruns can use a new script thread, so open and close one per run.
    try:
        _render(cfg, store)
    finally:
        store.conn.close()


def _render(cfg, store) -> None:
    state = build_dashboard_state(cfg, store)

    st.sidebar.title("TradeMind AI")
    st.sidebar.caption("Probabilistic market signal and decision-support platform")
    # Explicit navigation disables automatic pages/ discovery. Each route
    # invokes the renderer with the state and connection created for this run.
    selected = st.navigation([
        st.Page(
            partial(module.render, state, cfg, store),
            title=title,
            url_path=module.__name__.rsplit(".", 1)[-1],
            default=title == "Overview",
        )
        for title, module in PAGES.items()
    ])

    st.sidebar.divider()
    st.sidebar.warning(
        "**Research system.** Not investment advice, not a profitable "
        "strategy, not an autonomous trading system. Its value is the "
        "controls: leakage prevention, temporal validation, accounting "
        "correctness, and governance."
    )
    st.sidebar.caption(f"config hash `{cfg.config_hash}`")

    selected.run()


if __name__ == "__main__":
    main()
