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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402
from pages import drift, model, overview, performance, retraining, signals  # noqa: E402

from trademind.config import load_config  # noqa: E402
from trademind.reporting import build_dashboard_state  # noqa: E402
from trademind.storage import PredictionStore, init_db  # noqa: E402

PAGES = {
    "Overview": overview,
    "Signals": signals,
    "Performance": performance,
    "Model": model,
    "Drift": drift,
    "Retraining": retraining,
}


@st.cache_resource
def _load():
    cfg = load_config()
    conn = init_db(cfg.data_root / "trademind.db")
    return cfg, PredictionStore(conn)


def main() -> None:
    st.set_page_config(page_title="TradeMind AI", layout="wide")

    cfg, store = _load()
    state = build_dashboard_state(cfg, store)

    st.sidebar.title("TradeMind AI")
    st.sidebar.caption("Probabilistic market signal and decision-support platform")
    choice = st.sidebar.radio("Page", list(PAGES))

    st.sidebar.divider()
    st.sidebar.warning(
        "**Research system.** Not investment advice, not a profitable "
        "strategy, not an autonomous trading system. Its value is the "
        "controls: leakage prevention, temporal validation, accounting "
        "correctness, and governance."
    )
    st.sidebar.caption(f"config hash `{cfg.config_hash}`")

    PAGES[choice].render(state, cfg, store)


if __name__ == "__main__":
    main()
