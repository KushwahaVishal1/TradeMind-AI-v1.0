"""Routes render through the shared entrypoint, with a single navigation menu."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_routes_render_without_duplicate_radio():
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py"),
        default_timeout=30,
    ).run()
    assert not app.exception
    assert app.title[0].value == "Overview"
    assert not app.sidebar.radio
    for route, title in [
        ("evening", "Tomorrow's session forecast"),
        ("intraday", "Intraday indices"),
        ("signals", "Signals"),
        ("performance", "Performance"),
        ("retraining", "Retraining governance"),
    ]:
        app.switch_page(f"app_pages/{route}.py").run()
        assert not app.exception
        assert app.title[0].value == title
        assert not app.sidebar.radio
    # A bookmarked URL must also work on the first run, without visiting home.
    direct = AppTest.from_file(app._script_path, default_timeout=30)
    direct.switch_page("app_pages/evening.py").run()
    assert not direct.exception
    assert direct.title[0].value == "Tomorrow's session forecast"
