"""Historical performance must use aligned return forecasts and temporal axes."""

import pandas as pd
import pytest

from trademind.ensemble.pipeline import attach_expected_returns
from trademind.reporting.loaders import load_equity_curve


def test_returns_match_symbol_and_date_without_replacing_direction_labels():
    signals = pd.DataFrame({
        "date": ["2023-01-03", "2023-01-03"],
        "symbol": ["A", "B"], "y_true": [1, 0], "calibrated": [0.7, 0.6],
    })
    returns = pd.DataFrame({
        "date": ["2023-01-03", "2023-01-03"],
        "symbol": ["B", "A"], "y_pred": [-0.01, 0.02],
    })
    result = attach_expected_returns(signals, returns)
    assert result.expected_return.tolist() == [0.02, -0.01]
    assert result.y_true.tolist() == [1, 0]
    with pytest.raises(ValueError, match="finite OOF"):
        attach_expected_returns(signals, returns.iloc[:1])
    with pytest.raises(pd.errors.MergeError):
        attach_expected_returns(signals, pd.concat([returns, returns]))


def test_equity_dates_are_temporal_and_sorted(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "equity_curve.csv").write_text(
        "date,equity\n2023-01-04,101\n2023-01-03,100\n", encoding="utf-8"
    )
    curve = load_equity_curve(tmp_path)
    assert pd.api.types.is_datetime64_any_dtype(curve.date)
    assert curve.equity.tolist() == [100, 101]


def test_performance_page_accepts_csv_date_strings(monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "dashboard"))
    app = AppTest.from_string('''
import pandas as pd
from pages.performance import render
curve = pd.DataFrame({
    "date": ["2023-01-04", "2023-01-03"],
    "equity": [100., 100.], "exposure": [0., 0.], "n_positions": [0, 0],
})
render({"performance": {"total_return": 0., "n_sessions": 2},
        "equity_curve": curve}, None, None)
assert curve.date.iloc[0] == "2023-01-04"
''').run()
    assert not app.exception
    assert any("03 Jan 2023 to 04 Jan 2023" in item.value for item in app.caption)
    app.run()
    assert not app.exception
