import pytest
pytest.importorskip("pandas")

import pandas as pd
from prophet_analysis import monitor_bias


def test_monitor_bias_flags_sequence():
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "ds": dates,
        "yhat": [10.0] * 5,
        "actual": [20.0, 16.0, 15.0, 10.0, 10.0],
    })
    flagged = monitor_bias(df, window=3, threshold=5.0)
    assert not flagged.empty
    assert flagged.iloc[0]["ds"] == dates[2]
