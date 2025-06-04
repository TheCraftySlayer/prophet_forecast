import pytest
pytest.importorskip("pandas")

import pandas as pd
from prophet_analysis import monitor_residuals


def test_monitor_residuals_flags_large_error():
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    df = pd.DataFrame({
        "ds": dates,
        "yhat": [10.0] * 20,
        "actual": [10.0] * 20,
    })
    df.loc[14, "yhat"] = 0.0
    flagged = monitor_residuals(df)
    assert not flagged.empty
    assert flagged.iloc[0]["ds"] == dates[14]
