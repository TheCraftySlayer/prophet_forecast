import pytest
pytest.importorskip("pandas")

import pandas as pd
from prophet_analysis import build_flag_series


def test_build_flag_series_basic():
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    flags = build_flag_series(dates, [dates[1], dates[3]])
    assert list(flags) == [0, 1, 0, 1, 0]
