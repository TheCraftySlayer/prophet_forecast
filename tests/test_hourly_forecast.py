import pytest
pytest.importorskip("pandas")

from hourly_analysis import forecast_hourly_to_daily


def test_forecast_hourly_to_daily():
    _, _, daily = forecast_hourly_to_daily('hourly_call_data.csv', periods=24)
    assert not daily.empty
    assert len(daily) >= 1
