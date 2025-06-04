import pytest
pytest.importorskip("pandas")

from hourly_analysis import forecast_hourly_to_daily


def test_forecast_hourly_to_daily():
    _, _, daily, hour_metrics, stats = forecast_hourly_to_daily('hourly_call_data.csv', periods=24)
    assert not daily.empty
    assert len(daily) >= 1
    assert hour_metrics is not None
    assert 'MAE' in hour_metrics.columns
    assert not stats.empty
    assert {'dow', 'hour', 'mean', 'std'} <= set(stats.columns)
