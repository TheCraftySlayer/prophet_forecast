import pytest
pytest.importorskip("pandas")

from pathlib import Path

from prophet_analysis import aggregate_hourly_calls, load_time_series


def test_hourly_aggregation_matches_daily():
    hourly = aggregate_hourly_calls(Path('hourly_call_data.csv'))
    daily = load_time_series(Path('calls.csv'), metric='call')
    # Compare first overlapping date
    common = hourly.index.intersection(daily.index)
    assert not common.empty
    first = common[0]
    assert hourly.loc[first] == daily.loc[first]
