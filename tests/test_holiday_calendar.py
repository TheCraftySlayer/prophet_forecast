import pytest
pytest.importorskip("pandas")

import pandas as pd
from holidays_calendar import get_holidays_dataframe


def test_holiday_calendar_contents():
    df = get_holidays_dataframe()
    assert {'date', 'event'} <= set(df.columns)
    assert not df.empty
    # Check that types are datetime
    assert pd.api.types.is_datetime64_any_dtype(df['date'])

def test_assessor_events_included():
    df = get_holidays_dataframe()
    assert 'bill_mailed' in df['event'].values


def test_holiday_calendar_dedup_gaps():
    df = get_holidays_dataframe()
    assert not df.duplicated(subset=['date', 'event']).any()
    protest = df[df['event'] == 'days_to_protest_deadline']
    if not protest.empty:
        diffs = protest['date'].diff().dropna().dt.days
        assert diffs.max() <= 1
