import pytest
pytest.importorskip("pandas")

from pathlib import Path
import pandas as pd
from holidays_calendar import get_holidays_dataframe
from prophet_analysis import prepare_data


def test_spike_event_regressors_use_calendar():
    df, _ = prepare_data(Path('calls.csv'), Path('visitors.csv'), Path('queries.csv'))
    holiday_df = get_holidays_dataframe()
    notice_dates = holiday_df.loc[holiday_df['event'] == 'notice_mailout', 'date']
    tax_dates = holiday_df.loc[holiday_df['event'] == 'tax_deadline', 'date']
    holiday_dates = holiday_df.loc[holiday_df['event'] == 'county_holiday', 'date']

    df_idx = pd.to_datetime(df.index)
    # Check notice flag around first notice date within the range
    nd = notice_dates.iloc[1]  # 2024 notice date
    window = pd.date_range(nd, nd + pd.Timedelta(days=7))
    assert df.loc[df_idx.isin(window), 'notice_flag'].max() == 1

    # Check deadline flag around a tax deadline
    td = tax_dates.iloc[2]  # pick a mid-range date
    window = pd.date_range(td - pd.Timedelta(days=5), td + pd.Timedelta(days=1))
    assert df.loc[df_idx.isin(window), 'deadline_flag'].max() == 1

    # Check county holiday flag on a holiday
    hd = holiday_dates.iloc[0]
    assert df.loc[hd, 'county_holiday_flag'] == 1
