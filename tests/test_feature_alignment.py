from pathlib import Path
from unittest.mock import patch

import pandas as pd
from holidays_calendar import get_holidays_dataframe
from prophet_analysis import (
    create_prophet_holidays,
    prepare_data,
    prepare_prophet_data,
    train_prophet_model,
)
from tests.test_pipeline_alignment import DummyProphet


def dropping_cols(df, threshold=0.9, return_dropped=False):
    dropped = [c for c in [
        'holiday_flag', 'is_campaign', 'campaign_May2025', 'is_weekend'
    ] if c in df.columns]
    df = df.drop(columns=dropped)
    if return_dropped:
        return df, dropped
    return df


def test_no_feature_mismatch_after_drop():
    df, regs = prepare_data(Path('calls.csv'), Path('visitors.csv'), Path('queries.csv'))
    prophet_df = prepare_prophet_data(df)
    holiday_df = get_holidays_dataframe()
    mask = (
        (holiday_df['event'] == 'county_holiday')
        & (holiday_df['date'] >= df.index.min())
        & (holiday_df['date'] <= df.index.max())
    )
    holidays = create_prophet_holidays(
        holiday_df.loc[mask, 'date'],
        pd.date_range(df.index.min(), df.index.max(), freq='MS'),
        []
    )
    with patch('prophet_analysis.Prophet', DummyProphet), \
         patch('prophet_analysis.drop_collinear_features', side_effect=dropping_cols):
        model, forecast, future = train_prophet_model(
            prophet_df, holidays, regs, future_periods=3
        )
    assert len(forecast) == len(future)
