import pytest
pytest.importorskip("pandas")

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from holidays_calendar import get_holidays_dataframe
from prophet_analysis import (
    create_prophet_holidays,
    prepare_data,
    prepare_prophet_data,
    train_prophet_model,
)
from tests.test_pipeline_alignment import DummyProphet


def _setup():
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
        closure_dates=[],
        press_release_dates=[],
    )
    return prophet_df, holidays, regs


def test_capacity_from_config():
    prophet_df, holidays, regs = _setup()
    with patch('prophet_analysis.Prophet', DummyProphet):
        _, _, future = train_prophet_model(
            prophet_df,
            holidays,
            regs,
            future_periods=2,
            model_params={'capacity': 500, 'growth': 'logistic'},
        )
    assert future['cap'].iloc[0] == 500


def test_capacity_auto():
    prophet_df, holidays, regs = _setup()
    expected = float(prophet_df['y'].max() * 1.1)
    with patch('prophet_analysis.Prophet', DummyProphet):
        _, _, future = train_prophet_model(
            prophet_df,
            holidays,
            regs,
            future_periods=2,
        )
    assert future['cap'].iloc[0] == pytest.approx(expected)

