import pytest
pytest.importorskip("pandas")

from pathlib import Path
from unittest.mock import patch

import pandas as pd
from holidays_calendar import get_holidays_dataframe
from prophet_analysis import (
    create_prophet_holidays,
    prepare_data,
    prepare_prophet_data,
    create_stacked_ensemble,
)
from tests.test_pipeline_alignment import DummyProphet


def test_stacked_ensemble_runs():
    df, regs = prepare_data(Path("calls.csv"), Path("visitors.csv"), Path("queries.csv"))
    prophet_df = prepare_prophet_data(df)
    holiday_df = get_holidays_dataframe()
    mask = (
        (holiday_df["event"] == "county_holiday")
        & (holiday_df["date"] >= df.index.min())
        & (holiday_df["date"] <= df.index.max())
    )
    holidays = create_prophet_holidays(
        holiday_df.loc[mask, "date"],
        pd.date_range(df.index.min(), df.index.max(), freq="MS"),
        closure_dates=[],
        press_release_dates=[],
    )
    with patch("prophet_analysis._get_prophet", return_value=DummyProphet), \
         patch("prophet_analysis._fit_prophet_with_fallback"), \
         patch("prophet_analysis._ensure_tbb_on_path"):
        forecast, models = create_stacked_ensemble(prophet_df, holidays, regs)
    assert "yhat" in forecast.columns
    assert len(models) >= 3

