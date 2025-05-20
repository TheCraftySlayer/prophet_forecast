from pathlib import Path
from unittest.mock import patch

import pandas as pd
from holidays_calendar import get_holidays_dataframe
from prophet_analysis import (create_prophet_holidays, prepare_data,
                              prepare_prophet_data, train_prophet_model)


class DummyProphet:
    def __init__(self, **kwargs):
        self.history = None
        self.extra_regressors = {}
        self.growth = kwargs.get("growth", "linear")
        self.interval_width = kwargs.get("interval_width", 0.8)
        self.seasonality_mode = kwargs.get("seasonality_mode", "additive")
        self.n_changepoints = kwargs.get("n_changepoints", 25)
        self.changepoint_prior_scale = kwargs.get("changepoint_prior_scale", 0.2)
        self.holidays = kwargs.get("holidays")

    def add_seasonality(self, *args, **kwargs):
        pass

    def add_regressor(self, name, **kwargs):
        self.extra_regressors[name] = kwargs

    def fit(self, df, **kwargs):
        self.history = df.copy()

    def make_future_dataframe(self, periods, freq="D"):
        last = self.history["ds"].iloc[-1]
        dates = pd.date_range(last, periods=periods + 1, freq=freq)[1:]
        return pd.DataFrame({"ds": dates})

    def predict(self, df):
        out = df.copy()
        out["yhat"] = 0.0
        out["yhat_lower"] = 0.0
        out["yhat_upper"] = 0.0
        return out


def test_pipeline_alignment():
    df, regs = prepare_data(
        Path("calls.csv"), Path("visitors.csv"), Path("queries.csv")
    )
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
    with patch("prophet_analysis.Prophet", DummyProphet):
        model, forecast, future = train_prophet_model(
            prophet_df, holidays, regs, future_periods=3
        )
    assert len(forecast) == len(future)
