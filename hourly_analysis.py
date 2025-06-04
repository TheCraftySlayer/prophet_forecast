"""Simple hourly forecasting utilities."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

try:
    from prophet import Prophet
except Exception:  # pragma: no cover - stub missing
    Prophet = None  # type: ignore


def forecast_hourly_to_daily(path: str | Path, periods: int = 24 * 7):
    """Forecast hourly call volume and aggregate to daily totals.

    If the real ``prophet`` package is not available a naive average forecast is
    produced as a fallback so unit tests can run with the lightweight stubs.
    """
    df = pd.read_csv(path)
    df["ds"] = pd.to_datetime(df.iloc[:, 0], format="%m/%d/%Y %H:%M")
    df["y"] = df.iloc[:, 1].astype(float)

    if Prophet is not None and hasattr(Prophet, "fit"):
        model = Prophet(daily_seasonality=True, weekly_seasonality=True)
        model.fit(df[["ds", "y"]])
        future = model.make_future_dataframe(periods=periods, freq="H")
        forecast = model.predict(future)
    else:  # pragma: no cover - used in test environment without prophet
        model = None
        history = df[["ds", "y"]].rename(columns={"y": "yhat"})
        mean_val = history["yhat"].mean()
        future_dates = pd.date_range(
            start=history["ds"].max() + pd.Timedelta(hours=1),
            periods=periods,
            freq="H",
        )
        future = pd.DataFrame({"ds": future_dates, "yhat": mean_val})
        forecast = pd.concat([history, future], ignore_index=True)

    daily = forecast.set_index("ds").resample("D")["yhat"].sum()
    return model, forecast, daily


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Forecast hourly calls")
    p.add_argument("csv", type=Path, nargs="?", default=Path("hourly_call_data.csv"))
    p.add_argument("--periods", type=int, default=24 * 30)
    p.add_argument("--out", type=Path, default=Path("hourly_forecast.csv"))
    args = p.parse_args()

    _, fcst, daily = forecast_hourly_to_daily(args.csv, periods=args.periods)
    fcst.to_csv(args.out, index=False)
    daily.to_csv(args.out.with_name("daily_forecast.csv"))
