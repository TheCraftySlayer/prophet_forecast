"""Simple hourly forecasting utilities."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def _hourly_mae_by_hour(df: pd.DataFrame, fcst: pd.DataFrame) -> pd.DataFrame:
    """Return MAE and bias per hour-of-day."""
    merged = pd.merge(df, fcst[["ds", "yhat"]], on="ds", how="left")
    merged.dropna(subset=["yhat"], inplace=True)
    merged["hour"] = merged["ds"].dt.hour
    merged["error"] = merged["y"] - merged["yhat"]
    metrics = (
        merged.groupby("hour")["error"]
        .agg(MAE=lambda x: x.abs().mean(), bias="mean")
        .reset_index()
    )
    return metrics


try:
    from prophet import Prophet
except Exception:  # pragma: no cover - stub missing
    Prophet = None  # type: ignore


def forecast_hourly_to_daily(
    path: str | Path,
    periods: int = 24 * 7,
    *,
    bias_threshold: float = 1.0,
):
    """Forecast hourly call volume and aggregate to daily totals.

    If the real ``prophet`` package is not available a naive average forecast is
    produced as a fallback so unit tests can run with the lightweight stubs.
    """
    df = pd.read_csv(path)
    df["ds"] = pd.to_datetime(df.iloc[:, 0], format="%m/%d/%Y %H:%M")
    df["y"] = df.iloc[:, 1].astype(float)

    hourly_metrics = None

    if Prophet is not None and hasattr(Prophet, "fit"):
        prior = 10.0
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            seasonality_prior_scale=prior,
        )
        model.fit(df[["ds", "y"]])
        future = model.make_future_dataframe(periods=periods, freq="H")
        forecast = model.predict(future)
        hourly_metrics = _hourly_mae_by_hour(df[["ds", "y"]], forecast)
        if hourly_metrics["bias"].abs().max() > bias_threshold:
            prior *= 2
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                seasonality_prior_scale=prior,
            )
            model.fit(df[["ds", "y"]])
            forecast = model.predict(future)
            hourly_metrics = _hourly_mae_by_hour(df[["ds", "y"]], forecast)
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
        hourly_metrics = _hourly_mae_by_hour(df[["ds", "y"]], forecast)

    daily = forecast.set_index("ds").resample("D")["yhat"].sum()
    return model, forecast, daily, hourly_metrics


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
