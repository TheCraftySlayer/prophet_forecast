"""Simple hourly forecasting utilities."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
from typing import Dict, Tuple


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


def hourly_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean and standard deviation for each weekday/hour pair."""
    dow = df["ds"].dt.dayofweek.rename("dow")
    hour = df["ds"].dt.hour.rename("hour")
    grouped = df.groupby([dow, hour])["y"]
    return grouped.agg(mean="mean", std="std").reset_index()


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

    # ------------------------------------------------------------------
    # Restrict to open hours: weekdays 08:00-16:59
    # ------------------------------------------------------------------
    df = df[df.ds.dt.weekday < 5]
    df = df[(df.ds.dt.hour >= 8) & (df.ds.dt.hour < 17)]

    stats = hourly_stats(df)
    mean_map: Dict[Tuple[int, int], float] = {
        (int(r.dow), int(r.hour)): float(r.mean) for r in stats.itertuples()
    }
    std_map: Dict[Tuple[int, int], float] = {
        (int(r.dow), int(r.hour)): float(r.std) for r in stats.itertuples()
    }

    df["dow"] = df["ds"].dt.dayofweek
    df["hour"] = df["ds"].dt.hour
    df["open_flag"] = (
        (df["dow"] < 5) & (df["hour"] >= 8) & (df["hour"] < 17)
    ).astype(int)
    df["mean_hour"] = [mean_map[(d, h)] for d, h in zip(df["dow"], df["hour"])]
    df["std_hour"] = [std_map[(d, h)] for d, h in zip(df["dow"], df["hour"])]
    df["recent_dev"] = df["y"].shift(24) - df["mean_hour"]
    df.dropna(subset=["recent_dev"], inplace=True)

    hourly_metrics = None

    if Prophet is not None and hasattr(Prophet, "fit"):
        prior = 10.0
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            seasonality_prior_scale=prior,
        )
        model.add_regressor("open_flag", prior_scale=20)
        model.add_regressor("mean_hour", prior_scale=5, standardize=False)

        model.fit(df[["ds", "y", "open_flag", "mean_hour"]])
        future = model.make_future_dataframe(periods=periods, freq="H")
        future["dow"] = future["ds"].dt.dayofweek
        future["hour"] = future["ds"].dt.hour
        future["open_flag"] = (
            (future["dow"] < 5) & (future["hour"] >= 8) & (future["hour"] < 17)
        ).astype(int)
        future["mean_hour"] = [
            mean_map.get((int(d), int(h)), df["mean_hour"].mean())
            for d, h in zip(future["dow"], future["hour"])
        ]

        forecast = model.predict(future)
        hourly_metrics = _hourly_mae_by_hour(df[["ds", "y"]], forecast)
        if hourly_metrics["bias"].abs().max() > bias_threshold:
            prior *= 2
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                seasonality_prior_scale=prior,
            )
            model.add_regressor("open_flag", prior_scale=20)
            model.add_regressor("mean_hour", prior_scale=5, standardize=False)
            model.fit(df[["ds", "y", "open_flag", "mean_hour"]])
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
    return model, forecast, daily, hourly_metrics, stats


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Forecast hourly calls")
    p.add_argument("csv", type=Path, nargs="?", default=Path("hourly_call_data.csv"))
    p.add_argument("--periods", type=int, default=24 * 30)
    p.add_argument("--out", type=Path, default=Path("hourly_forecast.csv"))
    args = p.parse_args()

    _, fcst, daily, _, stats = forecast_hourly_to_daily(args.csv, periods=args.periods)
    fcst.to_csv(args.out, index=False)
    daily.to_csv(args.out.with_name("daily_forecast.csv"))
    stats.to_csv(args.out.with_name("hourly_stats.csv"), index=False)
