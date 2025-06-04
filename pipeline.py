# ruff: noqa: E402
import os
for var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[var] = "1"

import sys
if __name__ == "__main__":
    # Expose this module as ``pipeline`` when executed as a script so that
    # helper modules can import it without errors.
    sys.modules.setdefault("pipeline", sys.modules[__name__])
from pathlib import Path

# By default the real third-party packages are used. Set ``USE_STUB_LIBS=1``
# to temporarily prioritise the lightweight stubs bundled with the repository
# for faster testing.
_USE_STUB_LIBS = os.getenv("USE_STUB_LIBS") == "1"
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if not _USE_STUB_LIBS:
    # When using the real libraries ensure this repository's directory is
    # searched *after* site-packages so our stub modules do not shadow the
    # installed packages.
    sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _THIS_DIR]
    sys.path.append(_THIS_DIR)

from prophet import Prophet
print("Explicit Prophet import successful:", Prophet)
from staffing_diagnostics import (
    mean_call_volumes,
    relative_mae,
    staffing_cost,
)
try:
    from ruamel.yaml import YAML  # type: ignore
except ModuleNotFoundError:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as e:  # pragma: no cover - environment missing YAML libs
        raise ModuleNotFoundError(
            "Missing YAML parser. Install ruamel.yaml or PyYAML to run this script."
        ) from e
    YAML = None
import hashlib
import json
import logging
import subprocess


def safe_git_hash() -> str | None:
    """Return the current Git commit hash or ``None`` if unavailable."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:  # pragma: no cover - missing Git repo
        return None
from datetime import datetime

import numpy as np
import pandas as pd
from holidays_calendar import get_holidays_dataframe
from prophet_analysis import (
    create_prophet_holidays,
    prepare_data,
    prepare_prophet_data,
    evaluate_prophet_model,
    train_prophet_model,
    tune_prophet_hyperparameters,
    build_prophet_kwargs,
    compute_naive_baseline,
    load_time_series,
    aggregate_hourly_calls,
    export_baseline_forecast,
    export_prophet_forecast,
    monitor_residuals,
    blend_short_term,
    model_to_json,
    write_summary,
    select_likelihood,
)
from hourly_analysis import forecast_hourly_to_daily


def configure_logging(log_file: Path) -> None:
    """Send cmdstanpy INFO logs to a file while keeping console warnings."""
    formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(formatter)
    root.addHandler(console)

    logging.getLogger("cmdstanpy").setLevel(logging.INFO)


def _checksum(path: Path) -> str:
    """Return SHA1 checksum for the given file."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: Path) -> dict:
    """Load YAML configuration using ruamel if available else PyYAML."""
    if YAML is not None:
        yaml_loader = YAML(typ="safe")
        with open(path, "r") as f:
            return yaml_loader.load(f)
    else:  # Fallback to PyYAML
        with open(path, "r") as f:
            return yaml.safe_load(f)


def run_forecast(cfg: dict) -> None:
    """Execute the forecasting pipeline using a configuration dictionary."""
    logger = logging.getLogger(__name__)

    call_path = Path(cfg["data"]["calls"])
    visit_path = Path(cfg["data"]["visitors"])
    chat_path = Path(cfg["data"]["queries"])
    base_out = Path(cfg.get("output", "prophet_output"))
    base_out.mkdir(exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_out / run_id
    out_dir.mkdir(exist_ok=True)

    configure_logging(out_dir / "cmdstan.log")

    if cfg.get("model", {}).get("use_hourly"):
        hourly_path = cfg["data"].get("hourly_calls")
        if not hourly_path:
            raise ValueError("hourly_calls path required when use_hourly is true")
        periods = cfg["model"].get("hourly_periods", 24 * 7)
        model, hourly_fcst, daily_fcst, hour_metrics = forecast_hourly_to_daily(
            Path(hourly_path), periods=periods
        )
        hourly_fcst.to_csv(out_dir / "hourly_forecast.csv", index=False)
        daily_fcst.to_csv(out_dir / "daily_forecast.csv")
        if hour_metrics is not None:
            hour_metrics.to_csv(out_dir / "hour_of_day_metrics.csv", index=False)

        df_hourly = pd.read_csv(hourly_path)
        df_hourly["ds"] = pd.to_datetime(df_hourly.iloc[:, 0], format="%m/%d/%Y %H:%M")
        df_hourly["y"] = df_hourly.iloc[:, 1].astype(float)

        # --------------------------------------------------------------
        # Restrict evaluation to open hours (weekdays 08:00-16:59)
        # --------------------------------------------------------------
        df_hourly = df_hourly[df_hourly.ds.dt.weekday < 5]
        df_hourly = df_hourly[(df_hourly.ds.dt.hour >= 8) & (df_hourly.ds.dt.hour < 17)]

        def _evaluate_hourly(df: pd.DataFrame, fcst: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
            fcst = fcst[fcst.ds.dt.weekday < 5]
            fcst = fcst[(fcst.ds.dt.hour >= 8) & (fcst.ds.dt.hour < 17)]
            joined = pd.concat(
                [
                    df.set_index("ds")["y"].resample("D").sum().rename("actual"),
                    fcst.set_index("ds").resample("D")[["yhat", "yhat_lower", "yhat_upper"]].sum(),
                ],
                axis=1,
            ).dropna()
            joined["error"] = joined["actual"] - joined["yhat"]
            joined["abs_error"] = joined["error"].abs()
            coverage = (
                (
                    (joined["actual"] >= joined["yhat_lower"])
                    & (joined["actual"] <= joined["yhat_upper"])
                ).mean()
                * 100
            )
            zero_acc = ((joined["actual"] == 0) == (joined["yhat"] < 0.5)).mean() * 100
            smape = (
                2
                * joined["abs_error"]
                /
                (joined["actual"].abs() + joined["yhat"].abs())
            ).replace([np.inf, -np.inf], np.nan).mean() * 100
            summary = pd.DataFrame(
                {
                    "metric": [
                        "MAE",
                        "RMSE",
                        "sMAPE",
                        "Coverage",
                        "ZeroAcc",
                        "MAE_pct",
                        "Cost",
                    ],
                    "value": [
                        joined["abs_error"].mean(),
                        np.sqrt((joined["error"] ** 2).mean()),
                        smape,
                        coverage,
                        zero_acc,
                        float("nan"),  # filled later
                        float("nan"),
                    ],
                }
            )
            mean_daily, mean_hourly = mean_call_volumes(df["ds"], df["y"])
            summary.loc[summary["metric"] == "MAE_pct", "value"] = relative_mae(
                joined["abs_error"], mean_daily
            )
            summary.loc[summary["metric"] == "Cost", "value"] = staffing_cost(
                joined["actual"],
                joined["yhat"],
                mean_daily,
                understaff_penalty=2.0,
                overstaff_penalty=1.0,
            )
            horizon_rows = []
            for h in [1, 7, 14]:
                if len(joined) >= h:
                    sub = joined.head(h)
                    horizon_rows.append(
                        [
                            h,
                            sub["abs_error"].mean(),
                            np.sqrt((sub["error"] ** 2).mean()),
                            (
                                2
                                * sub["abs_error"]
                                /
                                (sub["actual"].abs() + sub["yhat"].abs())
                            ).replace([np.inf, -np.inf], np.nan).mean()
                            * 100,
                            ((sub["actual"] == 0) == (sub["yhat"] < 0.5)).mean() * 100,
                        ]
                    )
            horizon_df = pd.DataFrame(
                horizon_rows, columns=["horizon_days", "MAE", "RMSE", "sMAPE", "ZeroAcc"]
            )
            return summary, horizon_df

        summary, horizon_table = _evaluate_hourly(df_hourly, hourly_fcst)
        write_summary(summary, out_dir / "summary.csv")
        write_summary(horizon_table, out_dir / "horizon_metrics.csv")

        daily_actual = df_hourly.set_index("ds")["y"].resample("D").sum()
        df_daily = pd.DataFrame({"call_count": daily_actual})
        export_baseline_forecast(df_daily, out_dir)

        def _ensure_smape(table: pd.DataFrame, obs: str = "actual", pred: str = "yhat"):
            if "sMAPE" not in table.columns and {obs, pred}.issubset(table.columns):
                smape = (
                    2
                    * (table[obs] - table[pred]).abs()
                    /
                    (table[obs].abs() + table[pred].abs())
                ).replace([np.inf, -np.inf], np.nan).mean() * 100
                table["sMAPE"] = smape

        _ensure_smape(horizon_table)

        baseline_df, baseline_metrics, baseline_horizon = compute_naive_baseline(
            df_daily,
            hourly_df=df_hourly[["ds", "y"]],
        )
        _ensure_smape(baseline_horizon, obs="call_count", pred="predicted")
        cov_b = baseline_metrics.loc[
            baseline_metrics["metric"] == "Coverage", "value"
        ].iloc[0]
        metrics_baseline = baseline_horizon.rename(
            columns={"horizon_days": "horizon"}
        ).copy()
        metrics_baseline["model"] = "baseline"
        metrics_baseline["coverage"] = cov_b

        coverage = summary.loc[summary["metric"] == "Coverage", "value"].iloc[0]
        prophet_metrics = horizon_table.rename(columns={"horizon_days": "horizon"}).copy()
        prophet_metrics["model"] = "prophet"
        prophet_metrics["coverage"] = coverage

        wanted = ["model", "horizon", "MAE", "RMSE", "sMAPE", "coverage", "ZeroAcc"]
        metrics_baseline = metrics_baseline[[c for c in wanted if c in metrics_baseline]]
        prophet_metrics = prophet_metrics[[c for c in wanted if c in prophet_metrics]]

        metrics = pd.concat([metrics_baseline, prophet_metrics], ignore_index=True)
        write_summary(metrics, out_dir / "metrics.csv")

        if cfg["model"].get("weekly_incremental") and model_to_json is not None:
            with open(base_out / "latest_model.json", "w") as f:
                f.write(model_to_json(model))

        train_window = {
            "start": df_hourly["ds"].min().strftime("%Y-%m-%d"),
            "end": df_hourly["ds"].max().strftime("%Y-%m-%d"),
        }
        model_hash = None
        if model_to_json is not None:
            model_hash = hashlib.sha1(
                model_to_json(model).encode("utf-8")
            ).hexdigest()

        commit = safe_git_hash()
        checksums = {
            "calls": _checksum(call_path),
            "visitors": _checksum(visit_path),
            "queries": _checksum(chat_path),
        }
        overrides = {
            k: v for k, v in cfg.get("model", {}).items() if k not in {"use_hourly"}
        }
        logger.info("Run %s", run_id)
        if commit is not None:
            logger.info("commit: %s", commit)
        logger.info("model_hash: %s", model_hash)
        logger.info("training_window: %s", json.dumps(train_window))
        logger.info("checksums: %s", json.dumps(checksums))
        logger.info("overrides: %s", json.dumps(overrides))
        logger.info("params: %s", json.dumps({"hourly_periods": periods}))
        return

    df, regressors = prepare_data(
        call_path,
        visit_path,
        chat_path,
        events=cfg.get("events", {}),
        scale_features=True,
        hourly_call_path=Path(cfg["data"].get("hourly_calls")) if cfg["data"].get("hourly_calls") else None,
    )
    prophet_df = prepare_prophet_data(df)

    prophet_kwargs = build_prophet_kwargs(cfg["model"])
    prophet_kwargs.setdefault(
        "stan_backend",
        cfg["model"].get("stan_backend", "cmdstanpy"),
    )

    best_params = tune_prophet_hyperparameters(
        prophet_df,
        prophet_kwargs=prophet_kwargs,
        cv_params=cfg.get("cross_validation", {}),
    )
    model_params = {
        "seasonality_mode": cfg["model"]["seasonality_mode"],
        "seasonality_prior_scale": cfg["model"]["seasonality_prior_scale"],
        "holidays_prior_scale": cfg["model"]["holidays_prior_scale"],
        "changepoint_prior_scale": cfg["model"]["changepoint_prior_scale"],
        "n_changepoints": cfg["model"].get("n_changepoints", 8),
        "changepoint_range": cfg["model"].get("changepoint_range", 0.8),
        "mcmc_samples": cfg["model"]["mcmc_samples"],
        "interval_width": cfg["model"].get("interval_width", 0.9),
        "growth": cfg["model"].get("growth", "linear"),
        "regressor_prior_scale": cfg["model"].get("regressor_prior_scale", 0.05),
        "capacity": cfg["model"].get("capacity"),
        "changepoints": cfg.get("events", {}).get("changepoints"),
    }
    model_params.update(best_params)

    if not cfg["model"].get("enable_mcmc", False) and model_params.get("mcmc_samples", 0):
        logger.warning("Ignoring mcmc_samples because enable_mcmc is false")
        model_params["mcmc_samples"] = 0

    idx = df.index
    holiday_df = get_holidays_dataframe()
    mask = (
        (holiday_df["event"] == "county_holiday")
        & (holiday_df["date"] >= idx.min())
        & (holiday_df["date"] <= idx.max())
    )
    holiday_dates = holiday_df.loc[mask, "date"]
    deadline_dates = pd.date_range(start=idx.min(), end=idx.max(), freq="MS")
    closure_dates = df.index[df.get("closure_flag", 0) == 1]
    holidays = create_prophet_holidays(
        holiday_dates,
        deadline_dates,
        closure_dates=closure_dates,
        press_release_dates=[],
    )

    likelihood = cfg["model"].get("likelihood", "auto")
    if likelihood == "auto":
        likelihood = select_likelihood(df["call_count"])

    model, forecast, future = train_prophet_model(
        prophet_df,
        holidays,
        regressors,
        future_periods=30,
        model_params=model_params,
        prophet_kwargs=prophet_kwargs,
        likelihood=likelihood,
        transform=cfg["model"].get("transform", "log"),
    )

    cv_params = cfg.get("cross_validation", {})
    df_cv, horizon_table, summary, diag, _scale = evaluate_prophet_model(
        model,
        prophet_df,
        cv_params=cv_params,
        forecast=None,
        scaler=None,
        transform=cfg["model"].get("transform", "log"),
    )
    write_summary(summary, out_dir / "summary.csv")
    write_summary(horizon_table, out_dir / "horizon_metrics.csv")
    diag.to_csv(out_dir / "ljung_box.csv", index=False)
    forecast_blend = blend_short_term(forecast, df)
    pred_path, pred_df = export_prophet_forecast(
        model, forecast_blend, df, out_dir, scaler=None
    )
    flagged = monitor_residuals(pred_df)
    if not flagged.empty:
        logger.warning(
            "Residuals exceeded threshold on %s", 
            ", ".join(flagged['ds'].dt.strftime('%Y-%m-%d'))
        )
        logger.info("Retraining model due to residual spike")
        model, forecast, future = train_prophet_model(
            prophet_df,
            holidays,
            regressors,
            future_periods=30,
            model_params=model_params,
            prophet_kwargs=prophet_kwargs,
            likelihood=likelihood,
            transform=cfg["model"].get("transform", "log"),
        )
        forecast_blend = blend_short_term(forecast, df)
        export_prophet_forecast(model, forecast_blend, df, out_dir, scaler=None)
    export_baseline_forecast(df, out_dir)

 # ------------------------------------------------------------------
    # Guarantee that both result-tables contain a “sMAPE” column
    # ------------------------------------------------------------------

    def _ensure_smape(table: pd.DataFrame, obs: str = "y", pred: str = "yhat"):
        """
        Add a constant sMAPE column (one value per table) if it's missing.
        The helpers upstream sometimes don’t compute sMAPE.
        """
        if "sMAPE" not in table.columns and {obs, pred}.issubset(table.columns):
            smape = (
                2
                * (table[obs] - table[pred]).abs()
                /
                (table[obs].abs() + table[pred].abs())
            ).replace([np.inf, -np.inf], np.nan).mean() * 100
            table["sMAPE"] = smape

    _ensure_smape(horizon_table)

    baseline_df, baseline_metrics, baseline_horizon = compute_naive_baseline(df)
    _ensure_smape(baseline_horizon)
    cov_b = baseline_metrics.loc[
        baseline_metrics["metric"] == "Coverage", "value"
    ].iloc[0]
    metrics_baseline = baseline_horizon.rename(
        columns={"horizon_days": "horizon"}
    ).copy()
    metrics_baseline["model"] = "baseline"
    metrics_baseline["coverage"] = cov_b

    coverage = summary.loc[summary["metric"] == "Coverage", "value"].iloc[0]
    prophet_metrics = horizon_table.rename(columns={"horizon_days": "horizon"}).copy()
    prophet_metrics["model"] = "prophet"
    prophet_metrics["coverage"] = coverage

    wanted = ["model", "horizon", "MAE", "RMSE", "sMAPE", "coverage", "ZeroAcc"]
    metrics_baseline = metrics_baseline[[c for c in wanted if c in metrics_baseline]]
    prophet_metrics  = prophet_metrics [[c for c in wanted if c in prophet_metrics ]]

    metrics = pd.concat(
        [metrics_baseline, prophet_metrics],
        ignore_index=True,
    )
    write_summary(metrics, out_dir / "metrics.csv")

    if cfg["model"].get("weekly_incremental") and model_to_json is not None:
        with open(base_out / "latest_model.json", "w") as f:
            f.write(model_to_json(model))

    train_window = {
        "start": df.index.min().strftime("%Y-%m-%d"),
        "end": df.index.max().strftime("%Y-%m-%d"),
    }
    model_hash = None
    if model_to_json is not None:
        model_hash = hashlib.sha1(model_to_json(model).encode("utf-8")).hexdigest()

    commit = safe_git_hash()
    checksums = {
        "calls": _checksum(call_path),
        "visitors": _checksum(visit_path),
        "queries": _checksum(chat_path),
    }
    overrides = {
        k: v for k, v in cfg.get("model", {}).items() if k not in {"use_hourly"}
    }
    logger.info("Run %s", run_id)
    if commit is not None:
        logger.info("commit: %s", commit)
    logger.info("model_hash: %s", model_hash)
    logger.info("training_window: %s", json.dumps(train_window))
    logger.info("checksums: %s", json.dumps(checksums))
    logger.info("overrides: %s", json.dumps(overrides))
    logger.info("params: %s", json.dumps(model_params))


def check_baseline_coverage(config_path: Path) -> None:
    """Print naive baseline coverage and exit if outside 88‑92 percent.

    Uses ``pandas`` from the open‑source ecosystem to load the data.
    """
    logger = logging.getLogger(__name__)
    cfg = load_config(config_path)
    calls = load_time_series(Path(cfg["data"]["calls"]), metric="call")
    df = pd.DataFrame({"call_count": calls})
    hourly_df = None
    hourly_path = cfg["data"].get("hourly_calls")
    if hourly_path:
        hourly = aggregate_hourly_calls(Path(hourly_path))
        if not hourly.empty:
            hourly_df = pd.DataFrame({"ds": hourly.index, "y": hourly.values})
    try:
        _, metrics, _ = compute_naive_baseline(df, hourly_df=hourly_df)
    except ValueError as exc:
        logger.error("Unable to compute baseline coverage: %s", exc)
        raise SystemExit(1)

    coverage = metrics.loc[metrics["metric"] == "Coverage", "value"].iloc[0]
    print(f"Naive baseline coverage: {coverage:.2f}%")
    if coverage < 88 or coverage > 92:
        raise SystemExit(1)


def pipeline(config_path: Path) -> None:
    """Load configuration from ``config_path`` and run the forecast pipeline."""
    cfg = load_config(config_path)
    run_forecast(cfg)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()

    import argparse

    p = argparse.ArgumentParser(description="Run forecast pipeline")
    p.add_argument("config", type=Path, default=Path("config.yaml"), nargs="?")
    p.add_argument(
        "--check-baseline-coverage",
        action="store_true",
        help="Verify naive baseline coverage is between 88 and 92",
    )
    p.add_argument(
        "--baseline-coverage",
        action="store_true",
        help="Print naive baseline coverage and exit 1 if outside 88-92",
    )
    args = p.parse_args()
    if args.baseline_coverage or args.check_baseline_coverage:
        check_baseline_coverage(args.config)
    else:
        pipeline(args.config)
