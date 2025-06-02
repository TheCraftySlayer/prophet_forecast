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
    export_baseline_forecast,
    export_prophet_forecast,
    model_to_json,
    write_summary,
)


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

    df, regressors = prepare_data(
        call_path, visit_path, chat_path, events=cfg.get("events", {}), scale_features=True
    )
    prophet_df = prepare_prophet_data(df)

    prophet_kwargs = build_prophet_kwargs(cfg["model"]) 

    best_params = tune_prophet_hyperparameters(
        prophet_df, prophet_kwargs=prophet_kwargs
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

    model, forecast, future = train_prophet_model(
        prophet_df,
        holidays,
        regressors,
        future_periods=30,
        model_params=model_params,
        prophet_kwargs=prophet_kwargs,
        likelihood=cfg["model"].get("likelihood", "normal"),
        transform=cfg["model"].get("transform", "log"),
    )

    cv_params = cfg.get("cross_validation", {})
    df_cv, horizon_table, summary, diag, _scale = evaluate_prophet_model(
        model,
        prophet_df,
        cv_params=cv_params,
        forecast=forecast,
        scaler=None,
        transform=cfg["model"].get("transform", "log"),
    )
    write_summary(summary, out_dir / "summary.csv")
    write_summary(horizon_table, out_dir / "horizon_metrics.csv")
    diag.to_csv(out_dir / "ljung_box.csv", index=False)
    export_prophet_forecast(model, forecast, df, out_dir, scaler=None)
    export_baseline_forecast(df, out_dir)

    baseline_df, baseline_metrics, baseline_horizon = compute_naive_baseline(df)
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
    metrics = pd.concat(
        [
            metrics_baseline[["model", "horizon", "MAE", "RMSE", "MAPE", "coverage"]],
            prophet_metrics[["model", "horizon", "MAE", "RMSE", "MAPE", "coverage"]],
        ],
        ignore_index=True,
    )
    write_summary(metrics, out_dir / "metrics.csv")

    if cfg["model"].get("weekly_incremental") and model_to_json is not None:
        with open(base_out / "latest_model.json", "w") as f:
            f.write(model_to_json(model))

    commit = safe_git_hash()
    checksums = {
        "calls": _checksum(call_path),
        "visitors": _checksum(visit_path),
        "queries": _checksum(chat_path),
    }
    logger.info("Run %s", run_id)
    if commit is not None:
        logger.info("commit: %s", commit)
    logger.info("checksums: %s", json.dumps(checksums))
    logger.info("params: %s", json.dumps(model_params))


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
    args = p.parse_args()
    pipeline(args.config)
