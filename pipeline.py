import os
import sys
from pathlib import Path

# If the USE_REAL_LIBS environment variable is set, temporarily remove this
# directory from ``sys.path`` so the real third-party packages are imported
# instead of the lightweight stub modules bundled with the repository.
_USE_REAL_LIBS = os.getenv("USE_REAL_LIBS") == "1"
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _USE_REAL_LIBS:
    # When using the real third-party libraries, ensure this repository's
    # directory is searched **after** the standard site-packages so our stub
    # modules do not shadow the installed packages.  Remove the path first and
    # append it to the end of ``sys.path``.
    sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _THIS_DIR]
    sys.path.append(_THIS_DIR)
try:
    from ruamel.yaml import YAML
except Exception:  # pragma: no cover - optional dependency may be missing
    import yaml

    YAML = None
import hashlib
import json
import logging
import pickle
import subprocess
from datetime import datetime

import pandas as pd
from data_preparation import create_prophet_holidays, prepare_data, prepare_prophet_data
from holidays_calendar import get_holidays_dataframe
from modeling import (
    evaluate_prophet_model,
    train_prophet_model,
    tune_prophet_hyperparameters,
)
from prophet_analysis import (
    PROPHET_KWARGS,
    compute_naive_baseline,
    export_baseline_forecast,
    export_prophet_forecast,
    model_to_json,
)
from sklearn.preprocessing import StandardScaler


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
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s"
    )

    call_path = Path(cfg["data"]["calls"])
    visit_path = Path(cfg["data"]["visitors"])
    chat_path = Path(cfg["data"]["queries"])
    base_out = Path(cfg.get("output", "prophet_output"))
    base_out.mkdir(exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_out / run_id
    out_dir.mkdir(exist_ok=True)

    df, regressors = prepare_data(call_path, visit_path, chat_path, scale_features=True)
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled["call_count"] = scaler.fit_transform(df[["call_count"]])
    with open(out_dir / "call_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    prophet_df = prepare_prophet_data(df_scaled)

    best_params = tune_prophet_hyperparameters(
        prophet_df, prophet_kwargs=PROPHET_KWARGS
    )
    model_params = {
        "seasonality_mode": cfg["model"]["seasonality_mode"],
        "seasonality_prior_scale": cfg["model"]["seasonality_prior_scale"],
        "holidays_prior_scale": cfg["model"]["holidays_prior_scale"],
        "changepoint_prior_scale": cfg["model"]["changepoint_prior_scale"],
        "n_changepoints": cfg["model"].get("n_changepoints", 8),
        "changepoint_range": cfg["model"].get("changepoint_range", 0.8),
        "mcmc_samples": cfg["model"]["mcmc_samples"],
        "interval_width": cfg["model"].get("interval_width", 0.8),
        "growth": cfg["model"].get("growth", "linear"),
        "regressor_prior_scale": cfg["model"].get("regressor_prior_scale", 0.05),
    }
    model_params.update(best_params)

    idx = df.index
    holiday_df = get_holidays_dataframe()
    mask = (
        (holiday_df["event"] == "county_holiday")
        & (holiday_df["date"] >= idx.min())
        & (holiday_df["date"] <= idx.max())
    )
    holiday_dates = holiday_df.loc[mask, "date"]
    deadline_dates = pd.date_range(start=idx.min(), end=idx.max(), freq="MS")
    holidays = create_prophet_holidays(holiday_dates, deadline_dates, [])

    model, forecast, future = train_prophet_model(
        prophet_df,
        holidays,
        regressors,
        future_periods=30,
        model_params=model_params,
        prophet_kwargs=PROPHET_KWARGS,
        log_transform=True,
        likelihood=cfg["model"].get("likelihood", "normal"),
    )

    cv_params = cfg.get("cross_validation", {})
    df_cv, horizon_table, summary, diag, _scale = evaluate_prophet_model(
        model,
        prophet_df,
        cv_params=cv_params,
        log_transform=True,
        forecast=forecast,
        scaler=scaler,
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    horizon_table.to_csv(out_dir / "horizon_metrics.csv", index=False)
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
            metrics_baseline,
            prophet_metrics[["model", "horizon", "MAE", "RMSE", "coverage"]],
        ],
        ignore_index=True,
    )
    metrics.to_csv(out_dir / "metrics.csv", index=False)

    if cfg["model"].get("weekly_incremental") and model_to_json is not None:
        with open(base_out / "latest_model.json", "w") as f:
            f.write(model_to_json(model))

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    checksums = {
        "calls": _checksum(call_path),
        "visitors": _checksum(visit_path),
        "queries": _checksum(chat_path),
    }
    log_path = Path("model_log.md")
    with open(log_path, "a") as log_f:
        log_f.write(f"\n### Run {run_id}\n")
        log_f.write(f"- commit: {commit}\n")
        log_f.write(f"- checksums: {json.dumps(checksums)}\n")
        log_f.write(f"- params: {json.dumps(model_params)}\n")


def pipeline(config_path: Path) -> None:
    """Load configuration from ``config_path`` and run the forecast pipeline."""
    cfg = load_config(config_path)
    run_forecast(cfg)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Run forecast pipeline")
    p.add_argument("config", type=Path, default=Path("config.yaml"), nargs="?")
    args = p.parse_args()
    pipeline(args.config)
