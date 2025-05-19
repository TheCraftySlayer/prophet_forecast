from pathlib import Path
try:
    from ruamel.yaml import YAML
except Exception:  # pragma: no cover - optional dependency may be missing
    import yaml
    YAML = None
from prophet_analysis import (
    prepare_data,
    prepare_prophet_data,
    train_prophet_model,
    evaluate_prophet_model,
    tune_prophet_hyperparameters,
    create_prophet_holidays,
    export_prophet_forecast,
    export_baseline_forecast,
)
from datetime import datetime
import pandas as pd
from holidays_calendar import get_holidays_dataframe
import logging


def load_config(path: Path) -> dict:
    """Load YAML configuration using ruamel if available else PyYAML."""
    if YAML is not None:
        yaml_loader = YAML(typ="safe")
        with open(path, "r") as f:
            return yaml_loader.load(f)
    else:  # Fallback to PyYAML
        with open(path, "r") as f:
            return yaml.safe_load(f)


def pipeline(config_path: Path):
    cfg = load_config(config_path)
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

    call_path = Path(cfg['data']['calls'])
    visit_path = Path(cfg['data']['visitors'])
    chat_path = Path(cfg['data']['queries'])
    base_out = Path(cfg.get('output', 'prophet_output'))
    base_out.mkdir(exist_ok=True)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = base_out / run_id
    out_dir.mkdir(exist_ok=True)

    df, regressors = prepare_data(call_path, visit_path, chat_path, scale_features=True)
    prophet_df = prepare_prophet_data(df)

    best_params = tune_prophet_hyperparameters(prophet_df)
    model_params = {
        'seasonality_mode': cfg['model']['seasonality_mode'],
        'seasonality_prior_scale': cfg['model']['seasonality_prior_scale'],
        'holidays_prior_scale': cfg['model']['holidays_prior_scale'],
        'changepoint_prior_scale': cfg['model']['changepoint_prior_scale'],
        'n_changepoints': cfg['model'].get('n_changepoints', 8),
        'mcmc_samples': cfg['model']['mcmc_samples'],
        'interval_width': cfg['model'].get('interval_width', 0.8),
        'weekly_seasonality': cfg['model']['weekly_seasonality'],
        'yearly_seasonality': cfg['model']['yearly_seasonality'],
        'daily_seasonality': cfg['model']['daily_seasonality'],
        'growth': cfg['model'].get('growth', 'linear'),
        'regressor_prior_scale': cfg['model'].get('regressor_prior_scale', 0.05),
    }
    model_params.update(best_params)

    idx = df.index
    holiday_df = get_holidays_dataframe()
    mask = (holiday_df['event'] == 'county_holiday') & \
           (holiday_df['date'] >= idx.min()) & (holiday_df['date'] <= idx.max())
    holiday_dates = holiday_df.loc[mask, 'date']
    deadline_dates = pd.date_range(start=idx.min(), end=idx.max(), freq='MS')
    holidays = create_prophet_holidays(holiday_dates, deadline_dates, [])

    model, forecast, future = train_prophet_model(
        prophet_df,
        holidays,
        regressors,
        future_periods=30,
        model_params=model_params,
        log_transform=True,
        likelihood=cfg['model'].get('likelihood', 'normal'),
    )

    cv_params = cfg.get('cross_validation', {})
    df_cv, horizon_table, summary, diag = evaluate_prophet_model(
        model,
        prophet_df,
        cv_params=cv_params,
        log_transform=True,
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    horizon_table.to_csv(out_dir / "horizon_metrics.csv", index=False)
    diag.to_csv(out_dir / "ljung_box.csv", index=False)
    export_prophet_forecast(model, forecast, df, out_dir)
    export_baseline_forecast(df, out_dir)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run forecast pipeline")
    p.add_argument("config", type=Path, default=Path("config.yaml"), nargs="?")
    args = p.parse_args()
    pipeline(args.config)
