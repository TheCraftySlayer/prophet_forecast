from pathlib import Path
from ruamel.yaml import YAML
from prophet_analysis import (
    prepare_data,
    prepare_prophet_data,
    train_prophet_model,
    evaluate_prophet_model,
    tune_prophet_hyperparameters,
    create_prophet_holidays,
    export_prophet_forecast,
)
from datetime import date
import logging


def load_config(path: Path) -> dict:
    yaml = YAML(typ="safe")
    with open(path, "r") as f:
        return yaml.load(f)


def pipeline(config_path: Path):
    cfg = load_config(config_path)
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

    call_path = Path(cfg['data']['calls'])
    visit_path = Path(cfg['data']['visitors'])
    chat_path = Path(cfg['data']['queries'])
    out_dir = Path(cfg.get('output', 'prophet_output'))
    out_dir.mkdir(exist_ok=True)

    df, regressors = prepare_data(call_path, visit_path, chat_path, scale_features=True)
    prophet_df = prepare_prophet_data(df)

    best_params = tune_prophet_hyperparameters(prophet_df)
    model_params = {
        'seasonality_mode': cfg['model']['seasonality_mode'],
        'seasonality_prior_scale': cfg['model']['seasonality_prior_scale'],
        'holidays_prior_scale': cfg['model']['holidays_prior_scale'],
        'changepoint_prior_scale': cfg['model']['changepoint_prior_scale'],
        'mcmc_samples': cfg['model']['mcmc_samples'],
        'weekly_seasonality': cfg['model']['weekly_seasonality'],
        'yearly_seasonality': cfg['model']['yearly_seasonality'],
        'daily_seasonality': cfg['model']['daily_seasonality'],
    }
    model_params.update(best_params)

    holiday_dates = [date(2024,1,1)]
    deadline_dates = []
    press_release_dates = []
    holidays = create_prophet_holidays(holiday_dates, deadline_dates, press_release_dates)

    model, forecast, future = train_prophet_model(
        prophet_df,
        holidays,
        regressors,
        future_periods=30,
        model_params=model_params,
    )

    evaluate_prophet_model(model, prophet_df)
    export_prophet_forecast(model, forecast, df, out_dir)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run forecast pipeline")
    p.add_argument("config", type=Path, default=Path("config.yaml"), nargs="?")
    args = p.parse_args()
    pipeline(args.config)
