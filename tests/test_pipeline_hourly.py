import pytest
pytest.importorskip("pandas")
from pipeline import run_forecast


def test_run_forecast_hourly(tmp_path):
    cfg = {
        "data": {
            "calls": "calls.csv",
            "hourly_calls": "hourly_call_data.csv",
            "visitors": "visitors.csv",
            "queries": "queries.csv",
        },
        "output": str(tmp_path),
        "model": {"use_hourly": True, "hourly_periods": 24},
    }
    run_forecast(cfg)
    assert (tmp_path / next(tmp_path.iterdir()).name / "daily_forecast.csv").exists()

