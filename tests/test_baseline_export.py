import pandas as pd
from pathlib import Path
from prophet_analysis import load_time_series, export_baseline_forecast


def test_export_baseline(tmp_path):
    calls = load_time_series(Path('calls.csv'), metric='call')
    df = pd.DataFrame({'call_count': calls})
    export_baseline_forecast(df, tmp_path)
    assert (tmp_path / 'baseline_forecast.xlsx').exists()
    assert (tmp_path / 'baseline_forecast.csv').exists()
