import pandas as pd
from pathlib import Path
from prophet_analysis import load_time_series


def test_ingestion_includes_weekends():
    calls = load_time_series(Path('calls.csv'), metric='call')
    visits = load_time_series(Path('visitors.csv'), metric='visit')
    assert any(calls.index.dayofweek >= 5)
    assert any(visits.index.dayofweek >= 5)
