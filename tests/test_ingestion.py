import pandas as pd
from pathlib import Path
from prophet_analysis import load_time_series


def test_ingestion_weekdays():
    calls = load_time_series(Path('calls.csv'), metric='call')
    visits = load_time_series(Path('visitors.csv'), metric='visit')
    queries = pd.read_csv('queries.csv')['date']
    calls_idx = calls.index
    visits_idx = visits.index
    queries_idx = pd.to_datetime(queries, format="%m/%d/%y")
    assert all(calls_idx.dayofweek < 5)
    assert all(visits_idx.dayofweek < 5)
    assert all(queries_idx.dt.dayofweek < 5)
