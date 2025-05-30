import pytest
pytest.importorskip("pandas")

from pathlib import Path

import pandas as pd
from prophet_analysis import load_time_series


def test_ingestion_includes_weekends():
    calls = load_time_series(Path('calls.csv'), metric='call')
    visits = load_time_series(Path('visitors.csv'), metric='visit')
    assert any(calls.index.dayofweek >= 5)
    assert any(visits.index.dayofweek >= 5)


def test_load_time_series_no_header(tmp_path):
    df = pd.read_csv('calls.csv')
    no_header = tmp_path / 'no_header.csv'
    df.to_csv(no_header, index=False, header=False)
    series = load_time_series(no_header, metric='call')
    assert not series.empty
    expected_len = (series.index.max() - series.index.min()).days + 1
    assert len(series) == expected_len
