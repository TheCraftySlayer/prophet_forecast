import pytest
pytest.importorskip("pandas")

from pathlib import Path
import pandas as pd
from prophet_analysis import compute_naive_baseline, load_time_series


def test_csv_baseline_coverage_range():
    calls = load_time_series(Path('calls.csv'), metric='call')
    df = pd.DataFrame({'call_count': calls})
    _, metrics, _ = compute_naive_baseline(df)
    coverage = metrics.loc[metrics['metric'] == 'Coverage', 'value'].iloc[0]
    assert 88 <= coverage <= 92
