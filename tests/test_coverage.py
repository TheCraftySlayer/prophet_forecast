import pytest
pytest.importorskip("pandas")

import pandas as pd
from prophet_analysis import compute_naive_baseline


def test_baseline_coverage_for_constant_series():
    dates = pd.date_range('2023-01-01', periods=30, freq='D')
    df = pd.DataFrame({'call_count': [10] * 30}, index=dates)
    _, metrics, _ = compute_naive_baseline(df)
    coverage = metrics.loc[metrics['metric'] == 'Coverage', 'value'].iloc[0]
    assert coverage == 100.0


def test_baseline_coverage_above_ninety():
    dates = pd.date_range('2023-01-01', periods=30, freq='D')
    counts = [10 + (1 if i % 2 else -1) for i in range(30)]
    df = pd.DataFrame({'call_count': counts}, index=dates)
    _, metrics, _ = compute_naive_baseline(df)
    coverage = metrics.loc[metrics['metric'] == 'Coverage', 'value'].iloc[0]
    assert coverage >= 90.0
