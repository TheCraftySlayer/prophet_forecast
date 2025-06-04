import pytest
pytest.importorskip("pandas")

from pathlib import Path
from prophet_analysis import compute_interval_accuracy


def test_interval_accuracy_15():
    metrics = compute_interval_accuracy(Path('15_Minute_Call_Counts.csv'))
    assert 'WAPE' in metrics['metric'].tolist()


def test_interval_accuracy_30():
    metrics = compute_interval_accuracy(Path('30_Minute_Call_Counts.csv'))
    assert 'WAPE' in metrics['metric'].tolist()
