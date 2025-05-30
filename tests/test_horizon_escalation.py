import pytest
pytest.importorskip("pandas")

import logging
import pandas as pd
from prophet_analysis import _check_horizon_escalation


def test_horizon_escalation_warning(caplog):
    df = pd.DataFrame({
        'horizon_days': [1, 14],
        'MAE': [10.0, 13.0],
        'RMSE': [0.0, 0.0],
    })
    with caplog.at_level(logging.WARNING):
        _check_horizon_escalation(df, threshold=0.2)
    assert any('14-day horizon MAE escalates' in r.message for r in caplog.records)


def test_horizon_escalation_ok(caplog):
    df = pd.DataFrame({
        'horizon_days': [1, 14],
        'MAE': [10.0, 11.0],
        'RMSE': [0.0, 0.0],
    })
    with caplog.at_level(logging.WARNING):
        _check_horizon_escalation(df, threshold=0.2)
    assert not any('14-day horizon MAE escalates' in r.message for r in caplog.records)
