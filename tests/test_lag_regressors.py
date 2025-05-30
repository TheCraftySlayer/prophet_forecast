import pytest
pytest.importorskip("pandas")

from pathlib import Path

from prophet_analysis import prepare_data


def test_prepare_data_creates_lag_regressors():
    df, _ = prepare_data(Path('calls.csv'), Path('visitors.csv'), Path('queries.csv'))
    assert {'call_lag1', 'call_lag3', 'call_lag7'} <= set(df.columns)
    assert df['call_lag1'].iloc[1] == df['call_count'].iloc[0]
    assert df['call_lag7'].iloc[7] == df['call_count'].iloc[0]
