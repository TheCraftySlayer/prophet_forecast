import pytest
pytest.importorskip("pandas")

from pathlib import Path

from prophet_analysis import prepare_data


def test_prepare_data_creates_lag_regressors():
    df, _ = prepare_data(Path('calls.csv'), Path('visitors.csv'), Path('queries.csv'))
    expected = {'call_lag1', 'call_lag3', 'call_lag7', 'visit_lag7', 'query_lag7'}
    assert expected <= set(df.columns)
    assert df['call_lag1'].iloc[1] == df['call_count'].iloc[0]
    assert df['call_lag7'].iloc[7] == df['call_count'].iloc[0]

    import pandas as pd

    visit_exp = df['visit_count'].shift(7).fillna(0).astype(float)
    visit_exp = (visit_exp - visit_exp.mean()) / visit_exp.std()
    pd.testing.assert_series_equal(df['visit_lag7'], visit_exp, check_names=False)

    query_exp = df['chatbot_count'].shift(7).fillna(0).astype(float)
    query_exp = (query_exp - query_exp.mean()) / query_exp.std()
    pd.testing.assert_series_equal(df['query_lag7'], query_exp, check_names=False)
