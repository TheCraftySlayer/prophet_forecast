import pytest
pytest.importorskip("pandas")

from pathlib import Path

from prophet_analysis import prepare_data


def test_chatbot_counts_no_nan():
    df, _ = prepare_data(Path('calls.csv'), Path('visitors.csv'), Path('queries.csv'))
    assert not df['chatbot_count'].isna().any()
