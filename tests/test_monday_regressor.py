from pathlib import Path
from prophet_analysis import prepare_data


def test_monday_effect_regressor_present():
    df, regs = prepare_data(Path('calls.csv'), Path('visitors.csv'), Path('queries.csv'))
    assert 'monday_effect' in regs.columns
    assert regs['monday_effect'].abs().sum() > 0
