import pandas as pd
from prophet_analysis import blend_short_term


def test_blend_short_term():
    forecast = pd.DataFrame({
        'ds': pd.date_range('2024-01-01', periods=2, freq='D'),
        'yhat': [100.0, 200.0],
        'yhat_lower': [90.0, 190.0],
        'yhat_upper': [110.0, 210.0],
    })
    history = pd.DataFrame(
        {'call_count': range(14)},
        index=pd.date_range('2023-12-18', periods=14, freq='D'),
    )
    blended = blend_short_term(forecast, history, weight=0.5)
    assert blended.loc[0, 'yhat'] == 56.5
    assert blended.loc[1, 'yhat'] == 200.0

