import pandas as pd
from unittest.mock import patch

from prophet_analysis import compute_naive_baseline, evaluate_prophet_model
from tests.test_pipeline_alignment import DummyProphet


def test_baseline_returns_smape():
    dates = pd.date_range('2023-01-01', periods=30, freq='D')
    df = pd.DataFrame({'call_count': range(30)}, index=dates)
    _, metrics, horizon = compute_naive_baseline(df)
    assert 'sMAPE' in metrics['metric'].tolist()
    assert 'sMAPE' in horizon.columns


def _stub_cv(*_args, **_kwargs):
    return pd.DataFrame({
        'ds': pd.date_range('2023-01-01', periods=3, freq='D'),
        'y': [1.0, 2.0, 3.0],
        'yhat': [1.0, 2.0, 3.0],
    })


def _lb_mid(residuals, lags=14, return_df=True):
    return pd.DataFrame({'lb_stat': [0.0] * lags, 'lb_pvalue': [0.5] * lags})


def test_prophet_evaluation_includes_smape():
    model = DummyProphet()
    prophet_df = pd.DataFrame({'ds': pd.date_range('2023-01-01', periods=3), 'y': [1, 2, 3]})
    with patch('prophet_analysis.cross_validation_func', side_effect=_stub_cv), \
         patch('prophet_analysis.acorr_ljungbox', side_effect=_lb_mid), \
         patch('prophet_analysis._fit_prophet_with_fallback'), \
         patch('prophet_analysis._ensure_tbb_on_path'):
        _, horizon_table, summary, _, _ = evaluate_prophet_model(model, prophet_df)
    assert 'sMAPE' in summary['metric'].tolist()
    assert 'sMAPE' in horizon_table.columns
