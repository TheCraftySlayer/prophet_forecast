import pandas as pd
from unittest.mock import patch

from prophet_analysis import evaluate_prophet_model
from tests.test_pipeline_alignment import DummyProphet


def _stub_cv(*_args, **_kwargs):
    return pd.DataFrame({
        'ds': pd.date_range('2023-01-01', periods=3, freq='D'),
        'y': [1.0, 2.0, 3.0],
        'yhat': [1.0, 2.0, 3.0],
    })


def _lb_mid(residuals, lags=14, return_df=True):
    return pd.DataFrame({'lb_stat': [0.0] * lags, 'lb_pvalue': [0.5] * lags})


def test_stop_on_mid_range_pvalue():
    model = DummyProphet()
    prophet_df = pd.DataFrame({'ds': pd.date_range('2023-01-01', periods=3), 'y': [1, 2, 3]})
    with patch('prophet_analysis.cross_validation', side_effect=_stub_cv) as cv_mock, \
         patch('prophet_analysis.acorr_ljungbox', side_effect=_lb_mid), \
         patch('prophet_analysis._fit_prophet_with_fallback'), \
         patch('prophet_analysis._ensure_tbb_on_path'):
        evaluate_prophet_model(model, prophet_df)
        assert cv_mock.call_count == 1
        assert model.changepoint_prior_scale == 0.2


def test_changepoint_scale_persist_after_refit():
    model = DummyProphet(changepoint_prior_scale=0.4)
    prophet_df = pd.DataFrame({'ds': pd.date_range('2023-01-01', periods=3), 'y': [1, 2, 3]})

    def lb_side_effect(residuals, lags=14, return_df=True):
        if lb_side_effect.calls == 0:
            lb_side_effect.calls += 1
            return pd.DataFrame({'lb_stat': [1.0] * lags, 'lb_pvalue': [0.01] * lags})
        return pd.DataFrame({'lb_stat': [0.0] * lags, 'lb_pvalue': [0.5] * lags})

    lb_side_effect.calls = 0

    with patch('prophet_analysis.cross_validation', side_effect=_stub_cv) as cv_mock, \
         patch('prophet_analysis.acorr_ljungbox', side_effect=lb_side_effect), \
         patch('prophet_analysis._fit_prophet_with_fallback'), \
         patch('prophet_analysis._ensure_tbb_on_path'):
        evaluate_prophet_model(model, prophet_df)
        assert cv_mock.call_count == 2
        assert model.changepoint_prior_scale == 0.2
