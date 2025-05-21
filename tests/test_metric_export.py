import pandas as pd
from prophet_analysis import write_summary


def test_write_summary_creates_file(tmp_path):
    df = pd.DataFrame({'metric': ['MAE', 'RMSE'], 'value': [1.0, float('nan')]})
    path = tmp_path / 'summary.csv'
    write_summary(df, path)
    text = path.read_text()
    assert 'NaN' in text
    assert text.startswith('metric,value')


def test_write_summary_preserves_nan_columns(tmp_path):
    df = pd.DataFrame({
        'horizon_days': [1, 7],
        'MAE': [0.5, 0.7],
        'RMSE': [0.6, 0.8],
        'MAPE': [float('nan'), float('nan')],
    })
    path = tmp_path / 'metrics.csv'
    write_summary(df, path)
    text = path.read_text()
    assert text.splitlines()[0] == 'horizon_days,MAE,RMSE,MAPE'
    assert text.strip().endswith('NaN')


def test_write_summary_returns_path(tmp_path):
    df = pd.DataFrame({'metric': ['MAE'], 'value': [1.0]})
    path = tmp_path / 'out.csv'
    returned = write_summary(df, path)
    assert returned == path


def test_write_summary_computes_mape(tmp_path):
    df = pd.DataFrame({
        'actual': [0, 0],
        'predicted': [1, 2]
    })
    path = tmp_path / 'preds.csv'
    write_summary(df, path)
    text = path.read_text()
    assert 'MAPE' in text.splitlines()[0]
