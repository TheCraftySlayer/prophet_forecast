import pandas as pd
from prophet_analysis import write_summary


def test_write_summary_creates_file(tmp_path):
    df = pd.DataFrame({'metric': ['MAE', 'RMSE'], 'value': [1.0, float('nan')]})
    path = tmp_path / 'summary.csv'
    write_summary(df, path)
    text = path.read_text()
    assert 'NaN' in text
    assert text.startswith('metric,value')
