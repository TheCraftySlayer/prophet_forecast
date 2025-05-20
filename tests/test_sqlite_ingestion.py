import sqlite3
import pandas as pd
from pathlib import Path
from prophet_analysis import load_time_series, load_time_series_sqlite


def test_load_time_series_sqlite(tmp_path):
    df = pd.read_csv('calls.csv')
    df.columns = ['date', 'call_count']
    db_path = tmp_path / 'data.db'
    conn = sqlite3.connect(db_path)
    df.to_sql('calls', conn, index=False)
    conn.close()

    series_db = load_time_series_sqlite(db_path, 'calls', value_col='call_count')
    series_csv = load_time_series(Path('calls.csv'), metric='call')
    assert series_db.equals(series_csv)

    path = Path(str(db_path) + ':calls')
    series_auto = load_time_series(path, metric='call')
    assert series_auto.equals(series_csv)
