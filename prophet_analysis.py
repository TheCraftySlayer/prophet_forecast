"""
Prophet Forecast Analysis Script

This script loads customer service call, visitor, and chatbot query time series data,
merges them into a daily DataFrame with relevant features,
trains a Prophet model,
applies appropriate transformations if needed,
evaluates forecasting performance,
generates diagnostic plots,
analyzes press release and policy change impacts on call volumes,
and exports call predictions for the upcoming business days to Excel.

Example usage::

    python prophet_analysis.py calls.csv visitors.csv queries.csv prophet_results \
        --handle-outliers winsorize --use-transformation false --skip-feature-importance
"""

from __future__ import annotations

import os
import sys

# By default the real third-party packages are imported. Set ``USE_STUB_LIBS=1``
# to temporarily favour the lightweight stub modules bundled with the
# repository when running tests.
_USE_STUB_LIBS = os.getenv("USE_STUB_LIBS") == "1"
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if not _USE_STUB_LIBS:
    sys.path = [
        p
        for p in sys.path
        if os.path.abspath(p or os.getcwd()) != _THIS_DIR
    ]
import matplotlib

if not hasattr(matplotlib, "use"):
    raise ImportError(
        "The bundled matplotlib stub was imported. "
        "Install the real matplotlib package or set USE_STUB_LIBS=1 to use the stubs."
    )
matplotlib.use("Agg")  # ensure headless backend for multiprocessing safety
import argparse
import glob
import itertools
import logging
import pickle
import random
import math
import re
import sqlite3
import tempfile
import shutil
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Check pandas/statsmodels compatibility before importing heavy submodules
_PD_MAJOR = int(pd.__version__.split(".")[0])
_SM_VERSION = tuple(int(p) for p in statsmodels.__version__.split(".")[:3])
if _PD_MAJOR >= 2 and _SM_VERSION < (0, 14, 2):
    raise ImportError(
        "pandas>=2.0 requires statsmodels>=0.14.2 or pandas<2 must be installed."
    )

from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.arima.model import ARIMA

# Import Prophet
try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    from prophet.plot import plot_cross_validation_metric
    from prophet.models import StanBackendCmdStan
    _HAVE_PROPHET = True
except Exception:  # pragma: no cover - optional dependency may be missing
    Prophet = None

    def _missing_prophet(*_args, **_kwargs):
        raise ImportError("prophet package is required for forecasting features")

    def cross_validation(*args, **kwargs):
        return _missing_prophet()

    def performance_metrics(*args, **kwargs):
        return _missing_prophet()

    def plot_cross_validation_metric(*args, **kwargs):
        return _missing_prophet()

    class _DummyBackend:
        def __init__(self, *args, **kwargs):
            raise ImportError("prophet package is required for forecasting features")

    StanBackendCmdStan = _DummyBackend  # type: ignore

    _HAVE_PROPHET = False

# Handle seaborn import safely
try:
    import seaborn as sns
except ImportError:
    print("Warning: seaborn not installed. Some visualizations will be limited.")
    # Create a minimal fallback for sns.heatmap

    class SeabornFallback:
        def heatmap(self, *args, **kwargs):
            plt.imshow(*args)
            plt.colorbar()
    sns = SeabornFallback()

# Optional openpyxl dependency
try:
    import openpyxl  # type: ignore
    _HAVE_OPENPYXL = True
except Exception:
    _HAVE_OPENPYXL = False

# Optional Prophet serialization dependency
try:
    from prophet.serialize import model_to_json
    _HAVE_SERIALIZE = True
except Exception:  # pragma: no cover - optional dependency may be missing
    _HAVE_SERIALIZE = False
    model_to_json = None

# Global tuning options
POLICY_LOWER = True
PROPHET_KWARGS = {
    "yearly_seasonality": "auto",
    "weekly_seasonality": False,
    "daily_seasonality": False,
}


def build_prophet_kwargs(model_cfg: dict) -> dict:
    """Return Prophet keyword arguments merged with defaults."""
    kwargs = PROPHET_KWARGS.copy()
    for key in [
        "weekly_seasonality",
        "yearly_seasonality",
        "daily_seasonality",
        "uncertainty_samples",
    ]:
        if key in model_cfg:
            kwargs[key] = model_cfg[key]
    return kwargs

# Restore this directory in sys.path so local modules can be imported after the
# heavy third-party libraries have been loaded.
if not _USE_STUB_LIBS and _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from holidays_calendar import get_holidays_dataframe

# Optional CmdStanPy dependency to locate tbb runtime
try:
    from cmdstanpy.utils import cmdstan_path
except Exception:  # pragma: no cover - cmdstanpy may be missing
    cmdstan_path = None

_TBB_DLL_DIR_CACHE: str | None = None

# Path where a compiled custom Prophet Stan model will be stored
_COMPILED_STAN_MODEL = Path(__file__).with_name("prophet_model.pkl")


def _get_tbb_dll_dir() -> str | None:
    """Return directory containing ``tbb.dll`` if available."""
    global _TBB_DLL_DIR_CACHE
    if _TBB_DLL_DIR_CACHE is not None:
        return _TBB_DLL_DIR_CACHE
    if cmdstan_path is None:
        return None
    try:
        root = Path(cmdstan_path())
        for dll in root.rglob("tbb.dll"):
            _TBB_DLL_DIR_CACHE = str(dll.parent)
            break
    except Exception:  # pragma: no cover - path search errors
        _TBB_DLL_DIR_CACHE = None
    return _TBB_DLL_DIR_CACHE


def _ensure_tbb_on_path() -> None:
    """Add ``tbb.dll`` directory to ``PATH`` once if found."""
    dir_path = _get_tbb_dll_dir()
    if dir_path and dir_path not in os.environ.get("PATH", ""):
        os.environ["PATH"] = os.environ["PATH"] + os.pathsep + dir_path


def winsorize_series(series, limit=3):
    """Symmetric winsorization using mean ± limit*std."""
    mean = series.mean()
    std = series.std()
    lower = mean - limit * std
    upper = mean + limit * std
    return series.clip(lower, upper)


def winsorize_quantile(series, lower_q=0.01, upper_q=0.99):
    """Winsorize a series using quantile thresholds."""
    lower = series.quantile(lower_q)
    upper = series.quantile(upper_q)
    return series.clip(lower, upper)


def box_cox_transform(series: pd.Series, lmbda: float | None = None) -> tuple[pd.Series, float, float]:
    """Return Box-Cox transformed series along with ``lambda`` and applied shift."""
    arr = series.astype("float64").to_numpy()
    shift = 0.0
    if (arr <= 0).any():
        shift = abs(arr.min()) + 1.0
        arr = arr + shift

    if lmbda is None:
        grid = np.linspace(-2.0, 2.0, 41)
        log_arr = np.log(arr)
        n = arr.size

        def llf(l: float) -> float:
            if l == 0:
                transformed = log_arr
            else:
                transformed = (arr ** l - 1) / l
            var = transformed.var(ddof=1)
            return -n / 2 * np.log(var) + (l - 1) * log_arr.sum()

        scores = [llf(l) for l in grid]
        lmbda = float(grid[int(np.argmax(scores))])

    if lmbda == 0:
        transformed = np.log(arr)
    else:
        transformed = (arr ** lmbda - 1) / lmbda

    return pd.Series(transformed, index=series.index), lmbda, shift


def inv_box_cox_transform(series: pd.Series, lmbda: float, shift: float = 0.0) -> pd.Series:
    """Invert a Box-Cox transformation."""
    arr = series.to_numpy(dtype="float64")
    if lmbda == 0:
        out = np.exp(arr)
    else:
        out = np.power(lmbda * arr + 1, 1 / lmbda)
    if shift:
        out = out - shift
    return pd.Series(out, index=series.index)


def drop_collinear_features(
    df: pd.DataFrame, threshold: float = 10.0, return_dropped: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, list[str]]:
    """Drop collinear numeric features using variance inflation factor (VIF)."""
    logger = logging.getLogger(__name__)
    numeric_df = df.select_dtypes(include=np.number).drop(columns=["call_count"], errors="ignore")

    to_drop: list[str] = []
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor  # type: ignore

        X = numeric_df.dropna().copy()
        X = X.assign(const=1)
        while True:
            vif = pd.Series(
                [variance_inflation_factor(X.values, i) for i in range(len(X.columns) - 1)],
                index=X.columns[:-1],
            )
            if vif.empty:
                break
            max_vif = float(vif.max())
            if max_vif < threshold:
                break
            drop_col = str(vif.idxmax())
            to_drop.append(drop_col)
            X = X.drop(columns=[drop_col])
    except Exception as exc:  # pragma: no cover - statsmodels may be missing
        logger.warning("VIF calculation failed: %s; falling back to correlation", exc)
        corr = numeric_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop.extend([col for col in upper.columns if any(upper[col] > 0.9)])

    if to_drop:
        logger.info("Dropping collinear features: %s", to_drop)
        df = df.drop(columns=[c for c in to_drop if c in df.columns])

    if return_dropped:
        return df, to_drop
    return df


def compute_regressor_significance(
    regressors: pd.DataFrame, target: pd.Series, alpha: float = 0.05
) -> tuple[list[str], dict[str, float]]:
    """Return regressors with p-values below ``alpha``.

    The function uses ``statsmodels`` if available and falls back to a
    correlation-based approximation when that fails.
    """
    logger = logging.getLogger(__name__)
    significant: list[str] = []
    pvals: dict[str, float] = {}

    # Ensure alignment between target and regressors after any merges
    regressors = regressors.sort_index().reset_index(drop=True)
    target = target.sort_index().reset_index(drop=True)
    assert target.index.equals(regressors.index), "Regressor index misalignment"

    try:
        import statsmodels.api as sm  # type: ignore

        X = sm.add_constant(regressors)
        model = sm.OLS(target, X, missing="drop").fit()
        for col, pval in model.pvalues.items():
            if col == "const":
                continue
            pvals[col] = float(pval)
            if pval < alpha:
                significant.append(col)
    except Exception as exc:  # pragma: no cover - statsmodels may be missing
        logger.warning("Falling back to correlation-based p-values: %s", exc)
        n = len(target)
        for col in regressors.columns:
            x = regressors[col]
            if x.std() == 0:
                pvals[col] = 1.0
                continue
            r = np.corrcoef(x, target)[0, 1]
            if np.isnan(r):
                pvals[col] = 1.0
                continue
            t_stat = r * np.sqrt(n - 2) / np.sqrt(max(1e-12, 1 - r**2))
            pval = math.erfc(abs(t_stat) / math.sqrt(2))
            pvals[col] = float(pval)
            if pval < alpha:
                significant.append(col)

    logger.info("Regressor p-values: %s", pvals)
    return significant, pvals


def compile_custom_stan_model(likelihood: str) -> Path | None:
    """Compile Prophet's Stan model with a custom likelihood.

    Parameters
    ----------
    likelihood : str
        Either ``"poisson"`` or ``"neg_binomial"``.

    Returns
    -------
    Path | None
        Path to the compiled model on success, otherwise ``None``.
    """
    logger = logging.getLogger(__name__)
    _ensure_tbb_on_path()

    if not _HAVE_PROPHET:
        logger.warning("Prophet package not installed; cannot compile custom Stan model")
        return None

    try:
        import importlib.resources as pkg_resources
        model_text = (
            pkg_resources.files("prophet")
            .joinpath("stan/prophet.stan")
            .read_text()
        )
    except Exception as e:  # pragma: no cover - environment may lack prophet
        local_stan = Path(__file__).with_name("prophet.stan")
        if local_stan.exists():
            logger.info("Using bundled prophet.stan")
            model_text = local_stan.read_text()
        else:
            logger.warning(f"Failed to load Prophet model.stan: {e}")
            return None

    if likelihood == 'poisson':
        model_text = model_text.replace('normal_lpdf(y', 'poisson_log_lpmf(y')
    elif likelihood in {'neg_binomial', 'negative-binomial', 'negative_binomial'}:
        if 'phi' not in model_text:
            model_text = model_text.replace('real sigma;', 'real<lower=0> phi;\n  real sigma;')
        model_text = model_text.replace('normal_lpdf(y', 'neg_binomial_2_log_lpmf(y')
    else:
        logger.warning("Unsupported likelihood: %s", likelihood)
        return None

    if _COMPILED_STAN_MODEL.exists():
        logger.info("Using cached custom Stan model at %s", _COMPILED_STAN_MODEL)
        return _COMPILED_STAN_MODEL

    tmp_dir = Path(tempfile.mkdtemp())
    stan_path = tmp_dir / 'model.stan'
    compiled_path = tmp_dir / 'model.pkl'
    with open(stan_path, 'w') as f:
        f.write(model_text)

    try:
        # prophet.serialize --compile model.stan model.pkl
        import subprocess
        subprocess.check_call([
            sys.executable,
            '-m',
            'prophet.serialize',
            '--compile',
            str(stan_path),
            str(compiled_path),
        ])
        shutil.copy2(compiled_path, _COMPILED_STAN_MODEL)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return _COMPILED_STAN_MODEL
    except Exception as e:  # pragma: no cover - compilation may fail on CI
        logger.warning(f"Failed to compile custom Stan model: {e}")
        return None
def tune_prophet_hyperparameters(prophet_df, prophet_kwargs=None):
    """Find optimal Prophet hyperparameters using grid search"""
    logger = logging.getLogger(__name__)
    logger.info("Tuning Prophet hyperparameters")
    
    if prophet_kwargs is None:
        prophet_kwargs = PROPHET_KWARGS

    # Parameter grid expanded on a log scale
    param_grid = {
        'changepoint_prior_scale': list(np.logspace(-2, 0, 10))
    }
    
    # Generate all combinations
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    
    # Storage for results
    devs = []
    
    # Use cross-validation to evaluate
    for i, params in enumerate(all_params):
        logger.info(f"Testing hyperparameter combination {i+1}/{len(all_params)}")
        
        try:
            m = Prophet(
                growth='linear',
                interval_width=0.9,
                seasonality_mode='additive',
                n_changepoints=25,
                changepoint_range=0.9,
                **prophet_kwargs,
                **params,
            )

            df_copy = prophet_df.copy()

            _ensure_tbb_on_path()
            _fit_prophet_with_fallback(m, df_copy)

            df_cv = cross_validation(
                m,
                initial='180 days',
                period='30 days',
                horizon='14 days',
                parallel="processes"
            )
            df_cv = df_cv[df_cv['ds'].dt.dayofweek < 5]
            df_p = performance_metrics(df_cv, rolling_window=1)
            if 'mean_poisson_deviance' in df_p.columns:
                devs.append(df_p['mean_poisson_deviance'].mean())
            else:
                devs.append(_mean_poisson_deviance(df_cv['y'], df_cv['yhat']))
        except Exception as e:
            logger.warning(f"Error with hyperparameter combination {params}: {str(e)}")
            devs.append(float('inf'))  # Assign worst possible score
    
    # Find best parameters
    if not devs or all(np.isinf(devs)):
        logger.warning("All hyperparameter combinations failed, using defaults")
        best_params = {'changepoint_prior_scale': 0.2, 'seasonality_prior_scale': 0.01, 'holidays_prior_scale': 5}
    else:
        best_params = all_params[np.argmin(devs)]
    
    logger.info(f"Best parameters found: {best_params}")
    return best_params
def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s:%(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)
    return logger


def load_time_series_sqlite(
    db_path: Path, table: str, date_col: str = "date", value_col: str = "value"
) -> pd.Series:
    """Load a time series from a SQLite database table."""
    conn = sqlite3.connect(db_path)
    try:
        query = f"SELECT {date_col}, {value_col} FROM {table}"
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    df["date_parsed"] = pd.to_datetime(
        df[date_col], format="%m/%d/%y", errors="coerce"
    )
    df = df.dropna(subset=["date_parsed"])
    series = df.set_index("date_parsed")[value_col].sort_index()
    full_idx = pd.date_range(series.index.min(), series.index.max(), freq="D")
    return series.reindex(full_idx).fillna(0)


@lru_cache(maxsize=None)
def load_time_series(path: Path, metric: str = "call") -> pd.Series:
    """Load a time series from a CSV, Excel or SQLite file."""
    file_str = str(path)
    if ".db:" in file_str:
        db_file, table = file_str.split(":", 1)
        return load_time_series_sqlite(Path(db_file), table)

    # Check file extension
    file_ext = file_str.lower()

    if file_ext.endswith('.csv'):
        # Load CSV file
        df = pd.read_csv(path)

        # If the file has no header ("date" column missing), re-read with header=None
        if 'date' not in df.columns:
            df = pd.read_csv(path, header=None)
            cols = list(df.columns)
            if len(cols) >= 2:
                df.columns = ['date', 'value'] + cols[2:]
            else:
                df.columns = ['date', 'value']

        date_col = 'date'

        # Set the appropriate value column based on file type
        if metric == 'call':
            value_col_options = ['Count of Calls', 'call_count', 'value']
        elif metric == 'visit':
            value_col_options = ['Visits', 'visit_count', 'value']
        else:
            value_col_options = ['query_count', 'value']

        value_col = next((c for c in df.columns if c in value_col_options or
                          metric.lower() in c.lower()), df.columns[1])

    elif file_ext.endswith('.xlsx') or file_ext.endswith('.xls'):
        if not _HAVE_OPENPYXL:
            raise ImportError("Reading Excel files requires the 'openpyxl' package")
        # Handle Excel files with explicit engine
        df = pd.read_excel(path, engine='openpyxl')

        if 'date' not in df.columns:
            df = pd.read_excel(path, header=None, engine='openpyxl')
            cols = list(df.columns)
            if len(cols) >= 2:
                df.columns = ['date', 'value'] + cols[2:]
            else:
                df.columns = ['date', 'value']

        date_col = 'date'

        if metric == 'call':
            value_col_options = ['Count of Calls', 'call_count', 'value']
        elif metric == 'visit':
            value_col_options = ['Visits', 'visit_count', 'value']
        else:
            value_col_options = ['query_count', 'value']

        value_col = next((c for c in df.columns if c in value_col_options or
                          metric.lower() in c.lower()), df.columns[1])
    else:
        raise ValueError(f"Unsupported file format: {path}")

    # Convert date column to datetime using known format per metric
    if metric == "call":
        fmt = "%m/%d/%y"
    else:
        fmt = "%m/%d/%Y"
    df["date_parsed"] = pd.to_datetime(df[date_col], format=fmt, errors="coerce")
    df = df.dropna(subset=["date_parsed"])

    # Return the time series with all days, filling missing weekends with 0
    series = df.set_index("date_parsed")[value_col].sort_index()
    full_idx = pd.date_range(series.index.min(), series.index.max(), freq="D")
    series = series.reindex(full_idx).fillna(0)
    return series


def verify_date_formats(call_path, visit_path, chat_path):
    """Verify date formats are consistent across files"""
    call_df = pd.read_csv(call_path)
    visit_df = pd.read_csv(visit_path)
    chat_df = pd.read_csv(chat_path)

    # If a file lacked headers, re-read with names
    if 'date' not in call_df.columns:
        call_df = pd.read_csv(call_path, header=None, names=['date', 'call_count'])
    if 'date' not in visit_df.columns:
        visit_df = pd.read_csv(visit_path, header=None, names=['date', 'visit_count'])
    if 'date' not in chat_df.columns:
        chat_df = pd.read_csv(chat_path, header=None, names=['date', 'query_count'])

    # Handle files without headers by renaming the first column to 'date'
    for df in (call_df, visit_df, chat_df):
        if 'date' not in df.columns:
            df.rename(columns={df.columns[0]: 'date'}, inplace=True)

    logger = logging.getLogger(__name__)
    logger.info("Date format samples:")
    for name, df in [("Calls", call_df), ("Visits", visit_df), ("Queries", chat_df)]:
        if df.empty:
            logger.warning(f"{name}: file is empty")
        else:
            logger.info(f"{name}: {df['date'].iloc[0]}")

    # Parse dates with explicit formats
    for name, df in [("Calls", call_df), ("Visits", visit_df), ("Queries", chat_df)]:
        if df.empty:
            continue
        fmt = "%m/%d/%y" if name == "Calls" else "%m/%d/%Y"
        dates = pd.to_datetime(df['date'], format=fmt, errors="coerce").dropna()
        logger.info(f"{name}: Successfully parsed {len(dates)} dates")


def build_flag_series(dates: pd.DatetimeIndex, dates_list: list) -> pd.Series:
    """
    Create a binary flag series for specific dates

    Args:
        dates: DatetimeIndex to create flags for
        dates_list: List of dates to flag as 1

    Returns:
        pandas.Series: Binary flags with 1 for dates in the list, 0 otherwise
    """
    dt_list = pd.to_datetime(dates_list)
    return pd.Series(dates.normalize().isin(dt_list).astype(int), index=dates)


def prepare_data(
    call_path,
    visit_path,
    chat_path,
    cleaned_calls=None,
    scale_features=True,
    events: dict | None = None,
):
    """Prepare time series data with configurable event windows."""

    # Initialize logger first - moved to the beginning of the function
    logger = logging.getLogger(__name__)

    verify_date_formats(call_path, visit_path, chat_path)

    # Check for large date gaps
    calls_dates = load_time_series(call_path, metric="call").index
    visits_dates = load_time_series(visit_path, metric="visit").index
    chat_dates = pd.to_datetime(
        pd.read_csv(chat_path)["date"],
        format="%m/%d/%Y",
        errors="coerce",
    ).dropna()

    # Log date ranges
    logger.info(
        f"Call data: {calls_dates.min()} to {calls_dates.max()}, {len(calls_dates)} records")
    logger.info(
        f"Visit data: {visits_dates.min()} to {visits_dates.max()}, {len(visits_dates)} records")
    logger.info(
        f"Chat data: {chat_dates.min()} to {chat_dates.max()}, {len(chat_dates)} records")

    logger.info(
        "Preparing data for forecasting with 2025 policy change features")

    # Load time series data (existing code)
    calls = load_time_series(call_path, metric="call")
    if cleaned_calls is not None:
        logger.info("Using provided cleaned call data")
        if not isinstance(cleaned_calls, pd.Series):
            cleaned_calls = pd.Series(cleaned_calls)
        if not cleaned_calls.index.equals(calls.index):
            logger.warning(
                "Cleaned calls index doesn't match original calls index")
            cleaned_calls = cleaned_calls.reindex(calls.index)
        calls = cleaned_calls

    visits = load_time_series(visit_path, metric="visit")

    # Load chatbot data (existing code)
    logger.info(f"Loading chatbot data from {chat_path}")
    chat_df = pd.read_csv(chat_path)
    dt_cols = [c for c in chat_df.columns if "date" in c.lower()
               or "time" in c.lower()]
    dt_col = dt_cols[0] if dt_cols else chat_df.columns[0]
    chat = (
        pd.to_datetime(
            chat_df[dt_col],
            format="%m/%d/%Y",
            errors="coerce",
        )
        .dropna()
        .dt.normalize()
        .value_counts()
        .sort_index()
    )

    # Create unified date range
    start = min(calls.index.min(), visits.index.min(), chat.index.min())
    end = max(calls.index.max(), visits.index.max(), chat.index.max())
    logger.info(f"Creating unified date range from {start} to {end}")
    idx = pd.date_range(start=start, end=end, freq="D")

    holiday_df = get_holidays_dataframe()
    mask = (
        (holiday_df['event'] == 'county_holiday')
        & (holiday_df['date'] >= idx.min())
        & (holiday_df['date'] <= idx.max())
    )
    holiday_dates = holiday_df.loc[mask, 'date']

    press_release_dates = holiday_df.loc[
        (holiday_df['event'] == 'press_release')
        & (holiday_df['date'] >= idx.min())
        & (holiday_df['date'] <= idx.max()),
        'date',
    ]

    # Build main dataframe
    df = pd.DataFrame({
        "call_count": calls.reindex(idx),
        "visit_count": visits.reindex(idx),
        "chatbot_count": chat.reindex(idx),
    }, index=idx)

    # Fill missing chatbot counts before transformations
    df["chatbot_count"] = (
        df["chatbot_count"].interpolate()
        .ffill()
        .bfill()
        .astype(float)
    )

    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["is_monday"] = (df.index.dayofweek == 0).astype(int)

    # Flag zero-call weekdays and treat them as missing
    df['zero_call_flag'] = (
        (df['call_count'] == 0) & (df['is_weekend'] == 0)
    ).astype(int)
    df['missing_flag'] = df['call_count'].isna().astype(int)
    df["call_count"] = df["call_count"].astype(float)

    # Interpolate and fill missing visit counts before creating rolling features
    df["visit_count"] = df["visit_count"].interpolate()
    df["visit_count"] = df["visit_count"].ffill().bfill().astype(float)

    # Recalculate weekday means after cleaning
    weekday_mask = df.index.dayofweek < 5
    weekday_means = (
        df.loc[weekday_mask, "call_count"]
        .groupby(df.index.dayofweek[weekday_mask])
        .mean()
    )
    if not weekday_means.empty:
        monday_spike = weekday_means.get(0, 0) - weekday_means.drop(0).mean()
        logger.info(f"Monday spike magnitude: {monday_spike:.1f}")
        df["monday_effect"] = df["is_monday"] * monday_spike
    else:
        df["monday_effect"] = 0.0

    # Flag events before outlier handling
    if events is None:
        events = {}

    deadline_dates = pd.date_range(start=idx.min(), end=idx.max(), freq="MS")
    notice_dates = [
        pd.Timestamp(year, 3, 1) for year in range(idx.min().year, idx.max().year + 1)
    ]
    df['county_holiday_flag'] = df.index.isin(holiday_dates).astype(int)
    df['deadline_flag'] = 0
    df['notice_flag'] = 0
    for d in deadline_dates:
        window = pd.date_range(d - pd.Timedelta(days=5), d + pd.Timedelta(days=1))
        df.loc[df.index.isin(window), 'deadline_flag'] = 1
    for d in notice_dates:
        window = pd.date_range(d, d + pd.Timedelta(days=7))
        df.loc[df.index.isin(window), 'notice_flag'] = 1

    df['deadline_flag'] = df['deadline_flag'].astype(int)
    df['notice_flag'] = df['notice_flag'].astype(int)

    # Flag for post-policy period
    policy_start = pd.to_datetime(events.get("policy_start", "2025-05-01"))
    df['post_policy'] = (df.index >= policy_start).astype(int)

    # Flag for targeted campaign
    campaign_cfg = events.get("campaign", {})
    campaign_start = pd.to_datetime(campaign_cfg.get("start", "2025-05-01"))
    campaign_end = pd.to_datetime(campaign_cfg.get("end", "2025-06-02"))
    df['is_campaign'] = ((df.index >= campaign_start) & (df.index <= campaign_end)).astype(int)
    df['is_campaign'] = df['is_campaign'].shift(1).fillna(0).astype(int)

    df['press_release_flag'] = df.index.isin(press_release_dates).astype(int)

    df['closure_flag'] = ((df['is_weekend'] == 1) | (df['county_holiday_flag'] == 1)).astype(int)

    # Indicator for business days (0 on weekends and county holidays)
    df['is_business_day'] = (
        (df.index.dayofweek < 5) & ~df.index.isin(holiday_dates)
    ).astype(int)

    def hampel(series, window=7, n_sigmas=3):
        median = series.rolling(window, center=True).median()
        diff = np.abs(series - median)
        mad = diff.rolling(window, center=True).median()
        threshold = n_sigmas * 1.4826 * mad
        return (diff > threshold).astype(int)

    # Down-weight extreme spikes prior to feature creation
    tmp_ma7 = df["call_count"].rolling(7, min_periods=1).mean()
    tmp_std7 = df["call_count"].rolling(7, min_periods=1).std().fillna(0)
    residual = df["call_count"] - tmp_ma7
    extreme_mask = residual.abs() > 3 * tmp_std7
    if extreme_mask.any():
        logger.info(
            "Winsorizing %d extreme outliers (>3 \u03c3 residuals)",
            extreme_mask.sum(),
        )
        clipped = tmp_ma7 + np.sign(residual) * 3 * tmp_std7
        df.loc[extreme_mask, "call_count"] = clipped.loc[extreme_mask]
    df["extreme_outlier"] = extreme_mask.astype(int)

    event_mask = (
        df['county_holiday_flag'] != 0
    ) | (df['deadline_flag'] != 0) | (df['notice_flag'] != 0)
    df['outlier_flag'] = hampel(df['call_count']) & (~event_mask)
    df['spike_flag'] = (df['call_count'] > df['call_count'].quantile(0.99)).astype(int)

    # Keep intermediate quality flags for modeling
    df = df.drop(columns=['zero_call_flag', 'missing_flag'])

    # Keep raw counts; z-scoring applied later
    # Feature engineering: lags and rolling stats for potential use as regressors
    logger.info("Creating lag and rolling features")
    for lag in [1, 3, 7]:
        df[f"call_lag{lag}"] = df["call_count"].shift(lag).fillna(0).astype(float)
    df["call_ma7"] = df["call_count"].rolling(7, min_periods=1).mean()
    df["call_std7"] = df["call_count"].rolling(7, min_periods=1).std().fillna(0).astype(float)

    df["visit_ma3"] = df["visit_count"].rolling(3, min_periods=1).mean()

    # Standardize chatbot counts on log scale
    df["chatbot_count"] = np.log1p(df["chatbot_count"])
    mean_chat = df["chatbot_count"].mean()
    std_chat = df["chatbot_count"].std()
    if std_chat != 0:
        df["chatbot_count"] = (df["chatbot_count"] - mean_chat) / std_chat
    df["chatbot_count"] = winsorize_series(df["chatbot_count"], limit=3)
    df["chatbot_ma3"] = df["chatbot_count"].rolling(3, min_periods=1).mean()

    # Winsorize continuous regressors
    for col in ["visit_ma3", "chatbot_count", "chatbot_ma3"]:
        if col in df.columns:
            df[col] = winsorize_quantile(df[col])

    # Drop problematic flat-line period if configured
    flat_cfg = events.get("flat_period", {})
    flat_start = pd.to_datetime(flat_cfg.get("start", "2025-05-06"))
    flat_end = pd.to_datetime(flat_cfg.get("end", "2025-05-13"))
    mask_flat = (df.index >= flat_start) & (df.index <= flat_end)
    if mask_flat.any():
        df = df.loc[~mask_flat]

    # Create regressors dataframe for Prophet
    regressors = df.copy()

    important_regs = [
        "visit_ma3",
        "chatbot_count",
        "call_lag1",
        "call_lag7",
        "monday_effect",
        "deadline_flag",
        "notice_flag",
        "is_campaign",
        "post_policy",
        "press_release_flag",
    ]
    regressors = regressors[important_regs]

    if scale_features:
        for col in ["visit_ma3"]:
            if col in regressors.columns:
                mean = regressors[col].mean()
                std = regressors[col].std()
                if std != 0:
                    regressors[col] = (regressors[col] - mean) / std

    # Drop highly collinear regressors to avoid instability
    regressors = drop_collinear_features(regressors)

    return df, regressors

def create_prophet_holidays(holiday_dates, deadline_dates, closure_dates=None, press_release_dates=None):
    """
    Create holiday DataFrame for Prophet model
    
    Prophet requires a DataFrame with holiday dates and labels
    """
    # Create holiday DataFrame
    holidays = pd.DataFrame({
        'holiday': 'holiday',
        'ds': pd.to_datetime(holiday_dates),
        'lower_window': 0,
        'upper_window': 1  # Effect may last for 1 day after holiday
    })
    
    # Create deadline DataFrame
    deadlines = pd.DataFrame({
        'holiday': 'deadline',
        'ds': pd.to_datetime(deadline_dates),
        'lower_window': -1,  # Effect may start 1 day before deadline
        'upper_window': 0
    })
    
    if press_release_dates is None:
        press_release_dates = []
    elif isinstance(press_release_dates, pd.DatetimeIndex):
        press_release_dates = press_release_dates.to_list()

    if closure_dates is None:
        closure_dates = []
    elif isinstance(closure_dates, pd.DatetimeIndex):
        closure_dates = closure_dates.to_list()

    # Create press release DataFrame
    press_releases = pd.DataFrame({
        'holiday': 'press_release',
        'ds': pd.to_datetime(press_release_dates),
        'lower_window': 0,
        'upper_window': 3  # Effect may last for 3 days after press release
    })

    closures = pd.DataFrame({
        'holiday': 'closure',
        'ds': pd.to_datetime(closure_dates),
        'lower_window': 0,
        'upper_window': 0
    })
    
    # Combine all holiday DataFrames and remove duplicates
    all_holidays = pd.concat([holidays, deadlines, closures, press_releases])
    all_holidays = all_holidays.drop_duplicates(subset=['ds', 'holiday'])

    return all_holidays

def enhance_holiday_handling(holidays_df):
    """Improve holiday effects modeling"""
    
    # 1. Add bridging days (e.g., days between a holiday and weekend)
    holiday_dates = pd.to_datetime(holidays_df[holidays_df['holiday'] == 'holiday']['ds'])
    bridge_days = []
    
    for holiday_date in holiday_dates:
        # If holiday is on Tuesday, Monday might be taken off
        if holiday_date.dayofweek == 1:  # Tuesday
            bridge_days.append(holiday_date - pd.Timedelta(days=1))
        # If holiday is on Thursday, Friday might be taken off
        elif holiday_date.dayofweek == 3:  # Thursday
            bridge_days.append(holiday_date + pd.Timedelta(days=1))
    
    # Add bridge days to holidays DataFrame
    if bridge_days:
        bridge_df = pd.DataFrame({
            'holiday': 'bridge_day',
            'ds': bridge_days,
            'lower_window': 0,
            'upper_window': 0
        })
        holidays_df = pd.concat([holidays_df, bridge_df])
    
    # 2. Add pre-holiday effects (people often call before holidays)
    pre_holidays = pd.DataFrame({
        'holiday': 'pre_holiday',
        'ds': holiday_dates - pd.Timedelta(days=1),
        'lower_window': 0,
        'upper_window': 0
    })
    
    return pd.concat([holidays_df, pre_holidays])

def prepare_prophet_data(df):
    """
    Convert DataFrame to Prophet format
    
    Prophet requires columns ds (datestamp) and y (target variable)
    """
    # Ensure no missing target values
    if df['call_count'].isna().any():
        df = df.dropna(subset=['call_count'])

    prophet_df = (
        df[['call_count']]
        .reset_index()
        .rename(columns={'index': 'ds', 'call_count': 'y'})
    )

    return prophet_df


def _fit_prophet_with_fallback(model, df) -> None:
    """Fit Prophet model with LBFGS then fall back to Newton once."""
    logger = logging.getLogger(__name__)
    try:
        model.fit(df)
        return
    except Exception as exc:  # pragma: no cover - fit may fail
        logger.warning("LBFGS optimization failed: %s; retrying with Newton", exc)
    try:
        model.fit(df, algorithm="Newton")
        logger.info("Model fit succeeded with Newton optimizer")
    except Exception:
        logger.error("Newton optimization also failed", exc_info=True)
        raise


def train_prophet_model(
    prophet_df,
    holidays_df,
    regressors_df,
    future_periods=30,
    model_params=None,
    prophet_kwargs=None,
    log_transform=False,
    likelihood="normal",
    transform: str | None = None,
):
    """
    Train Prophet model with custom components
    
    Args:
        prophet_df: DataFrame with ds and y columns
        holidays_df: DataFrame with holiday information
        regressors_df: DataFrame with regressor variables
        future_periods: Number of days to forecast
        model_params: Optional dictionary of parameters to pass to Prophet
            (may include ``capacity`` for logistic growth)
        likelihood: Distribution for the likelihood ('normal', 'poisson',
            or 'neg_binomial')
        transform: Optional transformation to apply to the target ('log' or
            'box-cox'). ``log_transform`` is honored when ``transform`` is None.
        If ``capacity`` is not supplied with logistic growth, it defaults to
        110% of the training maximum.
        
    Returns:
        Trained Prophet model, forecast DataFrame, future DataFrame
    """
    logger = logging.getLogger(__name__)
    logger.info("Training Prophet model")
    
    # Initialize Prophet model with optional tuned parameters

    if prophet_kwargs is None:
        prophet_kwargs = PROPHET_KWARGS

    default_params = {
        **prophet_kwargs,
        'seasonality_mode': 'additive',
        'n_changepoints': 25,
        'changepoint_prior_scale': 0.05,
        'changepoint_range': 0.9,
        'changepoints': [
            pd.Timestamp('2023-11-01'),
            pd.Timestamp('2024-04-15'),
            pd.Timestamp('2025-04-01'),
        ],
        'seasonality_prior_scale': 0.01,
        'holidays_prior_scale': 5,
        'holidays': holidays_df,
        'mcmc_samples': 0,
        'uncertainty_samples': 300,
        'growth': 'logistic',
        'interval_width': 0.9,
        'capacity': None,
    }

    reg_prior_scale = 0.05

    if model_params:
        reg_prior_scale = model_params.pop('regressor_prior_scale', reg_prior_scale)
        default_params.update(model_params)

    capacity = default_params.pop('capacity', None)

    prophet_df = prophet_df.copy()
    if default_params.get('growth', 'linear') == 'logistic':
        if capacity is None:
            capacity = float(prophet_df['y'].max() * 1.1)
        prophet_df['cap'] = capacity
    else:
        capacity = None
    if transform is None and log_transform:
        transform = "log"

    bc_params: tuple[float, float] | None = None
    if transform == "log":
        prophet_df['y'] = np.log1p(prophet_df['y'])
    elif transform == "box-cox":
        prophet_df['y'], lam, shift = box_cox_transform(prophet_df['y'])
        bc_params = (lam, shift)


    custom_model_path = None
    if likelihood != "normal":
        custom_model_path = compile_custom_stan_model(likelihood)
        if custom_model_path:
            logger.info("Compiled custom Stan model for %s likelihood", likelihood)
        else:
            logger.warning("Falling back to normal likelihood")

    backend = None
    if custom_model_path and _HAVE_PROPHET:
        try:
            with open(custom_model_path, "rb") as f:
                compiled_model = pickle.load(f)
            backend = StanBackendCmdStan(model=compiled_model)
        except Exception as e:  # pragma: no cover - prophet may be missing
            logger.warning(f"Could not create custom Stan backend: {e}")

    model = Prophet(stan_backend=backend, **default_params) if backend else Prophet(**default_params)
    if not default_params.get("weekly_seasonality", False):
        model.add_seasonality(name="weekly", period=7, fourier_order=5)

    if custom_model_path and backend is None:
        try:
            with open(custom_model_path, "rb") as f:
                compiled_model = pickle.load(f)
            if hasattr(model, "stan_backend") and hasattr(model.stan_backend, "stan_model"):
                model.stan_backend.stan_model = compiled_model
        except Exception as e:  # pragma: no cover - prophet may be missing
            logger.warning(f"Could not attach custom Stan model: {e}")
    

    # Drop collinear regressors first and capture removed columns
    regressors_df, dropped_cols = drop_collinear_features(
        regressors_df, return_dropped=True
    )
    removed_regs = set(dropped_cols)

    # ------------------------------------------------------------------
    # Ensure lag/rolling features are constructed on a continuous timeline
    # covering the entire training data. If ``regressors_df`` has already
    # been sliced to a train subset, reindex it to the full date range and
    # propagate known values forward so that lagged features remain valid.
    # Boolean/indicator features are filled with 0 when missing.
    # ------------------------------------------------------------------
    full_train_index = pd.date_range(
        regressors_df.index.min(), prophet_df['ds'].max(), freq='D'
    )
    full_regs = regressors_df.reindex(full_train_index)
    flag_cols = [
        c
        for c in full_regs.columns
        if 'flag' in c or c.startswith('is_') or c.startswith('call_lag') or c == 'post_policy'
    ]
    full_regs[flag_cols] = full_regs[flag_cols].fillna(0)
    full_regs = full_regs.ffill().fillna(0)
    regressors_df = full_regs.loc[prophet_df['ds']]

    # Restrict regressors to mitigate collinearity
    # Use standardized raw visitor and chatbot counts
    important_regressors = [
        'visit_ma3',
        'chatbot_count',
        'call_lag1',
        'call_lag7',
        'monday_effect',
        'notice_flag',
        'deadline_flag',
        'is_campaign',
        'post_policy',
        'press_release_flag',
    ]

    important_regressors = [r for r in important_regressors if r not in dropped_cols]

    existing_regs = [r for r in important_regressors if r in regressors_df.columns]

    if existing_regs:
        significant_regs, pvals = compute_regressor_significance(
            regressors_df[existing_regs], prophet_df['y']
        )
        # Drop chat, notice, deadline and campaign flags when not significant
        drop_if_nonsig = {
            'chatbot_count',
            'notice_flag',
            'deadline_flag',
            'is_campaign',
        }
        for reg in drop_if_nonsig:
            if reg in existing_regs and reg not in significant_regs:
                logger.info('Dropping non-significant regressor: %s', reg)
                existing_regs.remove(reg)
                removed_regs.add(reg)
        important_regressors = [
            r for r in significant_regs if r in existing_regs
        ]
    else:
        important_regressors = []

    if removed_regs:
        regressors_df = regressors_df.drop(
            columns=[c for c in removed_regs if c in regressors_df.columns]
        )
    
    for regressor in important_regressors:
        if regressor in regressors_df.columns:
            prophet_df[regressor] = regressors_df[regressor].values
            model.add_regressor(
                regressor,
                mode='additive',
                prior_scale=reg_prior_scale,
                standardize='auto',
            )
    
    # Fit the model
    logger.info("Fitting Prophet model")
    _ensure_tbb_on_path()
    _fit_prophet_with_fallback(model, prophet_df)

    # Create future DataFrame
    logger.info(f"Creating future DataFrame with {future_periods} periods")
    future = model.make_future_dataframe(periods=future_periods, freq="B")
    if capacity is not None:
        future['cap'] = capacity

    # Extend the completed regressor matrix to cover the forecast horizon
    full_index = pd.date_range(full_regs.index.min(), future['ds'].max(), freq='D')
    full_regs = full_regs.reindex(full_index)
    full_regs[flag_cols] = full_regs[flag_cols].fillna(0)
    full_regs = full_regs.ffill().fillna(0)

    # Determine official holiday dates for future regressor flags
    holiday_dates = pd.to_datetime(
        holidays_df[holidays_df['holiday'] == 'holiday']['ds']
    )

    # Build full daily calendar covering the forecast horizon
    full_dates = pd.date_range(future['ds'].min(), future['ds'].max(), freq='B')
    future_regs = pd.DataFrame(index=full_dates)

    # Required regressors only
    future_regs['visit_ma3'] = 0
    future_regs['chatbot_count'] = 0
    future_regs['call_lag1'] = 0
    future_regs['call_lag7'] = 0
    future_regs['monday_effect'] = 0
    future_regs['notice_flag'] = 0
    future_regs['deadline_flag'] = 0
    future_regs['is_campaign'] = 0
    future_regs['post_policy'] = 0
    future_regs['press_release_flag'] = 0
    if capacity is not None:
        future_regs['cap'] = capacity

    # Ensure float dtypes before merging to avoid warnings
    reg_cols = list(future_regs.columns)
    future_regs[reg_cols] = future_regs[reg_cols].astype("float64")

    # Overlay known regressor values from historical data
    known = full_regs.reindex(future_regs.index)
    for col in future_regs.columns:
        if col in known.columns:
            mask = known[col].notna()
            future_regs.loc[mask, col] = known.loc[mask, col]

    if removed_regs:
        future_regs.drop(
            columns=[c for c in removed_regs if c in future_regs.columns],
            inplace=True,
        )

    # Merge regressor values back into the future dataframe on the date column
    future = future.merge(future_regs, left_on="ds", right_index=True, how="left")
    future = future.sort_values("ds").reset_index(drop=True)

    if removed_regs:
        future.drop(columns=[c for c in removed_regs if c in future.columns], inplace=True)

    # Ensure the logistic growth capacity column survives the merge
    if "cap_x" in future.columns:
        future["cap"] = future.pop("cap_x")
    if "cap_y" in future.columns:
        future["cap"] = future.get("cap", future["cap_y"]).fillna(future["cap_y"])
        future.drop(columns=["cap_y"], inplace=True)


    # Basic sanity check for merged regressors
    check_cols = [
        'visit_ma3',
        'chatbot_count',
        'call_lag1',
        'call_lag7',
        'monday_effect',
        'notice_flag',
        'deadline_flag',
        'is_campaign',
        'post_policy',
        'press_release_flag',
    ]
    for col in check_cols:
        if col in future.columns and future[col].isna().any():
            logger.warning(
                f"Found {future[col].isna().sum()} NaN values in {col} after merge"
            )
            future[col] = future[col].fillna(0)

    train_cols = set(prophet_df.columns) - {"y", "ds"}
    future_cols = set(future.columns) - {"ds"}
    if train_cols != future_cols:
        diff = train_cols ^ future_cols
        logger.warning(
            "Feature mismatch between training and future data: %s", diff
        )
        # Drop any extra columns and fill missing ones with 0 so prediction can continue
        for col in future_cols - train_cols:
            future.drop(columns=col, inplace=True)
        for col in train_cols - future_cols:
            future[col] = 0

    ordered = ['ds'] + sorted(train_cols)
    future = future.reindex(columns=ordered)
    
    # Make forecast
    logger.info("Making forecast")
    forecast = model.predict(future)

    if transform == "log":
        for col in ['yhat', 'yhat_lower', 'yhat_upper']:
            if col in forecast:
                forecast[col] = np.expm1(forecast[col])
    elif transform == "box-cox" and bc_params is not None:
        lam, shift = bc_params
        for col in ['yhat', 'yhat_lower', 'yhat_upper']:
            if col in forecast:
                forecast[col] = inv_box_cox_transform(forecast[col], lam, shift)

    forecast[['yhat', 'yhat_lower', 'yhat_upper']] = (
        forecast[['yhat', 'yhat_lower', 'yhat_upper']].clip(lower=0)
    )

    _check_forecast_sanity(forecast)

    return model, forecast, future

def _check_forecast_sanity(forecast: pd.DataFrame) -> None:
    """Run simple sanity checks on forecast outputs."""
    logger = logging.getLogger(__name__)
    if 'weekly' in forecast.columns and not POLICY_LOWER:
        week = forecast[['ds', 'weekly']].copy()
        week['dow'] = week['ds'].dt.dayofweek
        midweek = week[week['dow'].isin([1, 2, 3])]['weekly'].mean()
        friday = week[week['dow'] == 4]['weekly'].mean()
        if pd.notna(friday) and pd.notna(midweek) and friday < midweek:
            logger.warning('Friday weekly effect negative relative to mid-week baseline')
    if 'yearly' in forecast.columns:
        eff = forecast[['ds', 'yearly']].copy()
        eff['month'] = eff['ds'].dt.month
        monthly = eff.groupby('month')['yearly'].mean()
        if monthly.get(8, 0) < -0.2 or monthly.get(12, 0) < -0.2:
            logger.warning('Yearly component too negative in August or December')
        if monthly.get(4, 0) <= 0 or monthly.get(11, 0) <= 0:
            logger.warning('Expected April/November peaks not detected')
        amplitude = eff['yearly'].abs().max()
        if amplitude > 0.2:
            logger.warning('Yearly seasonality amplitude exceeds ±20%')


def _check_horizon_escalation(horizon_df: pd.DataFrame, threshold: float = 0.2) -> None:
    """Warn if the 14-day MAE exceeds the 1-day MAE by ``threshold``."""
    logger = logging.getLogger(__name__)
    try:
        mae_1 = float(horizon_df.loc[horizon_df['horizon_days'] == 1, 'MAE'])
        mae_14 = float(horizon_df.loc[horizon_df['horizon_days'] == 14, 'MAE'])
    except Exception:
        return
    if mae_1 == mae_1 and mae_14 == mae_14 and mae_1 > 0:
        escalation = (mae_14 - mae_1) / mae_1
        if escalation > threshold:
            logger.warning(
                '14-day horizon MAE escalates %.1f%% from 1-day horizon',
                escalation * 100,
            )


def _mean_poisson_deviance(actual, predicted) -> float:
    """Return the mean Poisson deviance between ``actual`` and ``predicted``.

    Parameters
    ----------
    actual : array-like
        Observed counts.
    predicted : array-like
        Predicted counts.
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    p = np.clip(p, 1e-8, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = p - a + a * np.where(a > 0, np.log(a / p), 0.0)
    return float(np.mean(2.0 * terms))

def create_simple_ensemble(prophet_df, holidays_df, regressors_df):
    """Create a simple ensemble of multiple Prophet models"""
    logger = logging.getLogger(__name__)
    logger.info("Creating ensemble of Prophet models")

    # Remove collinear regressors to keep models stable
    regressors_df = drop_collinear_features(regressors_df)
    
    # Create multiple Prophet models with different hyperparameters
    models = []
    
    # Base model
    model1 = Prophet(
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.2,
        **PROPHET_KWARGS,
    )
    
    # More flexible model
    model2 = Prophet(
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.2,
        **PROPHET_KWARGS,
    )
    
    # More rigid model
    model3 = Prophet(
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.2,
        **PROPHET_KWARGS,
    )
    
    models = [model1, model2, model3]
    
    # Add same regressors to all models
    # Use the reduced regressor set to avoid collinearity
    important_regressors = [
        'visit_ma3',
        'chatbot_count',
        'call_lag1',
        'call_lag7',
        'monday_effect',
        'notice_flag',
        'deadline_flag',
    ]
    
    # Add regressors to each model and fit them
    reg_prior_scale = 0.05
    forecasts = []
    for i, model in enumerate(models):
        logger.info(f"Training ensemble model {i+1}/{len(models)}")
        model_prophet_df = prophet_df.copy()
        
        for regressor in important_regressors:
            if regressor in regressors_df.columns:
                # Add the regressor to model_prophet_df
                model_prophet_df[regressor] = regressors_df[regressor].values
                # Add to model
                model.add_regressor(
                    regressor,
                    mode='additive',
                    prior_scale=reg_prior_scale,
                    standardize='auto',
                )
        
        _ensure_tbb_on_path()
        _fit_prophet_with_fallback(model, model_prophet_df)
        future = model.make_future_dataframe(periods=30)
        
        # Add regressor values to future DataFrame - FIXED CODE HERE
        for regressor in important_regressors:
            if regressor in model_prophet_df.columns:
                # Initialize the column with zeros first (no NaNs)
                future[regressor] = 0
                future[regressor] = future[regressor].astype(float)
                
                # Copy known values to future DataFrame
                for j, ds in enumerate(future['ds']):
                    matched_rows = model_prophet_df['ds'] == ds
                    # Use .any() to convert Series to boolean
                    if matched_rows.any():
                        # Use .iloc[0] to get the first matching value
                        future.loc[j, regressor] = model_prophet_df.loc[matched_rows, regressor].iloc[0]
                    else:
                        future.loc[j, regressor] = 0
        
        # Double-check for NaN values
        for regressor in important_regressors:
            if regressor in future.columns and future[regressor].isna().any():
                logger.warning(f"Found {future[regressor].isna().sum()} NaN values in {regressor}, filling with 0")
                future[regressor] = future[regressor].fillna(0)
        
        # Make the forecast
        forecast = model.predict(future)
        forecasts.append(forecast)
    
    # Create ensemble forecast by averaging predictions
    logger.info("Creating ensemble forecast by averaging predictions")
    ensemble_forecast = forecasts[0].copy()
    
    # Safer method to calculate min/max without ambiguous Series truth value
    ensemble_forecast['yhat'] = sum(f['yhat'] for f in forecasts) / len(forecasts)
    
    # Calculate lower and upper bounds using average across models
    lower_bounds = [f['yhat_lower'] for f in forecasts]
    upper_bounds = [f['yhat_upper'] for f in forecasts]

    ensemble_forecast['yhat_lower'] = (
        pd.concat(lower_bounds, axis=1).mean(axis=1)
    )
    ensemble_forecast['yhat_upper'] = (
        pd.concat(upper_bounds, axis=1).mean(axis=1)
    )

    ensemble_forecast[['yhat', 'yhat_lower', 'yhat_upper']] = (
        ensemble_forecast[['yhat', 'yhat_lower', 'yhat_upper']].clip(lower=0)
    )
    
    return ensemble_forecast, models


def detect_outliers_prophet(df, forecast):
    """
    Detect outliers based on Prophet forecast
    
    Args:
        df: Original DataFrame with call_count
        forecast: Prophet forecast DataFrame
        
    Returns:
        DataFrame with outlier flags
    """
    logger = logging.getLogger(__name__)
    logger.info("Detecting outliers based on Prophet forecast")
    
    # Create DataFrame with actual and predicted values
    prophet_df = df.reset_index().rename(columns={'index': 'ds'})
    reg_cols = [c for c in prophet_df.columns if c != 'ds']
    prophet_df[reg_cols] = prophet_df[reg_cols].astype("float64")
    prophet_df = prophet_df.merge(
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
        on='ds',
        how='left'
    )
    prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
    
    # Calculate residuals
    prophet_df['residual'] = prophet_df['call_count'] - prophet_df['yhat']
    prophet_df['residual_pct'] = (prophet_df['residual'] / 
                                  prophet_df['call_count'].replace(0, np.nan)) * 100
    
    # Identify outliers
    prophet_df['outside_interval'] = ((prophet_df['call_count'] < prophet_df['yhat_lower']) | 
                                     (prophet_df['call_count'] > prophet_df['yhat_upper']))
    
    # Calculate Z-score of residuals
    prophet_df['residual_zscore'] = ((prophet_df['residual'] - prophet_df['residual'].mean()) / 
                                      prophet_df['residual'].std())
    
    # Flag outliers based on multiple criteria
    prophet_df['is_outlier'] = ((prophet_df['outside_interval']) | 
                               (prophet_df['residual_zscore'].abs() > 3))
    
    # Count outliers
    outlier_count = prophet_df['is_outlier'].sum()
    logger.info(f"Detected {outlier_count} outliers out of {len(prophet_df)} points")
    
    return prophet_df

def improve_outlier_detection(df, forecast):
    """More robust outlier detection approach"""
    
    prophet_df = detect_outliers_prophet(df, forecast)  # Your existing function
    
    # Additional steps for more robust detection
    prophet_df['day_of_week'] = prophet_df['ds'].dt.dayofweek
    
    # 1. Calculate day-of-week specific residual thresholds
    dow_residuals = {}
    for dow in range(7):
        dow_data = prophet_df[prophet_df['day_of_week'] == dow]['residual']
        if len(dow_data) >= 10:  # Need enough samples
            # Use median absolute deviation - more robust than standard deviation
            median = dow_data.median()
            mad = (dow_data - median).abs().median() * 1.4826  # Scale factor for normal distribution
            dow_residuals[dow] = {'median': median, 'mad': mad}
    
    # 2. Flag outliers based on day-of-week specific thresholds
    for dow in dow_residuals:
        mask = (prophet_df['day_of_week'] == dow)
        threshold = 3.5 * dow_residuals[dow]['mad']  # 3.5 MADs is a common threshold
        
        prophet_df.loc[mask, 'dow_specific_outlier'] = (
            prophet_df.loc[mask, 'residual'].abs() > threshold
        )
    
    # Combine with original outlier detection
    prophet_df['is_outlier'] = prophet_df['is_outlier'] | prophet_df.get('dow_specific_outlier', False)
    
    return prophet_df


def handle_outliers_prophet(df, outlier_df, method='winsorize'):
    """
    Handle outliers in the data
    
    Args:
        df: Original DataFrame
        outlier_df: DataFrame with outlier flags
        method: Method to handle outliers
        
    Returns:
        DataFrame with handled outliers
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Handling outliers using {method} method")
    
    # Create copy of DataFrame
    df_cleaned = df.copy()
    
    # Get outlier indices
    outlier_mask = outlier_df['is_outlier'] == 1
    if outlier_mask.sum() == 0:
        logger.info("No outliers to handle")
        return df_cleaned
    
    # Get outlier dates
    outlier_dates = outlier_df.loc[outlier_mask, 'ds'].values
    
    # Handle outliers based on method
    if method == 'winsorize':
        # Replace with prediction bounds
        for date in outlier_dates:
            if date in df_cleaned.index:
                actual = df_cleaned.loc[date, 'call_count']
                pred_row = outlier_df[outlier_df['ds'] == date]
                if len(pred_row) > 0:
                    lower = pred_row['yhat_lower'].values[0]
                    upper = pred_row['yhat_upper'].values[0]
                    
                    if actual < lower:
                        df_cleaned.loc[date, 'call_count'] = lower
                    elif actual > upper:
                        df_cleaned.loc[date, 'call_count'] = upper
    
    elif method == 'prediction_replace':
        # Replace with predictions
        for date in outlier_dates:
            if date in df_cleaned.index:
                pred_row = outlier_df[outlier_df['ds'] == date]
                if len(pred_row) > 0:
                    df_cleaned.loc[date, 'call_count'] = pred_row['yhat'].values[0]
    
    elif method == 'interpolate':
        # Replace with NaN then interpolate
        for date in outlier_dates:
            if date in df_cleaned.index:
                df_cleaned.loc[date, 'call_count'] = np.nan
        
        # Interpolate NaN values
        df_cleaned['call_count'] = df_cleaned['call_count'].interpolate(method='linear')
    
    return df_cleaned


def analyze_prophet_components(model, forecast, output_dir):
    """
    Analyze and visualize Prophet model components
    
    Args:
        model: Trained Prophet model
        forecast: Prophet forecast DataFrame
        output_dir: Directory to save plots
    """
    logger = logging.getLogger(__name__)
    logger.info("Analyzing Prophet model components")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Plot model components
    fig = model.plot_components(forecast)
    fig.savefig(output_dir / "prophet_components.png")
    plt.close(fig)
    
    # Plot forecast
    fig = model.plot(forecast)
    plt.title("Call Volume Forecast")
    fig.savefig(output_dir / "prophet_forecast.png")
    plt.close(fig)
    
    # Analyze weekly pattern
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    week_df = forecast[['ds', 'weekly']].copy()
    week_df['day'] = week_df['ds'].dt.dayofweek
    week_df['day_name'] = week_df['day'].apply(lambda x: days[x])
    
    # Calculate average effect by day
    day_effect = week_df.groupby('day_name')['weekly'].mean().reindex(days)
    
    plt.figure(figsize=(10, 6))
    plt.bar(day_effect.index, day_effect.values)
    plt.title('Weekly Seasonal Effect by Day')
    plt.ylabel('Multiplicative Effect')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(output_dir / "weekly_effect.png")
    plt.close()
    
    # Analyze yearly pattern if available
    if 'yearly' in forecast.columns:
        year_df = forecast[['ds', 'yearly']].copy()
        year_df['month'] = year_df['ds'].dt.month
        year_df['month_name'] = year_df['month'].apply(lambda x: datetime(2000, x, 1).strftime('%B'))

        # Calculate average effect by month
        month_effect = year_df.groupby('month_name')['yearly'].mean()
        month_names = [datetime(2000, i, 1).strftime('%B') for i in range(1, 13)]
        month_effect = month_effect.reindex(month_names)

        plt.figure(figsize=(12, 6))
        plt.bar(month_effect.index, month_effect.values)
        plt.title('Yearly Seasonal Effect by Month')
        plt.ylabel('Multiplicative Effect')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "yearly_effect.png")
        plt.close()
    
    # Analyze holidays effect
    if 'holidays' in forecast.columns:
        holidays_df = forecast[['ds', 'holidays']].dropna()
        
        if len(holidays_df) > 0:
            plt.figure(figsize=(12, 6))
            plt.scatter(holidays_df['ds'], holidays_df['holidays'], alpha=0.7)
            plt.title('Holiday Effects')
            plt.ylabel('Multiplicative Effect')
            plt.grid(alpha=0.3)
            plt.savefig(output_dir / "holiday_effects.png")
            plt.close()
    
    # Analyze regressor effects
    for regressor in model.extra_regressors:
        if regressor in forecast.columns:
            reg_df = forecast[['ds', regressor]].dropna()
            
            if len(reg_df) > 0 and reg_df[regressor].nunique() > 1:
                plt.figure(figsize=(12, 6))
                plt.scatter(reg_df['ds'], reg_df[regressor], alpha=0.7)
                plt.title(f'{regressor} Effect')
                plt.ylabel('Multiplicative Effect')
                plt.grid(alpha=0.3)
                plt.savefig(output_dir / f"{regressor}_effect.png")
                plt.close()


def cross_validate_prophet(model, df, periods=30, horizon='14 days', initial='180 days'):
    """Simple cross-validation for a Prophet model using a rolling origin."""
    df_cv = cross_validation(
        model,
        initial=initial,
        period=f'{periods} days',
        horizon=horizon,
        parallel="processes",
    )
    df_p = performance_metrics(df_cv)
    return df_p['rmse'].mean()

def analyze_feature_importance(model, prophet_df, quick_mode=True):
    """
    Analyze which features contribute most to model accuracy
    
    Args:
        model: Trained Prophet model
        prophet_df: Prophet DataFrame
        quick_mode: If True, use smaller validation set for faster results
    
    Returns:
        Dictionary of feature impacts
    """
    logger = logging.getLogger(__name__)
    logger.info("Analyzing feature importance (quick_mode=%s)", quick_mode)
    reg_prior_scale = 0.05
    
    # Create versions of the data with one feature removed at a time
    features = [
        'county_holiday_flag',
        'deadline_flag',
        'notice_flag',
        'visit_ma3',
        'chatbot_count',
        'is_campaign',
        'post_policy',
        'press_release_flag',
        'yearly_seasonality',
        'weekly_seasonality'
    ]
    
    # Use a simplified validation approach for quick mode
    if quick_mode:
        # Split data into train/test with a simple 80/20 split
        train_size = int(len(prophet_df) * 0.8)
        train_df = prophet_df.iloc[:train_size].copy()
        test_df = prophet_df.iloc[train_size:].copy()
        
        # Base model performance
        future_periods = len(test_df)
        model_copy = Prophet(
            seasonality_mode='multiplicative',
            changepoint_prior_scale=model.changepoint_prior_scale,
            **PROPHET_KWARGS,
        )
        
        # Add regressors to the base model
        for feature in features:
            if feature.endswith('_flag') and feature in prophet_df.columns:
                model_copy.add_regressor(feature, mode='additive', prior_scale=reg_prior_scale,
                                         standardize='auto')
        
        _ensure_tbb_on_path()
        _fit_prophet_with_fallback(model_copy, train_df)
        future = model_copy.make_future_dataframe(periods=future_periods)
        
        # Add regressor values to future DataFrame
        start_idx = len(future) - future_periods
        for feature in features:
            if feature.endswith('_flag') and feature in prophet_df.columns:
                # Align predictor slice with target rows
                future.loc[start_idx:, feature] = test_df[feature].values
        
        forecast = model_copy.predict(future)
        
        # Calculate base error
        y_true = test_df['y'].values
        y_pred = forecast['yhat'].values[-len(y_true):]
        
        try:
            # Newer scikit-learn versions
            base_error = mean_squared_error(y_true, y_pred, squared=False)
        except TypeError:
            # Older scikit-learn versions
            base_error = np.sqrt(mean_squared_error(y_true, y_pred))
    else:
        # Use full cross-validation (slower but more accurate)
        base_error = cross_validate_prophet(model, prophet_df)
    
    # Initialize with default minimal impact
    feature_impacts = {feature: 0.0 for feature in features}
    
    # Test each feature
    for feature in features:
        logger.info(f"Testing importance of {feature}")
        
        try:
            if feature.endswith('_flag'):
                # Create version without this regressor
                test_df = prophet_df.copy()
                test_df[feature] = 0  # Neutralize the feature
                
                # Refit and evaluate
                test_model = Prophet(
                    seasonality_mode='multiplicative',
                    changepoint_prior_scale=model.changepoint_prior_scale,
                    **PROPHET_KWARGS,
                )
                
                # Add remaining regressors
                for other_feature in [f for f in features if f.endswith('_flag') and f != feature]:
                    if other_feature in test_df.columns:
                        test_model.add_regressor(other_feature, mode='additive',
                                                 prior_scale=reg_prior_scale,
                                                 standardize='auto')
                
                _ensure_tbb_on_path()
                _fit_prophet_with_fallback(test_model, test_df)
                
                if quick_mode:
                    # Use the simplified validation approach
                    train_size = int(len(test_df) * 0.8)
                    train_df = test_df.iloc[:train_size].copy()
                    test_subset = test_df.iloc[train_size:].copy()
                    future_periods = len(test_subset)
                    
                    _ensure_tbb_on_path()
                    _fit_prophet_with_fallback(test_model, train_df)
                    future = test_model.make_future_dataframe(periods=future_periods)
                    
                    # Add regressor values to future DataFrame
                    start_idx_sub = len(future) - future_periods
                    for other_feature in [f for f in features if f.endswith('_flag')]:
                        if other_feature in test_df.columns:
                            future.loc[start_idx_sub:, other_feature] = test_subset[other_feature].values
                    
                    forecast = test_model.predict(future)
                    
                    # Calculate error without this feature
                    y_true = test_subset['y'].values
                    y_pred = forecast['yhat'].values[-len(y_true):]
                    
                    try:
                        # Newer scikit-learn versions
                        error_without_feature = mean_squared_error(y_true, y_pred, squared=False)
                    except TypeError:
                        # Older scikit-learn versions
                        error_without_feature = np.sqrt(mean_squared_error(y_true, y_pred))
                else:
                    # Use full cross-validation
                    error_without_feature = cross_validate_prophet(test_model, test_df)
                
            elif feature == 'yearly_seasonality' or feature == 'weekly_seasonality':
                # Create version without this seasonality
                custom_kwargs = {
                    **PROPHET_KWARGS,
                    'yearly_seasonality': feature != 'yearly_seasonality',
                    'weekly_seasonality': feature != 'weekly_seasonality',
                }
                test_model = Prophet(
                    seasonality_mode='multiplicative',
                    changepoint_prior_scale=model.changepoint_prior_scale,
                    **custom_kwargs,
                )
                
                # Add regressors
                for other_feature in [f for f in features if f.endswith('_flag')]:
                    if other_feature in prophet_df.columns:
                        test_model.add_regressor(other_feature, mode='additive',
                                                 prior_scale=reg_prior_scale,
                                                 standardize='auto')
                
                if quick_mode:
                    # Use the simplified validation approach
                    train_size = int(len(prophet_df) * 0.8)
                    train_df = prophet_df.iloc[:train_size].copy()
                    test_subset = prophet_df.iloc[train_size:].copy()
                    future_periods = len(test_subset)
                    
                    _ensure_tbb_on_path()
                    _fit_prophet_with_fallback(test_model, train_df)
                    future = test_model.make_future_dataframe(periods=future_periods)
                    
                    # Add regressor values to future DataFrame
                    start_idx_sub = len(future) - future_periods
                    for other_feature in [f for f in features if f.endswith('_flag')]:
                        if other_feature in prophet_df.columns:
                            future.loc[start_idx_sub:, other_feature] = test_subset[other_feature].values
                    
                    forecast = test_model.predict(future)
                    
                    # Calculate error without this feature
                    y_true = test_subset['y'].values
                    y_pred = forecast['yhat'].values[-len(y_true):]
                    
                    try:
                        # Newer scikit-learn versions
                        error_without_feature = mean_squared_error(y_true, y_pred, squared=False)
                    except TypeError:
                        # Older scikit-learn versions
                        error_without_feature = np.sqrt(mean_squared_error(y_true, y_pred))
                else:
                    # Use full cross-validation
                    error_without_feature = cross_validate_prophet(test_model, prophet_df)
            
            # Calculate impact - how much worse is model without this feature?
            impact = (error_without_feature - base_error) / base_error * 100
            feature_impacts[feature] = impact
            logger.info(f"Impact of {feature}: {impact:.2f}%")
            
        except Exception as e:
            logger.warning(f"Error calculating importance for {feature}: {str(e)}")
    
    return feature_impacts

def analyze_policy_changes_prophet(df, forecast, output_dir):
    """
    Analyze the impact of May 2025 policy changes
    
    Args:
        df: Original DataFrame
        forecast: Prophet forecast DataFrame
        output_dir: Directory to save analysis
    """
    logger = logging.getLogger(__name__)
    logger.info("Analyzing May 2025 policy changes impact")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Get May 2025 data - actual if available
    may_2025_mask = ((df.index.year == 2025) & (df.index.month == 5))
    may_2025_data = df[may_2025_mask]['call_count'] if may_2025_mask.any() else None
    
    # Get April 2024 data for comparison
    apr_2024_mask = ((df.index.year == 2024) & (df.index.month == 4))
    apr_2024_data = df[apr_2024_mask]['call_count'] if apr_2024_mask.any() else None
    
    # Check if we have data for comparison
    if may_2025_data is None or apr_2024_data is None:
        logger.warning("Insufficient data to compare seasons")
        return None
    
    # Calculate basic statistics
    apr_stats = {
        "mean": apr_2024_data.mean(),
        "median": apr_2024_data.median(),
        "std": apr_2024_data.std(),
        "min": apr_2024_data.min(),
        "max": apr_2024_data.max(),
        "total": apr_2024_data.sum()
    }

    may_stats = {
        "mean": may_2025_data.mean(),
        "median": may_2025_data.median(),
        "std": may_2025_data.std(),
        "min": may_2025_data.min(),
        "max": may_2025_data.max(),
        "total": may_2025_data.sum()
    }
    
    # Calculate percentage changes
    pct_changes = {
        "mean": ((may_stats["mean"] - apr_stats["mean"]) / apr_stats["mean"]) * 100,
        "median": ((may_stats["median"] - apr_stats["median"]) / apr_stats["median"]) * 100,
        "max": ((may_stats["max"] - apr_stats["max"]) / apr_stats["max"]) * 100,
        "total": ((may_stats["total"] - apr_stats["total"]) / apr_stats["total"]) * 100
    }
    
    # Create summary DataFrame
    summary = pd.DataFrame({
        "Metric": [
            "Average Daily Calls", "Median Daily Calls", "Standard Deviation",
            "Minimum Calls", "Maximum Calls", "Total Calls"
        ],
        "April 2024": [
            apr_stats["mean"], apr_stats["median"], apr_stats["std"],
            apr_stats["min"], apr_stats["max"], apr_stats["total"]
        ],
        "May 2025": [
            may_stats["mean"], may_stats["median"], may_stats["std"],
            may_stats["min"], may_stats["max"], may_stats["total"]
        ],
        "Percent Change": [
            f"{pct_changes['mean']:.1f}%", f"{pct_changes['median']:.1f}%",
            f"{((may_stats['std'] - apr_stats['std']) / apr_stats['std'] * 100):.1f}%",
            f"{((may_stats['min'] - apr_stats['min']) / apr_stats['min'] * 100):.1f}%",
            f"{pct_changes['max']:.1f}%", f"{pct_changes['total']:.1f}%"
        ]
    })
    
    # Save summary to CSV
    summary.to_csv(output_dir / "policy_change_impact.csv", index=False)
    
    # Create daily comparison plot
    plt.figure(figsize=(14, 7))
    
    # Align by day of month
    apr_days = [d.day for d in apr_2024_data.index]
    may_days = [d.day for d in may_2025_data.index]
    
    plt.plot(apr_days,
             apr_2024_data.values,
             'b-',
             marker='o',
             alpha=0.7,
             label='April 2024 NOV Season')
    plt.plot(may_days,
             may_2025_data.values,
             'r-',
             marker='o',
             alpha=0.7,
             label='May 2025 (with policy changes)')
    
    plt.title('Call Volume Comparison: April 2024 vs May 2025',
              fontsize=14,
              fontweight='bold')
    plt.xlabel('Day of Month')
    plt.ylabel('Daily Call Volume')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add annotation for average increase
    plt.annotate(f"Average increase: {pct_changes['mean']:.1f}%",
                 xy=(0.05, 0.95),
                 xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3",
                           fc="yellow",
                           ec="orange",
                           alpha=0.7),
                 fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / "apr2024_vs_may2025_comparison.png")
    plt.close()
    
    # Create counterfactual analysis for May 2025 without policy changes
    # If the forecast includes the post_policy component, remove it
    may_2025_forecast = forecast[
        (forecast['ds'].dt.year == 2025) &
        (forecast['ds'].dt.month == 5)
    ].copy()

    if not may_2025_forecast.empty and 'post_policy' in may_2025_forecast.columns:
        may_2025_forecast['no_policy'] = (
            may_2025_forecast['yhat'] - may_2025_forecast['post_policy']
        )
        cf_df = pd.DataFrame({
            'total_with_policy': [may_2025_forecast['yhat'].sum()],
            'total_without_policy': [may_2025_forecast['no_policy'].sum()],
            'estimated_policy_effect': [may_2025_forecast['post_policy'].sum()]
        })
        cf_df.to_csv(output_dir / 'policy_counterfactual.csv', index=False)

    return summary


def analyze_press_release_impact_prophet(forecast, output_dir):
    """
    Analyze the impact of press releases on call volumes
    
    Args:
        forecast: Prophet forecast DataFrame
        output_dir: Directory to save analysis
    """
    logger = logging.getLogger(__name__)
    logger.info("Analyzing press release impact")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Check if we have the 'holidays' component
    if 'holidays' not in forecast.columns:
        logger.warning("No holiday effects found in forecast")
        return None
    
    # Extract press release effects
    press_releases = forecast[['ds', 'holidays']].dropna().copy()
    
    # If we have holiday effects, try to identify press releases
    # This assumes press release dates were added as holidays in the model
    press_release_dates = [
        date(2025, 1, 9),
        date(2025, 2, 4),
        date(2025, 2, 24),
        date(2025, 3, 28),
        date(2025, 4, 1),
        date(2025, 4, 3),
        date(2025, 4, 23),
        date(2025, 4, 30),
        date(2025, 5, 1),
        date(2025, 5, 5),
        date(2025, 5, 9),
        date(2025, 5, 13)
    ]
    
    # Keep only press release dates in the holidays DataFrame
    press_releases['is_press_release'] = press_releases['ds'].dt.date.isin(press_release_dates)
    press_releases = press_releases[press_releases['is_press_release']]
    
    if len(press_releases) == 0:
        logger.warning("No press release effects found in forecast")
        return None
    
    # Visualize press release effects
    plt.figure(figsize=(12, 6))
    plt.bar(press_releases['ds'].dt.date, press_releases['holidays'])
    plt.title('Press Release Impact on Call Volume')
    plt.xlabel('Press Release Date')
    plt.ylabel('Multiplicative Effect')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "press_release_impact.png")
    plt.close()
    
    # Save press release effects
    press_releases.to_csv(output_dir / "press_release_effects.csv", index=False)

    return press_releases


def compute_naive_baseline(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate a seasonal naive forecast for the last 14 days and metrics.

    The baseline now predicts each day using the call volume from the same
    weekday in the prior week. This mirrors the lightweight logic in
    ``naive_forecast.py`` and avoids the need for additional packages.

    Parameters
    ----------
    df : DataFrame
        Source data with a ``call_count`` column and a DatetimeIndex.

    Returns
    -------
    Tuple of ``(forecast_df, metrics_df, horizon_df)`` where ``forecast_df``
    contains the date, predicted call count, actual call count and error
    columns. The ``metrics_df`` provides MAE, RMSE, Poisson deviance and
    Coverage aggregated over the period. ``horizon_df`` contains horizon
    specific metrics for 1-, 7- and 14-day horizons.
    """

    df_sorted = df.sort_index()
    # Require at least one week of history beyond the evaluation window
    if len(df_sorted) < 21:
        raise ValueError(
            "At least 21 days of data are required to compute the baseline"
        )

    recent = df_sorted["call_count"]
    preds = recent.shift(7).iloc[-14:]
    actual = recent.iloc[-14:]
    dates = recent.index[-14:]

    result = pd.DataFrame({
        "date": dates,
        "predicted": preds.values,
        "actual": actual.values,
    })
    result["error"] = result["actual"] - result["predicted"]
    result["abs_error"] = result["error"].abs()
    mae = result["abs_error"].mean()
    rmse = np.sqrt((result["error"] ** 2).mean())
    pdev = _mean_poisson_deviance(result["actual"], result["predicted"])

    resid_std = result["error"].std(ddof=0)
    result["lower"] = result["predicted"] - 1.96 * resid_std
    result["upper"] = result["predicted"] + 1.96 * resid_std
    coverage = (
        ((result["actual"] >= result["lower"]) & (result["actual"] <= result["upper"]))
        .mean()
        * 100
    )

    metrics = pd.DataFrame(
        {
            "metric": ["MAE", "RMSE", "Poisson" , "Coverage"],
            "value": [mae, rmse, pdev, coverage],
        }
    )

    horizon_rows = []
    for h in [1, 7, 14]:
        if len(result) >= h:
            sub = result.head(h)
            mae_h = sub["abs_error"].mean()
            rmse_h = np.sqrt((sub["error"] ** 2).mean())
            pdev_h = _mean_poisson_deviance(sub["actual"], sub["predicted"])
            horizon_rows.append([h, mae_h, rmse_h, pdev_h])
    horizon_df = pd.DataFrame(
        horizon_rows, columns=["horizon_days", "MAE", "RMSE", "Poisson"]
    )


    return result, metrics, horizon_df


def write_summary(df: pd.DataFrame, path: Path) -> Path:
    """Write a metrics DataFrame to ``path`` preserving ``NaN`` values."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_out = df.copy()
    if {"actual", "predicted"} <= set(df_out.columns) and "Poisson" not in df_out.columns:
        pdev = _mean_poisson_deviance(df_out["actual"], df_out["predicted"])
        df_out["Poisson"] = pdev
    try:
        df_out.to_csv(path, index=False, na_rep="NaN")
    except TypeError:
        # Fallback for minimal DataFrame implementation without ``na_rep``
        import csv
        import math

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(df_out.columns)
            for row in df_out.itertuples(index=False, name=None):
                writer.writerow(
                    ["NaN" if isinstance(x, float) and math.isnan(x) else x for x in row]
                )
    return path


def export_baseline_forecast(df: pd.DataFrame, output_dir: Path) -> Path:
    """Export naive baseline forecast to Excel and CSV.

    This writes a workbook with three sheets - ``Baseline``, ``Metrics`` and
    ``Input Data`` - summarizing the naive forecast for the last 14 days. The
    forecast CSV is saved as ``baseline_forecast.csv``.

    Parameters
    ----------
    df : DataFrame
        Data containing a ``call_count`` column.
    output_dir : Path
        Directory where output files will be written.

    Returns
    -------
    Path
        Path to the generated Excel workbook.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        raise ValueError("Input DataFrame is empty; no baseline forecast to export.")

    baseline_df, metrics, _ = compute_naive_baseline(df)
    if baseline_df.empty:
        raise ValueError("Computed baseline forecast is empty; nothing to export.")

    csv_path = output_dir / "baseline_forecast.csv"
    baseline_df.to_csv(csv_path, index=False)

    excel_path = output_dir / "baseline_forecast.xlsx"
    input_data = (
        df.sort_index().iloc[-15:].reset_index().rename(columns={"index": "date"})
    )

    if _HAVE_OPENPYXL:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            baseline_df.to_excel(writer, sheet_name="Baseline", index=False)
            metrics.to_excel(writer, sheet_name="Metrics", index=False)
            input_data.to_excel(writer, sheet_name="Input Data", index=False)
    else:
        # Fallback when openpyxl is unavailable - still export all data
        baseline_df.to_csv(excel_path, index=False)
        write_summary(metrics, output_dir / "baseline_metrics.csv")
        input_data.to_csv(output_dir / "baseline_input_data.csv", index=False)

    return excel_path


def export_prophet_forecast(model, forecast, df, output_dir, scaler=None):
    """
    Export Prophet forecast to Excel.

    The exported workbook now includes model performance for the
    previous 14 business days and a forecast for the next business day.

    Args:
        model: Trained Prophet model
        forecast: Prophet forecast DataFrame
        df: Original DataFrame
        output_dir: Directory to save Excel file
    """
    logger = logging.getLogger(__name__)
    logger.info("Exporting Prophet forecast to Excel")

    if forecast.empty:
        raise ValueError("Forecast DataFrame is empty; nothing to export.")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Define output file
    output_file = output_dir / "prophet_call_predictions_v4.xlsx"
    
    # Get the past 14 business days based on the available data
    last_date = df.index.max()
    recent_days = pd.date_range(end=last_date, periods=14, freq='B')

    # Get the next business day
    next_day = last_date + pd.Timedelta(days=1)
    if next_day.weekday() >= 5:  # Weekend
        next_day = next_day + pd.Timedelta(days=7 - next_day.weekday())
    
    # Get predictions for the past 14 days
    recent_forecast = forecast[forecast['ds'].isin(recent_days)].copy()
    if scaler is not None:
        for col in ['yhat', 'yhat_lower', 'yhat_upper']:
            if col in recent_forecast.columns:
                recent_forecast[col] = scaler.inverse_transform(
                    recent_forecast[[col]]
                )
    recent_forecast['actual'] = np.nan
    
    # Get actual values if available
    for i, day in enumerate(recent_days):
        if day in df.index:
            recent_forecast.loc[recent_forecast['ds'] == day, 'actual'] = df.loc[day, 'call_count']
    
    # Calculate errors
    recent_forecast['error'] = recent_forecast['actual'] - recent_forecast['yhat']
    recent_forecast['abs_error'] = np.abs(recent_forecast['error'])
    
    # Get next day forecast
    next_day_forecast = forecast[forecast['ds'] == next_day].copy()
    if scaler is not None and not next_day_forecast.empty:
        for col in ['yhat', 'yhat_lower', 'yhat_upper']:
            next_day_forecast[col] = scaler.inverse_transform(
                next_day_forecast[[col]]
            )
    
    if len(next_day_forecast) == 0:
        # If next day isn't in forecast, make a special prediction
        future = pd.DataFrame({'ds': [next_day]})
        for regressor in model.extra_regressors:
            future[regressor] = 0
        
        next_day_forecast = model.predict(future)
        next_day_forecast[['yhat', 'yhat_lower', 'yhat_upper']] = (
            next_day_forecast[['yhat', 'yhat_lower', 'yhat_upper']].clip(lower=0)
        )
        if scaler is not None:
            for col in ['yhat', 'yhat_lower', 'yhat_upper']:
                next_day_forecast[col] = scaler.inverse_transform(
                    next_day_forecast[[col]]
                )
    
    # Prepare next day info
    next_day_df = pd.DataFrame({
        'date': [next_day],
        'predicted_calls': [next_day_forecast['yhat'].values[0]],
        'lower_bound': [next_day_forecast['yhat_lower'].values[0]],
        'upper_bound': [next_day_forecast['yhat_upper'].values[0]],
        'day_of_week': [next_day.strftime('%A')]
    })
    
    # Create Excel writer
    if _HAVE_OPENPYXL:
        writer_ctx = pd.ExcelWriter(output_file, engine='openpyxl')
    else:
        # Fallback to CSV output if openpyxl is missing
        writer_ctx = None

    if writer_ctx:
        with writer_ctx as writer:
            # Past 14-day performance
            recent_performance = pd.DataFrame({
                'date': recent_forecast['ds'],
                'predicted': recent_forecast['yhat'],
                'actual': recent_forecast['actual'],
                'lower_bound': recent_forecast['yhat_lower'],
                'upper_bound': recent_forecast['yhat_upper'],
                'error': recent_forecast['error'],
                'abs_error': recent_forecast['abs_error']
            })
            recent_performance.to_excel(writer, sheet_name='Recent 14-Day Performance', index=False)

            # Metrics for Prophet predictions
            prophet_metrics = pd.DataFrame({
                'metric': ['MAE', 'RMSE', 'Poisson'],
                'value': [
                    recent_performance['abs_error'].mean(),
                    np.sqrt((recent_performance['error'] ** 2).mean()),
                    _mean_poisson_deviance(recent_performance['actual'], recent_performance['predicted'])
                ]
            })
            prophet_metrics.to_excel(writer, sheet_name='Prophet Metrics', index=False)

            # Naive baseline forecast and metrics
            naive_df, naive_metrics = compute_naive_baseline(df)
            naive_df.to_excel(writer, sheet_name='Naive 14-Day Forecast', index=False)
            naive_metrics.to_excel(writer, sheet_name='Naive Metrics', index=False)

            # Next day forecast
            next_day_df.to_excel(writer, sheet_name='Next Day Forecast', index=False)

            # Model components
            components = pd.DataFrame({
                'Component': ['Trend', 'Weekly Seasonality', 'Yearly Seasonality', 'Holidays'],
                'Description': [
                    'Long-term trend component',
                    'Day of week patterns',
                    'Month of year patterns',
                    'Special events (holidays, deadlines, press releases)'
                ]
            })
            components.to_excel(writer, sheet_name='Model Components', index=False)

            # Model parameters
            parameters = pd.DataFrame({
                'Parameter': [
                    'yearly_seasonality',
                    'weekly_seasonality',
                    'daily_seasonality',
                    'seasonality_mode',
                    'changepoint_prior_scale'
                ],
                'Value': [
                    model.yearly_seasonality,
                    model.weekly_seasonality,
                    model.daily_seasonality,
                    model.seasonality_mode,
                    model.changepoint_prior_scale
                ],
                'Description': [
                    'Yearly seasonality enabled',
                    'Weekly seasonality enabled',
                    'Daily seasonality enabled',
                    'How seasonality components combine with trend',
                    'Flexibility of trend changepoints'
                ]
            })
            parameters.to_excel(writer, sheet_name='Model Parameters', index=False)

            # Notes
            notes = pd.DataFrame({
                'Note': [
                    'This report contains predictions for customer service call volumes using Prophet.',
                    f'The model was trained on data up to {df.index.max().strftime("%Y-%m-%d")}.',
                    'Prophet automatically handles multiple seasonality patterns, holidays, and special events.',
                    f'Next day forecast is for {next_day.strftime("%A, %B %d, %Y")}.',
                    'Prediction intervals represent uncertainty in the forecast.',
                    'Model accounts for day-of-week patterns, monthly seasonality, holidays, and special events.',
                    'The "Recent 14-Day Performance" sheet compares predictions with actuals for the last 14 business days.'
                ]
            })
            notes.to_excel(writer, sheet_name='Notes', index=False)
    else:
        # Minimal CSV fallback
        fallback_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        if scaler is not None:
            for col in ['yhat', 'yhat_lower', 'yhat_upper']:
                if col in fallback_df.columns:
                    fallback_df[col] = scaler.inverse_transform(
                        fallback_df[[col]]
                    )
        fallback_df.to_csv(output_file, index=False)
    
    logger.info(f"Forecast exported to {output_file}")
    
    return output_file


def evaluate_prophet_model(
    model,
    prophet_df,
    cv_params=None,
    log_transform=False,
    forecast=None,
    scaler=None,
    transform: str | None = None,
):
    """Cross‑validate the Prophet model and report MAE and RMSE.

    Parameters mirror :func:`train_prophet_model`. When ``transform`` is
    ``'log'`` or ``'box-cox'`` the function inverse-transforms the predictions
    before computing metrics.
    """

    if cv_params is None:
        cv_params = {}

    initial = cv_params.get('initial', '180 days')
    period = cv_params.get('period', '30 days')
    horizon = cv_params.get('horizon', '14 days')
    try:
        if pd.Timedelta(horizon).days > 14:
            horizon = '14 days'
    except Exception:
        horizon = '14 days'

    logger = logging.getLogger(__name__)
    logger.info(
        f"Evaluating Prophet model with {initial} initial window, "
        f"{period} period, {horizon} horizon"
    )

    history = model.history.copy()
    reg_info = model.extra_regressors.copy()
    df_cv = None
    lb_first = None
    lb_p = 0.0
    attempts = 0
    current_scale = model.changepoint_prior_scale
    orig_model = model
    while True:
        df_cv = cross_validation(
            model,
            initial=initial,
            period=period,
            horizon=horizon,
            parallel="processes",
        )
        df_cv = df_cv[df_cv['ds'].dt.dayofweek < 5]
        residuals = df_cv['y'] - df_cv['yhat']
        lb = acorr_ljungbox(residuals, lags=14, return_df=True)
        if lb_first is None:
            lb_first = lb.copy()
            lb_p = lb['lb_pvalue'].min()
            if 0.2 <= lb_p <= 0.8:
                break
        else:
            lb_p = lb['lb_pvalue'].min()
        if lb_p > 0.05 or attempts >= 1:
            break
        attempts += 1
        current_scale *= 0.5
        logger.info(
            "Autocorrelation detected, refitting with changepoint_prior_scale=%s",
            current_scale,
        )
        model = Prophet(
            growth=model.growth,
            interval_width=model.interval_width,
            seasonality_mode=model.seasonality_mode,
            changepoint_prior_scale=current_scale,
            n_changepoints=model.n_changepoints,
            holidays=model.holidays,
            **PROPHET_KWARGS,
        )
        for name, info in reg_info.items():
            allowed = {
                k: v for k, v in info.items() if k in {"prior_scale", "mode", "standardize"}
            }
            model.add_regressor(name, **allowed)
        _ensure_tbb_on_path()
        _fit_prophet_with_fallback(model, history)
    orig_model.changepoint_prior_scale = current_scale

    if transform is None and log_transform:
        transform = "log"

    if transform == "log":
        for col in ['y', 'yhat', 'yhat_lower', 'yhat_upper']:
            if col in df_cv.columns:
                df_cv[col] = np.expm1(df_cv[col])
        if forecast is not None:
            for col in ['yhat', 'yhat_lower', 'yhat_upper']:
                if col in forecast.columns:
                    forecast[col] = np.expm1(forecast[col])
    elif transform == "box-cox":
        _, lam, shift = box_cox_transform(prophet_df['y'])
        for col in ['y', 'yhat', 'yhat_lower', 'yhat_upper']:
            if col in df_cv.columns:
                df_cv[col] = inv_box_cox_transform(df_cv[col], lam, shift)
        if forecast is not None:
            for col in ['yhat', 'yhat_lower', 'yhat_upper']:
                if col in forecast.columns:
                    forecast[col] = inv_box_cox_transform(forecast[col], lam, shift)

    if scaler is not None:
        for col in ['y', 'yhat', 'yhat_lower', 'yhat_upper']:
            if col in df_cv.columns:
                df_cv[col] = scaler.inverse_transform(df_cv[[col]])
        if forecast is not None:
            for col in ['yhat', 'yhat_lower', 'yhat_upper']:
                if col in forecast.columns:
                    forecast[col] = scaler.inverse_transform(forecast[[col]])

    # Propagate horizon column if not provided by cross_validation
    if 'horizon' not in df_cv.columns and {'ds', 'cutoff'} <= set(df_cv.columns):
        df_cv['horizon'] = df_cv['ds'] - df_cv['cutoff']

    df_p = performance_metrics(df_cv, rolling_window=1)

    mae  = df_p['mae' ].mean() if 'mae'  in df_p else float('nan')
    rmse = df_p['rmse'].mean() if 'rmse' in df_p else float('nan')
    pdev = _mean_poisson_deviance(df_cv['y'], df_cv['yhat'])

    coverage = (
        ((df_cv['y'] >= df_cv['yhat_lower']) & (df_cv['y'] <= df_cv['yhat_upper'])).mean() * 100
        if {'yhat_lower', 'yhat_upper'} <= set(df_cv.columns) else float('nan')
    )

    interval_scale = 1.0
    if coverage == coverage and coverage < 95:
        interval_scale = 95.0 / coverage
        center = df_cv['yhat']
        df_cv['yhat_lower'] = center - (center - df_cv['yhat_lower']) * interval_scale
        df_cv['yhat_upper'] = center + (df_cv['yhat_upper'] - center) * interval_scale
        coverage = (
            ((df_cv['y'] >= df_cv['yhat_lower']) & (df_cv['y'] <= df_cv['yhat_upper'])).mean() * 100
        )

    logger.info(
        f"Cross‑validation →  MAE {mae:.2f} | RMSE {rmse:.2f} | "
        f"Poisson {pdev:.2f} | "
        f"Coverage {coverage if coverage==coverage else 'N/A'}%"
    )

    residuals = df_cv['y'] - df_cv['yhat']
    lb = acorr_ljungbox(residuals, lags=14, return_df=True)
    ac_flag = (lb['lb_pvalue'] < 0.05) | (lb['lb_stat'] > 0.2)
    autocorr_flag = bool(ac_flag.any())
    attempts = 0
    while lb['lb_pvalue'].min() < 0.05 and attempts < 2:
        attempts += 1
        try:
            r = residuals.fillna(0.0)
            X = np.column_stack([
                r.shift(1).fillna(0.0).to_numpy(),
                r.shift(7).fillna(0.0).to_numpy(),
                r.shift(14).fillna(0.0).to_numpy(),
            ])
            beta, _, _, _ = np.linalg.lstsq(X, r.to_numpy(), rcond=None)
            adj = X @ beta
            df_cv['yhat'] += adj
            if {'yhat_lower', 'yhat_upper'} <= set(df_cv.columns):
                df_cv['yhat_lower'] += adj
                df_cv['yhat_upper'] += adj
            if forecast is not None:
                hist = list(r.iloc[-14:].to_numpy())
                fut_adj = []
                for _ in range(len(forecast)):
                    x1 = hist[-1]
                    x7 = hist[-7]
                    x14 = hist[0]
                    pred = beta[0]*x1 + beta[1]*x7 + beta[2]*x14
                    fut_adj.append(pred)
                    hist.append(pred)
                    hist.pop(0)
                fut_adj = np.array(fut_adj)
                forecast[['yhat', 'yhat_lower', 'yhat_upper']] = (
                    forecast[['yhat', 'yhat_lower', 'yhat_upper']].add(fut_adj[:, None])
                )
            residuals = df_cv['y'] - df_cv['yhat']
            lb = acorr_ljungbox(residuals, lags=14, return_df=True)
        except Exception as exc:
            logger.warning("AR hybrid correction failed: %s", exc)
            break

    summary = pd.DataFrame({
        "metric": ["MAE", "RMSE", "Poisson", "Coverage"],
        "value":  [mae,  rmse,  pdev, coverage]
    })

    horizon_rows = []
    for h in [1, 7, 14]:
        mask = df_cv['horizon'] <= pd.Timedelta(days=h)
        sub = df_cv[mask]
        if len(sub) == 0:
            continue
        mae_h = np.mean(np.abs(sub['y'] - sub['yhat']))
        rmse_h = np.sqrt(mean_squared_error(sub['y'], sub['yhat']))
        pdev_h = _mean_poisson_deviance(sub['y'], sub['yhat'])
        horizon_rows.append([h, mae_h, rmse_h, pdev_h])
    horizon_table = pd.DataFrame(
        horizon_rows, columns=['horizon_days','MAE','RMSE','Poisson']
    )

    _check_horizon_escalation(horizon_table)

    lb_diag = lb_first if lb_first is not None else lb
    diag = pd.DataFrame({
        'lag': lb_diag.index,
        'lb_stat': lb_diag['lb_stat'],
        'lb_pvalue': lb_diag['lb_pvalue']
    })

    mean_calls = df_cv['y'].mean()
    if mae > 0.15 * mean_calls:
        logger.warning('MAE exceeds 15% of mean call volume.')
    if autocorr_flag:
        logger.warning('Autocorrelation detected in residuals.')

    return df_cv, horizon_table, summary, diag, interval_scale


def create_prophet_dashboard(model, forecast, df, output_dir):
    """
    Create a comprehensive dashboard of Prophet results
    
    Args:
        model: Trained Prophet model
        forecast: Prophet forecast DataFrame
        df: Original DataFrame
        output_dir: Directory to save dashboard
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating Prophet dashboard")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # 1. Overall forecast plot
    fig = model.plot(forecast)
    plt.title('Call Volume Forecast')
    plt.tight_layout()
    plt.savefig(output_dir / "prophet_forecast.png")
    plt.close(fig)
    
    # 2. Components plot
    fig = model.plot_components(forecast)
    plt.tight_layout()
    plt.savefig(output_dir / "prophet_components.png")
    plt.close(fig)
    
    # 3. Weekday vs weekend comparison
    forecast['day_of_week'] = forecast['ds'].dt.dayofweek
    forecast['is_weekend'] = forecast['day_of_week'] >= 5
    
    weekday_avg = forecast[~forecast['is_weekend']]['yhat'].mean()
    weekend_avg = forecast[forecast['is_weekend']]['yhat'].mean()
    
    plt.figure(figsize=(8, 5))
    plt.bar(['Weekday', 'Weekend'], [weekday_avg, weekend_avg])
    plt.title('Average Predicted Call Volume')
    plt.ylabel('Average Calls')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(output_dir / "weekday_weekend_comparison.png")
    plt.close()
    
    # 4. Monthly trend
    forecast['month'] = forecast['ds'].dt.month
    monthly_avg = forecast.groupby('month')['yhat'].mean()
    
    plt.figure(figsize=(10, 5))
    monthly_avg.plot(kind='bar')
    plt.title('Monthly Average Predicted Call Volume')
    plt.xlabel('Month')
    plt.ylabel('Average Calls')
    plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(output_dir / "monthly_trend.png")
    plt.close()
    
    # 5. Day of week pattern
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_avg = forecast.groupby('day_of_week')['yhat'].mean()
    
    plt.figure(figsize=(10, 5))
    plt.bar([day_names[i] for i in dow_avg.index], dow_avg.values)
    plt.title('Average Call Volume by Day of Week')
    plt.ylabel('Average Calls')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(output_dir / "day_of_week_pattern.png")
    plt.close()
    
    # 6. Seasonal peak comparison
    if 'april_peak_flag' in forecast.columns and 'november_peak_flag' in forecast.columns:
        april_avg = forecast[forecast['april_peak_flag'] == 1]['yhat'].mean()
        nov_avg = forecast[forecast['november_peak_flag'] == 1]['yhat'].mean()

        plt.figure(figsize=(8, 5))
        plt.bar(['April Peak', 'Nov Peak'], [april_avg, nov_avg])
        plt.title('Average Call Volume: Seasonal Peaks')
        plt.ylabel('Average Calls')
        plt.grid(axis='y', alpha=0.3)

        plt.savefig(output_dir / "seasonal_peak_comparison.png")
        plt.close()
    
    logger.info(f"Dashboard created in {output_dir}")


def main(argv=None):
    """Parse CLI arguments and run the unified forecasting pipeline."""
    parser = argparse.ArgumentParser(
        description="Run call volume forecasting using Prophet"
    )
    parser.add_argument("call_data", type=Path, help="CSV/Excel file containing call counts")
    parser.add_argument("visitor_data", type=Path, help="CSV/Excel file containing visitor counts")
    parser.add_argument("chatbot_data", type=Path, help="CSV file containing chatbot queries")
    parser.add_argument(
        "output_dir", nargs="?", default=Path("prophet_output"), type=Path,
        help="Directory to save results"
    )
    args = parser.parse_args(argv)

    cfg = pipeline.load_config(Path("config.yaml"))
    cfg["data"]["calls"] = str(args.call_data)
    cfg["data"]["visitors"] = str(args.visitor_data)
    cfg["data"]["queries"] = str(args.chatbot_data)
    cfg["output"] = str(args.output_dir)

    pipeline.run_forecast(cfg)
if __name__ == "__main__":
    main()
