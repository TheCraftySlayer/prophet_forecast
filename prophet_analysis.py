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
import pandas as pd
import numpy as np
import itertools
from datetime import date, datetime
import matplotlib.pyplot as plt
import logging
import argparse
import sys
import os
from functools import lru_cache
from pathlib import Path
import glob
import pickle
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import acorr_ljungbox
from pandas.tseries.holiday import USFederalHolidayCalendar

# Import Prophet
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric

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

# Import pandas with openpyxl dependency check
try:
    import openpyxl
except ImportError:
    raise ImportError(
        "Missing optional dependency 'openpyxl'. Install via pip install openpyxl"
    )


def winsorize_series(series, limit=3):
    """Symmetric winsorization using mean ± limit*std."""
    mean = series.mean()
    std = series.std()
    lower = mean - limit * std
    upper = mean + limit * std
    return series.clip(lower, upper)
def tune_prophet_hyperparameters(prophet_df):
    """Find optimal Prophet hyperparameters using grid search"""
    logger = logging.getLogger(__name__)
    logger.info("Tuning Prophet hyperparameters")
    
    # Expanded parameter grid for broader search
    param_grid = {
        'changepoint_prior_scale': [1.5, 2.0, 3.0],
        'seasonality_prior_scale': [1.0, 10.0, 20.0, 30.0],
        'holidays_prior_scale': [1.0, 5.0, 10.0]
    }
    
    # Generate all combinations
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    
    # Storage for results
    rmses = []
    
    # Use cross-validation to evaluate
    for i, params in enumerate(all_params):
        logger.info(f"Testing hyperparameter combination {i+1}/{len(all_params)}")
        
        try:
            # Create train/test split - hold out last 30 days
            train = prophet_df[prophet_df['ds'] < prophet_df['ds'].max() - pd.Timedelta(days=30)]
            test = prophet_df[prophet_df['ds'] >= prophet_df['ds'].max() - pd.Timedelta(days=30)]
            
            # Create a new model instance with these parameters
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='additive',
                **params
            )
            
            # Fit on training data only
            m.fit(train)
            
            # Predict on held-out period
            future = m.make_future_dataframe(periods=30)
            forecast = m.predict(future)
            
            # Evaluate only on held-out period
            y_true = test['y'].values
            y_pred = forecast.tail(len(test))['yhat'].values
            
            # Handle different scikit-learn versions
            try:
                # Newer scikit-learn versions
                rmse = mean_squared_error(y_true, y_pred, squared=False)
            except TypeError:
                # Older scikit-learn versions
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            rmses.append(rmse)
        except Exception as e:
            logger.warning(f"Error with hyperparameter combination {params}: {str(e)}")
            rmses.append(float('inf'))  # Assign worst possible score
    
    # Find best parameters
    if not rmses or all(np.isinf(rmses)):
        logger.warning("All hyperparameter combinations failed, using defaults")
        best_params = {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10.0, 'holidays_prior_scale': 10.0}
    else:
        best_params = all_params[np.argmin(rmses)]
    
    logger.info(f"Best parameters found: {best_params}")
    return best_params
def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s:%(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)
    return logger


@lru_cache(maxsize=None)
def load_time_series(path: Path, metric: str = "call") -> pd.Series:
    """Load a time series from a CSV or Excel file with improved column detection"""
    # Check file extension
    file_ext = str(path).lower()

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

    # Convert date column to datetime
    df["date_parsed"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date_parsed"])

    # Drop weekends before further processing
    df = df[df["date_parsed"].dt.dayofweek < 5]

    # Return the time series
    return df.set_index("date_parsed")[value_col].sort_index()


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

    # Try parsing dates from each file
    for name, df in [("Calls", call_df), ("Visits", visit_df), ("Queries", chat_df)]:
        if df.empty:
            continue
        try:
            dates = pd.to_datetime(df['date'], errors='raise')
            logger.info(f"{name}: Successfully parsed {len(dates)} dates")
        except Exception as e:
            logger.warning(f"{name}: Error parsing dates - {str(e)}")


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


def prepare_data(call_path,
                 visit_path,
                 chat_path,
                 cleaned_calls=None,
                 scale_features=True):
    """
    Prepare time series data with features for forecasting including May 2025 policy changes
    """

    # Initialize logger first - moved to the beginning of the function
    logger = logging.getLogger(__name__)

    verify_date_formats(call_path, visit_path, chat_path)

    # Check for large date gaps
    calls_dates = load_time_series(call_path, metric="call").index
    visits_dates = load_time_series(visit_path, metric="visit").index
    chat_dates = pd.to_datetime(pd.read_csv(chat_path)['date']).dropna()

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
            errors="coerce").dropna().dt.normalize().value_counts().sort_index())

    # Create unified date range
    start = min(calls.index.min(), visits.index.min(), chat.index.min())
    end = max(calls.index.max(), visits.index.max(), chat.index.max())
    logger.info(f"Creating unified date range from {start} to {end}")
    idx = pd.date_range(start=start, end=end, freq="B")

    # Build main dataframe
    df = pd.DataFrame({
        "call_count": calls.reindex(idx),
        "visit_count": visits.reindex(idx, fill_value=0),
        "chatbot_count": chat.reindex(idx, fill_value=0)
    }, index=idx)

    df['missing_flag'] = df['call_count'].isna().astype(int)
    df['call_count'] = df['call_count'].ffill(limit=1)
    df['call_count'] = df['call_count'].interpolate()
    df['call_count'] = df['call_count'].astype(float)

    # Zero out weekends and holidays
    holiday_cal = USFederalHolidayCalendar()
    holiday_dates = holiday_cal.holidays(start=idx.min(), end=idx.max())
    closed_mask = (df.index.dayofweek >= 5) | (df.index.isin(holiday_dates))
    df.loc[closed_mask, 'call_count'] = 0

    # Recalculate weekday means after zeroing
    weekday_means = df.loc[df.index.dayofweek < 5, 'call_count'].groupby(df.index.dayofweek).mean()
    if not weekday_means.empty:
        monday_spike = weekday_means.get(0, 0) - weekday_means.drop(0).mean()
        logger.info(f"Monday spike magnitude: {monday_spike:.1f}")

    # Flag events before outlier handling
    deadline_dates = pd.date_range(start=idx.min(), end=idx.max(), freq='MS')
    df['holiday_flag'] = df.index.isin(holiday_dates).astype(int)
    df['deadline_flag'] = 0
    for d in deadline_dates:
        window = pd.date_range(d - pd.Timedelta(days=5), d + pd.Timedelta(days=1))
        df.loc[df.index.isin(window), 'deadline_flag'] = 1

    # Flag for post-policy period starting May 2025
    policy_start = pd.Timestamp('2025-05-01')
    df['post_policy'] = (df.index >= policy_start).astype(int)

    event_mask = (df['holiday_flag'] != 0) | (df['deadline_flag'] != 0)
    z = (df['call_count'] - df['call_count'].mean()) / df['call_count'].std()
    df['outlier_flag'] = ((z.abs() > 3) & ~event_mask).astype(int)
    df.loc[df['outlier_flag'] == 1, 'call_count'] = winsorize_series(df.loc[df['outlier_flag'] == 1, 'call_count'])

    # Clip extreme call counts at the 95th percentile
    clip_val = df['call_count'].quantile(0.95)
    df['call_count'] = df['call_count'].clip(upper=clip_val)

    df['visit_log1p'] = np.log1p(df['visit_count'])
    df['chatbot_log1p'] = np.log1p(df['chatbot_count'])
    df['is_weekday'] = (df.index.dayofweek < 5).astype(int)
    df['visit_weekday_interaction'] = df['visit_log1p'] * df['is_weekday']



    # Feature engineering: lags and rolling stats for potential use as regressors
    logger.info("Creating lag and rolling features")
    for lag in [1, 3, 7]:
        df[f"call_lag{lag}"] = df["call_count"].shift(lag).fillna(0).astype(
            float)
    df["call_ma7"] = df["call_count"].rolling(7, min_periods=1).mean()
    df["call_std7"] = df["call_count"].rolling(
        7, min_periods=1).std().fillna(0).astype(float)

    for lag in [1, 3]:
        df[f"visit_lag{lag}"] = df["visit_count"].shift(lag).fillna(0).astype(
            float)
    df["visit_ma7"] = df["visit_count"].rolling(7, min_periods=1).mean()
    df["visit_std7"] = df["visit_count"].rolling(
        7, min_periods=1).std().fillna(0).astype(float)
    df["chatbot_ma3"] = df["chatbot_count"].rolling(3, min_periods=1).mean()







    # Create regressors dataframe for Prophet
    regressors = df.copy()

    # Add day of week information for completeness
    regressors["day_of_week"] = regressors.index.dayofweek

    if scale_features:
        # Standardize numeric regressors (z-score) except binary flags
        for col in regressors.columns:
            if col == "call_count" or col.endswith("_flag") or col == "post_policy":
                continue
            if np.issubdtype(regressors[col].dtype, np.number):
                mean = regressors[col].mean()
                std = regressors[col].std()
                if std != 0:
                    regressors[col] = (regressors[col] - mean) / std

    return df, regressors

def create_prophet_holidays(holiday_dates, deadline_dates, press_release_dates):
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
    
    # Create press release DataFrame
    press_releases = pd.DataFrame({
        'holiday': 'press_release',
        'ds': pd.to_datetime(press_release_dates),
        'lower_window': 0,
        'upper_window': 3  # Effect may last for 3 days after press release
    })
    
    # Combine all holiday DataFrames
    all_holidays = pd.concat([holidays, deadlines, press_releases])
    
    return all_holidays

def enhance_holiday_handling(holidays_df):
    """Improve holiday effects modeling"""
    
    # 1. Add bridging days (e.g., days between a holiday and weekend)
    holiday_dates = pd.to_datetime(holidays_df[holidays_df['holiday'] == 'holiday']['ds'])
    bridge_days = []
    
    for date in holiday_dates:
        # If holiday is on Tuesday, Monday might be taken off
        if date.dayofweek == 1:  # Tuesday
            bridge_days.append(date - pd.Timedelta(days=1))
        # If holiday is on Thursday, Friday might be taken off
        elif date.dayofweek == 3:  # Thursday
            bridge_days.append(date + pd.Timedelta(days=1))
    
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
    # Create Prophet DataFrame
    prophet_df = df.reset_index().rename(columns={'index': 'ds', 'call_count': 'y'})
    
    return prophet_df


def train_prophet_model(prophet_df, holidays_df, regressors_df, future_periods=30, model_params=None):
    """
    Train Prophet model with custom components
    
    Args:
        prophet_df: DataFrame with ds and y columns
        holidays_df: DataFrame with holiday information
        regressors_df: DataFrame with regressor variables
        future_periods: Number of days to forecast
        model_params: Optional dictionary of parameters to pass to Prophet
        
    Returns:
        Trained Prophet model, forecast DataFrame, future DataFrame
    """
    logger = logging.getLogger(__name__)
    logger.info("Training Prophet model")
    
    # Initialize Prophet model with optional tuned parameters
    max_calls = prophet_df['y'].max()

    default_params = {
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': False,
        'seasonality_mode': 'additive',
        'n_changepoints': 25,
        'changepoint_prior_scale': 1.5,
        'changepoints': [pd.Timestamp('2025-05-01')],
        'seasonality_prior_scale': 0.05,
        'holidays_prior_scale': 20,
        'holidays': holidays_df,
        'mcmc_samples': 0,
        'growth': 'linear'
    }

    if model_params:
        default_params.update(model_params)


    prophet_df['cap'] = max_calls * 1.1
    prophet_df['floor'] = 0

    model = Prophet(**default_params)
    model.add_seasonality('yearly', period=365.25, fourier_order=8, mode='additive', prior_scale=0.05)
    

    # Restrict regressors to mitigate collinearity
    # Only use visitors, queries and one policy indicator
    important_regressors = [
        'visit_log1p',
        'chatbot_log1p',
        'post_policy',
    ]
    
    for regressor in important_regressors:
        if regressor in regressors_df.columns:
            prophet_df[regressor] = regressors_df[regressor].values
        if regressor == 'post_policy':
            model.add_regressor(regressor, mode='additive', prior_scale=10)
        else:
            model.add_regressor(regressor, mode='additive')
    
    # Fit the model
    logger.info("Fitting Prophet model")
    model.fit(prophet_df)
    
    # Create future DataFrame
    logger.info(f"Creating future DataFrame with {future_periods} periods")
    future = model.make_future_dataframe(periods=future_periods)
    future['cap'] = max_calls * 1.1
    future['floor'] = 0

    # Create an empty regressor frame indexed by the forecast dates
    future_regs = pd.DataFrame(index=future['ds'])

    # Required regressors only
    future_regs['post_policy'] = (future['ds'] >= pd.Timestamp('2025-05-01')).astype(int)
    future_regs['visit_log1p'] = np.log1p(0)
    future_regs['chatbot_log1p'] = np.log1p(0)

    # Overlay known regressor values from historical data
    known = regressors_df.reindex(future['ds'])
    for col in future_regs.columns:
        if col in known.columns:
            mask = known[col].notna()
            future_regs.loc[mask, col] = known.loc[mask, col]

    # Merge regressor values back into the future dataframe on the date column
    future = future.merge(future_regs, left_on="ds", right_index=True, how="left")

    # Guard against any unexpected NaNs in the merged regressor columns
    for col in future_regs.columns:
        if col in future.columns:
            if future[col].isna().any():
                logger.warning(
                    f"Found {future[col].isna().sum()} NaN values in {col} after merge, filling with 0"
                )
                future[col] = future[col].fillna(0)
    
    # Make forecast
    logger.info("Making forecast")
    forecast = model.predict(future)
    forecast[['yhat', 'yhat_lower', 'yhat_upper']] = forecast[['yhat', 'yhat_lower', 'yhat_upper']].clip(lower=0)

    # Enforce zero forecast on weekends and holidays
    closed_mask = (forecast['ds'].dt.dayofweek >= 5) | (forecast['ds'].isin(holidays_df['ds']))
    forecast.loc[closed_mask, ['yhat', 'yhat_lower', 'yhat_upper']] = 0

    _check_forecast_sanity(forecast)

    return model, forecast, future

def _check_forecast_sanity(forecast: pd.DataFrame) -> None:
    """Run simple sanity checks on forecast outputs."""
    logger = logging.getLogger(__name__)
    if 'weekly' in forecast.columns:
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

def create_simple_ensemble(prophet_df, holidays_df, regressors_df):
    """Create a simple ensemble of multiple Prophet models"""
    logger = logging.getLogger(__name__)
    logger.info("Creating ensemble of Prophet models")
    
    # Create multiple Prophet models with different hyperparameters
    models = []
    
    # Base model
    model1 = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=1.5
    )
    
    # More flexible model
    model2 = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=2.0
    )
    
    # More rigid model
    model3 = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=3.0
    )
    
    models = [model1, model2, model3]
    
    # Add same regressors to all models
    # Use the reduced regressor set to avoid collinearity
    important_regressors = [
        'visit_log1p',
        'chatbot_log1p',
        'post_policy'
    ]
    
    # Add regressors to each model and fit them
    forecasts = []
    for i, model in enumerate(models):
        logger.info(f"Training ensemble model {i+1}/{len(models)}")
        model_prophet_df = prophet_df.copy()
        
        for regressor in important_regressors:
            if regressor in regressors_df.columns:
                # Add the regressor to model_prophet_df
                model_prophet_df[regressor] = regressors_df[regressor].values
                # Add to model
                # Use additive mode for stability
                model.add_regressor(regressor, mode='additive')
        
        model.fit(model_prophet_df)
        future = model.make_future_dataframe(periods=30)
        
        # Add regressor values to future DataFrame - FIXED CODE HERE
        for regressor in important_regressors:
            if regressor in model_prophet_df.columns:
                # Initialize the column with zeros first (no NaNs)
                future[regressor] = 0
                
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
    
    # Calculate lower and upper bounds
    lower_bounds = [f['yhat_lower'] for f in forecasts]
    upper_bounds = [f['yhat_upper'] for f in forecasts]
    
    # For each row, get the minimum lower bound and maximum upper bound
    ensemble_forecast['yhat_lower'] = pd.concat(lower_bounds, axis=1).min(axis=1)
    ensemble_forecast['yhat_upper'] = pd.concat(upper_bounds, axis=1).max(axis=1)

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
    prophet_df = prophet_df.merge(
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
        on='ds', 
        how='left'
    )
    
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
    
    # Analyze yearly pattern
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


def cross_validate_prophet(model, df, periods=30, horizon='60 days'):
    """Simple cross-validation for a Prophet model using a rolling origin."""
    df_cv = cross_validation(
        model,
        initial='180 days',
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
    
    # Create versions of the data with one feature removed at a time
    features = [
        'holiday_flag',
        'deadline_flag',
        'visit_count',
        'chatbot_count',
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
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='additive',
            changepoint_prior_scale=model.changepoint_prior_scale
        )
        
        # Add regressors to the base model
        for feature in features:
            if feature.endswith('_flag') and feature in prophet_df.columns:
                model_copy.add_regressor(feature, mode='multiplicative')
        
        model_copy.fit(train_df)
        future = model_copy.make_future_dataframe(periods=future_periods)
        
        # Add regressor values to future DataFrame
        for feature in features:
            if feature.endswith('_flag') and feature in prophet_df.columns:
                future[feature] = prophet_df[feature].values[-future_periods:]
        
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
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    seasonality_mode='additive',
                    changepoint_prior_scale=model.changepoint_prior_scale
                )
                
                # Add remaining regressors
                for other_feature in [f for f in features if f.endswith('_flag') and f != feature]:
                    if other_feature in test_df.columns:
                        test_model.add_regressor(other_feature, mode='multiplicative')
                
                test_model.fit(test_df)
                
                if quick_mode:
                    # Use the simplified validation approach
                    train_size = int(len(test_df) * 0.8)
                    train_df = test_df.iloc[:train_size].copy()
                    test_subset = test_df.iloc[train_size:].copy()
                    future_periods = len(test_subset)
                    
                    test_model.fit(train_df)
                    future = test_model.make_future_dataframe(periods=future_periods)
                    
                    # Add regressor values to future DataFrame
                    for other_feature in [f for f in features if f.endswith('_flag')]:
                        if other_feature in test_df.columns:
                            future[other_feature] = test_df[other_feature].values[-future_periods:]
                    
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
                test_model = Prophet(
                    yearly_seasonality=(feature != 'yearly_seasonality'),
                    weekly_seasonality=(feature != 'weekly_seasonality'),
                    daily_seasonality=False,
                    seasonality_mode='additive',
                    changepoint_prior_scale=model.changepoint_prior_scale
                )
                
                # Add regressors
                for other_feature in [f for f in features if f.endswith('_flag')]:
                    if other_feature in prophet_df.columns:
                        test_model.add_regressor(other_feature, mode='multiplicative')
                
                if quick_mode:
                    # Use the simplified validation approach
                    train_size = int(len(prophet_df) * 0.8)
                    train_df = prophet_df.iloc[:train_size].copy()
                    test_subset = prophet_df.iloc[train_size:].copy()
                    future_periods = len(test_subset)
                    
                    test_model.fit(train_df)
                    future = test_model.make_future_dataframe(periods=future_periods)
                    
                    # Add regressor values to future DataFrame
                    for other_feature in [f for f in features if f.endswith('_flag')]:
                        if other_feature in prophet_df.columns:
                            future[other_feature] = prophet_df[other_feature].values[-future_periods:]
                    
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


def compute_naive_baseline(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a naive forecast for the last 14 days and metrics.

    The naive approach simply uses the previous day's call count as the
    prediction for the current day. This mirrors the lightweight logic in
    ``naive_forecast.py`` and avoids the need for additional packages.

    Parameters
    ----------
    df : DataFrame
        Source data with a ``call_count`` column and a DatetimeIndex.

    Returns
    -------
    Tuple of ``(forecast_df, metrics_df)`` where ``forecast_df`` contains the
    date, predicted call count, actual call count and error columns. The
    ``metrics_df`` provides MAE, RMSE and MAPE aggregated over the period.
    """

    df_sorted = df.sort_index()
    recent = df_sorted["call_count"].iloc[-15:]
    preds = recent.shift(1).dropna()
    actual = recent.iloc[1:]
    dates = actual.index

    result = pd.DataFrame({
        "date": dates,
        "predicted": preds.values,
        "actual": actual.values,
    })
    result["error"] = result["actual"] - result["predicted"]
    result["abs_error"] = result["error"].abs()
    result["pct_error"] = (
        result["abs_error"] / result["actual"].replace(0, np.nan)
    ) * 100

    mae = result["abs_error"].mean()
    rmse = np.sqrt((result["error"] ** 2).mean())
    mape = result["pct_error"].mean()

    metrics = pd.DataFrame({
        "metric": ["MAE", "RMSE", "MAPE"],
        "value": [mae, rmse, mape],
    })

    return result, metrics


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

    baseline_df, metrics = compute_naive_baseline(df)

    csv_path = output_dir / "baseline_forecast.csv"
    baseline_df.to_csv(csv_path, index=False)

    excel_path = output_dir / "baseline_forecast.xlsx"
    input_data = (
        df.sort_index().iloc[-15:].reset_index().rename(columns={"index": "date"})
    )

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        baseline_df.to_excel(writer, sheet_name="Baseline", index=False)
        metrics.to_excel(writer, sheet_name="Metrics", index=False)
        input_data.to_excel(writer, sheet_name="Input Data", index=False)

    return excel_path


def export_prophet_forecast(model, forecast, df, output_dir):
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
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Define output file
    output_file = output_dir / "prophet_call_predictions.xlsx"
    
    # Get the past 14 business days
    today = pd.Timestamp.today()
    recent_days = pd.date_range(end=today, periods=14, freq='B')
    
    # Get the next business day
    next_day = today + pd.Timedelta(days=1)
    if next_day.weekday() >= 5:  # Weekend
        next_day = next_day + pd.Timedelta(days=7 - next_day.weekday())
    
    # Get predictions for the past 14 days
    recent_forecast = forecast[forecast['ds'].isin(recent_days)].copy()
    recent_forecast['actual'] = np.nan
    
    # Get actual values if available
    for i, date in enumerate(recent_days):
        if date in df.index:
            recent_forecast.loc[recent_forecast['ds'] == date, 'actual'] = df.loc[date, 'call_count']
    
    # Calculate errors
    recent_forecast['error'] = recent_forecast['actual'] - recent_forecast['yhat']
    recent_forecast['abs_error'] = np.abs(recent_forecast['error'])
    recent_forecast['pct_error'] = (
        recent_forecast['abs_error'] /
        recent_forecast['actual'].replace(0, np.nan)
    ) * 100
    
    # Get next day forecast
    next_day_forecast = forecast[forecast['ds'] == next_day].copy()
    
    if len(next_day_forecast) == 0:
        # If next day isn't in forecast, make a special prediction
        future = pd.DataFrame({'ds': [next_day]})
        # Include cap and floor for logistic growth models
        if hasattr(model, 'history'):
            if 'cap' in model.history:
                future['cap'] = model.history['cap'].max()
            if 'floor' in model.history:
                future['floor'] = model.history['floor'].min()
        
        # Add any required regressors
        for regressor in model.extra_regressors:
            future[regressor] = 0
        
        next_day_forecast = model.predict(future)
        next_day_forecast[['yhat', 'yhat_lower', 'yhat_upper']] = (
            next_day_forecast[['yhat', 'yhat_lower', 'yhat_upper']].clip(lower=0)
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
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Past 14-day performance
        recent_performance = pd.DataFrame({
            'date': recent_forecast['ds'],
            'predicted': recent_forecast['yhat'],
            'actual': recent_forecast['actual'],
            'lower_bound': recent_forecast['yhat_lower'],
            'upper_bound': recent_forecast['yhat_upper'],
            'error': recent_forecast['error'],
            'abs_error': recent_forecast['abs_error'],
            'pct_error': recent_forecast['pct_error']
        })
        recent_performance.to_excel(writer, sheet_name='Recent 14-Day Performance', index=False)

        # Metrics for Prophet predictions
        prophet_metrics = pd.DataFrame({
            'metric': ['MAE', 'RMSE', 'MAPE'],
            'value': [
                recent_performance['abs_error'].mean(),
                np.sqrt((recent_performance['error'] ** 2).mean()),
                recent_performance['pct_error'].mean()
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
    
    logger.info(f"Forecast exported to {output_file}")
    
    return output_file


def evaluate_prophet_model(model, prophet_df, cv_params=None):
    """Cross‑validate the Prophet model and report MAE, RMSE, and MAPE."""

    if cv_params is None:
        cv_params = {}

    initial = cv_params.get('initial', '365 days')
    period = cv_params.get('period', '30 days')
    horizon = cv_params.get('horizon', '30 days')

    logger = logging.getLogger(__name__)
    logger.info(
        f"Evaluating Prophet model with {initial} initial window, "
        f"{period} period, {horizon} horizon"
    )

    df_cv = cross_validation(
        model,
        initial=initial,
        period=period,
        horizon=horizon,
        parallel="processes",
    )

    # Propagate horizon column if not provided by cross_validation
    if 'horizon' not in df_cv.columns and {'ds', 'cutoff'} <= set(df_cv.columns):
        df_cv['horizon'] = df_cv['ds'] - df_cv['cutoff']

    df_p = performance_metrics(df_cv, rolling_window=1)

    mae  = df_p['mae' ].mean() if 'mae'  in df_p else float('nan')
    rmse = df_p['rmse'].mean() if 'rmse' in df_p else float('nan')

    # Manual fallback for MAPE if it was dropped
    if 'mape' in df_p.columns:
        mape = df_p['mape'].mean()
    else:
        nonzero = df_cv['y'] != 0
        mape = (
            (np.abs(df_cv.loc[nonzero, 'y'] - df_cv.loc[nonzero, 'yhat'])
             / np.abs(df_cv.loc[nonzero, 'y']))
            .mean() * 100
        ) if nonzero.any() else float('nan')

    coverage = (
        ((df_cv['y'] >= df_cv['yhat_lower']) & (df_cv['y'] <= df_cv['yhat_upper'])).mean() * 100
        if {'yhat_lower', 'yhat_upper'} <= set(df_cv.columns) else float('nan')
    )

    logger.info(
        f"Cross‑validation →  MAE {mae:.2f} | RMSE {rmse:.2f} | MAPE {mape if mape==mape else 'N/A'} | "
        f"Coverage {coverage if coverage==coverage else 'N/A'}%"
    )

    residuals = df_cv['y'] - df_cv['yhat']
    lb = acorr_ljungbox(residuals, lags=14, return_df=True)
    ac_flag = (lb['lb_pvalue'] < 0.05) | (lb['lb_stat'] > 0.2)
    autocorr_flag = bool(ac_flag.any())

        # ---------- NEW BLOCK ----------
    summary = pd.DataFrame({
        "metric": ["MAE", "RMSE", "MAPE", "Coverage"],
        "value":  [mae,  rmse,  mape, coverage]
    })

    horizon_rows = []
    for h in [7, 14, 30]:
        mask = df_cv['horizon'] <= pd.Timedelta(days=h)
        sub = df_cv[mask]
        if len(sub) == 0:
            continue
        mae_h = np.mean(np.abs(sub['y'] - sub['yhat']))
        rmse_h = np.sqrt(mean_squared_error(sub['y'], sub['yhat']))
        nonzero = sub['y'] != 0
        mape_h = (
            np.abs(sub.loc[nonzero, 'y'] - sub.loc[nonzero, 'yhat']) /
            np.abs(sub.loc[nonzero, 'y'])
        ).mean() * 100 if nonzero.any() else float('nan')
        horizon_rows.append([h, mae_h, rmse_h, mape_h])
    horizon_table = pd.DataFrame(horizon_rows, columns=['horizon_days','MAE','RMSE','MAPE'])

    diag = pd.DataFrame({
        'lag': lb.index,
        'lb_stat': lb['lb_stat'],
        'lb_pvalue': lb['lb_pvalue']
    })

    if mae > 60 or (mape == mape and mape > 15):
        logger.warning('Model performance below acceptance thresholds.')
    if autocorr_flag:
        logger.warning('Autocorrelation detected in residuals.')

    return df_cv, horizon_table, summary, diag


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
    """Main entry point for the script"""
    logger = setup_logging()
    logger.info("Starting Prophet call volume forecasting script")

    parser = argparse.ArgumentParser(
        description="Run call volume forecasting using Prophet"
    )
    parser.add_argument(
        "call_data",
        type=Path,
        help="CSV/Excel file containing call counts",
    )
    parser.add_argument(
        "visitor_data",
        type=Path,
        help="CSV/Excel file containing visitor counts",
    )
    parser.add_argument(
        "chatbot_data",
        type=Path,
        help="CSV file containing chatbot queries",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=Path("prophet_output"),
        type=Path,
        help="Directory to save results",
    )
    parser.add_argument(
        "--handle-outliers",
        dest="handle_outliers",
        metavar="METHOD",
        help="Method to handle outliers (winsorize, prediction_replace, interpolate)",
    )
    parser.add_argument(
        "--use-transformation",
        metavar="BOOL",
        default="false",
        help="Apply log transformation to target (true/false)",
    )
    parser.add_argument(
        "--skip-feature-importance",
        action="store_true",
        help="Skip feature importance analysis",
    )
    parser.add_argument(
        "--cross-validate",
        action="store_true",
        help="Run full cross-validation after training",
    )

    args = parser.parse_args(argv)

    call_path = args.call_data
    visit_path = args.visitor_data
    chat_path = args.chatbot_data
    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True)

    handle_outliers_method = args.handle_outliers
    use_transformation = str(args.use_transformation).lower() == "true"
    skip_feature_importance = args.skip_feature_importance
    run_cross_validation = args.cross_validate

    # Check if files exist
    for p in [call_path, visit_path, chat_path]:
        if not p.exists():
            logger.error(f"File not found: {p}")
            sys.exit(1)

    try:
        # Load and prepare data
        df, regressors = prepare_data(call_path, visit_path, chat_path,
                                     scale_features=True)
        
        # Convert to Prophet format
        prophet_df = prepare_prophet_data(df)
        
        # Try to tune hyperparameters, but use defaults if it fails
        try:
            best_params = tune_prophet_hyperparameters(prophet_df)
        except Exception as e:
            logger.warning(f"Hyperparameter tuning failed: {str(e)}. Using defaults.")
            best_params = {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10.0, 'holidays_prior_scale': 10.0}
        
        # Create holidays DataFrame
        holiday_dates = [
            date(2023, 1, 2),
            date(2023, 1, 16),
            date(2023, 4, 7),
            date(2023, 5, 29),
            date(2023, 6, 19),
            date(2023, 7, 4),
            date(2023, 9, 4),
            date(2023, 11, 10),
            date(2023, 11, 23),
            date(2023, 11, 24),
            date(2023, 12, 25),
            date(2023, 12, 26),
            date(2024, 1, 1),
            date(2024, 1, 15),
            date(2024, 3, 29),
            date(2024, 5, 27),
            date(2024, 6, 19),
            date(2024, 7, 4),
            date(2024, 10, 14),
            date(2024, 11, 11),
            date(2024, 11, 28),
            date(2024, 11, 29),
            date(2024, 12, 24),
            date(2024, 12, 25),
            date(2025, 1, 1),
            date(2025, 1, 20),
            date(2025, 4, 18),
            date(2025, 5, 26),
            date(2025, 6, 19),
            date(2025, 7, 4),
            date(2025, 9, 1),
            date(2025, 10, 13),
            date(2025, 11, 11),
            date(2025, 11, 27),
            date(2025, 11, 28),
            date(2025, 12, 24),
            date(2025, 12, 25)
        ]

        deadline_dates = [
            date(2023, 9, 1),
            date(2023, 9, 6),
            date(2023, 10, 1),
            date(2023, 11, 1),
            date(2023, 12, 1),
            date(2024, 2, 28),
            date(2024, 4, 1),
            date(2024, 4, 30),
            date(2024, 6, 1),
            date(2024, 6, 15),
            date(2024, 6, 30),
            date(2024, 8, 1),
            date(2024, 9, 1),
            date(2024, 9, 6),
            date(2024, 10, 1),
            date(2024, 11, 1),
            date(2024, 12, 1),
            date(2025, 2, 28),
            date(2025, 4, 1),
            date(2025, 4, 30),
            date(2025, 6, 1),
            date(2025, 6, 15),
            date(2025, 6, 30),
            date(2025, 8, 1),
            date(2025, 9, 1),
            date(2025, 9, 6),
            date(2025, 10, 1),
            date(2025, 11, 1),
            date(2025, 12, 1)
        ]

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
        
        holidays_df = create_prophet_holidays(holiday_dates, deadline_dates, press_release_dates)
        holidays_df = enhance_holiday_handling(holidays_df)

        # Apply log transformation if requested
        if use_transformation:
            logger.info("Applying log transformation to target variable")
            prophet_df['y'] = np.log1p(prophet_df['y'])

        # Train Prophet model using tuned hyperparameters
        model, forecast, future = train_prophet_model(
            prophet_df,
            holidays_df,
            regressors,
            future_periods=30,
            model_params=best_params
        )
        
        # Try to create ensemble model, but continue with single model if it fails
        try:
            logger.info("Attempting to create ensemble forecast")
            ensemble_forecast, ensemble_models = create_simple_ensemble(prophet_df, holidays_df, regressors)
            forecast = ensemble_forecast  # Use ensemble forecast if successful
            logger.info("Using ensemble forecast")
        except Exception as e:
            logger.warning(f"Ensemble model creation failed: {str(e)}. Using single model forecast.")
            # Keep using the forecast from the single model

        # Save Prophet model to disk
        logger.info("Saving Prophet model to disk")
        try:
            with open(output_dir / "prophet_model.pkl", 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            logger.warning(f"Failed to save model: {str(e)}")

        # Analyze feature importance if not skipped
        if not skip_feature_importance:
            try:
                logger.info("Running feature importance analysis (quick mode)")
                feature_impacts = analyze_feature_importance(model, prophet_df, quick_mode=True)
                logger.info(f"Feature importance: {feature_impacts}")
                
                # Save feature importance results
                pd.DataFrame({'Feature': list(feature_impacts.keys()), 
                            'Impact (%)': list(feature_impacts.values())}).to_csv(
                    output_dir / "feature_importance.csv", index=False)
            except Exception as e:
                logger.warning(f"Feature importance analysis failed: {str(e)}")
        else:
            logger.info("Skipping feature importance analysis as requested")
            
        # Back-transform if needed
        if use_transformation:
            logger.info("Back-transforming forecast")
            forecast['yhat'] = np.expm1(forecast['yhat'])
            forecast['yhat_lower'] = np.expm1(forecast['yhat_lower'])
            forecast['yhat_upper'] = np.expm1(forecast['yhat_upper'])
            
            logger.info(f"Forecast range after back-transformation: {forecast['yhat'].min():.2f} to {forecast['yhat'].max():.2f}")

        # Continue with the rest of the analysis functions
        try:
            # Detect outliers
            outlier_df = improve_outlier_detection(df, forecast)

            # Handle outliers if requested
            if handle_outliers_method:
                logger.info(f"Handling outliers using {handle_outliers_method}")
                df_cleaned = handle_outliers_prophet(df, outlier_df, method=handle_outliers_method)
                
                # Update df with cleaned values
                df = df_cleaned
            
            # Analyze model components
            analyze_prophet_components(model, forecast, output_dir)
            
            # Analyze policy changes
            policy_summary = analyze_policy_changes_prophet(df, forecast, output_dir)
            

            if run_cross_validation:
                try:
                    logger.info("Evaluating model with cross-validation")
                    df_cv, horizon_table, summary = evaluate_prophet_model(model, prophet_df)
                    horizon_table.to_csv(output_dir / "cross_validation_metrics.csv", index=False)
                    summary.to_csv(output_dir / "cross_validation_summary.csv", index=False)
                except Exception as e:
                    logger.warning(f"Cross-validation failed: {str(e)}")
            else:
                # Simplified evaluation
                try:
                    logger.info("Evaluating model with simplified approach")
                    train_size = int(len(prophet_df) * 0.8)
                    test_prophet_df = prophet_df.iloc[train_size:].copy()
                    test_dates = test_prophet_df['ds']

                    test_forecast = forecast[forecast['ds'].isin(test_dates)]

                    y_true = test_prophet_df['y'].values
                    y_pred = test_forecast['yhat'].values

                    mae = mean_absolute_error(y_true, y_pred)

                    try:
                        rmse = mean_squared_error(y_true, y_pred, squared=False)
                    except TypeError:
                        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

                    nonzero = y_true != 0
                    mape = (np.abs(y_true[nonzero] - y_pred[nonzero]) / np.abs(y_true[nonzero])).mean() * 100

                    perf_summary = pd.DataFrame({
                        "metric": ["MAE", "RMSE", "MAPE"],
                        "value": [mae, rmse, mape]
                    })

                    perf_summary.to_csv(output_dir / "performance_metrics.csv", index=False)
                    logger.info(
                        f"Model evaluation: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%"
                    )

                except Exception as e:
                    logger.warning(f"Model evaluation failed: {str(e)}")
            
            # Create dashboard
            create_prophet_dashboard(model, forecast, df, output_dir)
            
            # Export forecast to Excel
            export_prophet_forecast(model, forecast, df, output_dir)
            
            logger.info("Prophet analysis completed successfully")
            print(f"Analysis completed successfully. Results saved to {output_dir}")
        except Exception as e:
            logger.error(f"Error during analysis functions: {str(e)}", exc_info=True)
            print(f"Analysis partially completed with errors. Check logs and results in {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
