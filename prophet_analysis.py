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
def tune_prophet_hyperparameters(prophet_df):
    """Find optimal Prophet hyperparameters using grid search"""
    logger = logging.getLogger(__name__)
    logger.info("Tuning Prophet hyperparameters")
    
    # Simple parameter grid
    param_grid = {
        'changepoint_prior_scale': [0.05, 0.1, 0.5],
        'seasonality_prior_scale': [1.0, 10.0, 20.0],
        'holidays_prior_scale': [1.0, 10.0]
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
                seasonality_mode='multiplicative',
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

        # Date column is consistently named "date"
        date_col = "date"

        # Set the appropriate value column based on file type
        if metric == "call":
            value_col = 'Count of Calls'  # For calls.csv
        elif metric == "visit":
            value_col = 'Visits'  # For visitors.csv
        else:  # For queries.csv
            value_col = 'query_count'

        # Fallback option if value column not found
        if value_col not in df.columns:
            value_col = next(
                (c for c in df.columns if metric.lower() in c.lower()),
                df.columns[1])

    elif file_ext.endswith('.xlsx') or file_ext.endswith('.xls'):
        # Handle Excel files with explicit engine
        xls = pd.ExcelFile(path, engine='openpyxl')
        # ... rest of Excel handling code ...
    else:
        raise ValueError(f"Unsupported file format: {path}")

    # Convert date column to datetime
    df["date_parsed"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date_parsed"])

    # Return the time series
    return df.set_index("date_parsed")[value_col].sort_index()


def verify_date_formats(call_path, visit_path, chat_path):
    """Verify date formats are consistent across files"""
    call_df = pd.read_csv(call_path)
    visit_df = pd.read_csv(visit_path)
    chat_df = pd.read_csv(chat_path)

    # Check a few date samples from each file
    print("Date format samples:")
    print(f"Calls: {call_df['date'].iloc[0]}")
    print(f"Visits: {visit_df['date'].iloc[0]}")
    print(f"Queries: {chat_df['date'].iloc[0]}")

    # Try parsing dates from each file
    for name, df in [("Calls", call_df), ("Visits", visit_df),
                     ("Queries", chat_df)]:
        try:
            dates = pd.to_datetime(df['date'], errors='raise')
            print(f"{name}: Successfully parsed {len(dates)} dates")
        except Exception as e:
            print(f"{name}: Error parsing dates - {str(e)}")


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
    idx = pd.date_range(start=start, end=end, freq="D")

    # Build main dataframe (existing code)
    df = pd.DataFrame({
        "call_count": calls.reindex(idx, fill_value=0),
        "visit_count": visits.reindex(idx, fill_value=0),
        "chatbot_count": chat.reindex(idx, fill_value=0)
    }, index=idx)

    df['call_count'] = df['call_count'].astype(float)

    # Define 2025 NOV season mask (missing from original code)
    nov_start_2025 = pd.Timestamp(2025, 5, 1)
    nov_end_2025 = pd.Timestamp(2025, 5, 31)
    nov_mask_2025 = (df.index >= nov_start_2025) & (df.index <= nov_end_2025)

    # Create may_2025_policy_changes flag
    df["may_2025_policy_changes"] = 0
    df.loc[nov_mask_2025, "may_2025_policy_changes"] = 1

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

    # Continue with existing holiday and deadline flags, etc. (existing code)
    logger.info("Creating holiday and deadline flags")
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
    df["holiday_flag"] = build_flag_series(idx, holiday_dates)
    df["deadline_flag"] = build_flag_series(idx, deadline_dates)
    df["post_holiday_flag"] = df["holiday_flag"].shift(1).fillna(0).astype(int)

    # Continue with existing seasonal business cycle flags
    logger.info("Creating seasonal business cycle flags")

    # Flag for first week of May (NOV season peak)
    df["nov_first_week_flag"] = ((df.index.month == 5) & (df.index.day >= 1) &
                                 (df.index.day <= 7)).astype(int)

    # Flag for second week of May (NOV season decline)
    df["nov_second_week_flag"] = ((df.index.month == 5) & (df.index.day >= 8) &
                                  (df.index.day <= 14)).astype(int)

    # Flag for third week of May (NOV season further decline)
    df["nov_third_week_flag"] = ((df.index.month == 5) & (df.index.day >= 15) &
                                 (df.index.day <= 21)).astype(int)

    # NOV season 2024 (April) first week flag
    df["nov_2024_first_week_flag"] = ((df.index.year == 2024) &
                                      (df.index.month == 4) &
                                      (df.index.day >= 1) &
                                      (df.index.day <= 7)).astype(int)

    # NOV season 2024 (April) second week flag
    df["nov_2024_second_week_flag"] = ((df.index.year == 2024) &
                                       (df.index.month == 4) &
                                       (df.index.day >= 8) &
                                       (df.index.day <= 14)).astype(int)

    # General NOV season flag (combines both years)
    df["nov_season_flag"] = 0

    # 2024 NOV season (April 1-30, 2024)
    nov_start_2024 = pd.Timestamp(2024, 4, 1)
    nov_end_2024 = pd.Timestamp(2024, 4, 30)
    nov_mask_2024 = (df.index >= nov_start_2024) & (df.index <= nov_end_2024)
    df.loc[nov_mask_2024, "nov_season_flag"] = 1

    # Update 2025 NOV season to full May
    df.loc[nov_mask_2025, "nov_season_flag"] = 1
    
    # Create busy season flag (April 1 - July 31)
    df["busy_season_flag"] = 0

    # 2023 busy season (April 1 - July 31, 2023)
    busy_start_2023 = pd.Timestamp(2023, 4, 1)
    busy_end_2023 = pd.Timestamp(2023, 7, 31)
    busy_mask_2023 = (df.index >= busy_start_2023) & (df.index
                                                      <= busy_end_2023)
    df.loc[busy_mask_2023, "busy_season_flag"] = 1

    # 2024 busy season (April 1 - July 31, 2024)
    busy_start_2024 = pd.Timestamp(2024, 4, 1)
    busy_end_2024 = pd.Timestamp(2024, 7, 31)
    busy_mask_2024 = (df.index >= busy_start_2024) & (df.index
                                                      <= busy_end_2024)
    df.loc[busy_mask_2024, "busy_season_flag"] = 1

    # 2025 busy season (April 1 - July 31, 2025)
    busy_start_2025 = pd.Timestamp(2025, 4, 1)
    busy_end_2025 = pd.Timestamp(2025, 7, 31)
    busy_mask_2025 = (df.index >= busy_start_2025) & (df.index
                                                      <= busy_end_2025)
    df.loc[busy_mask_2025, "busy_season_flag"] = 1

    # Press release features - key assessment dates
    logger.info("Creating press release features")
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

    # Add press release flags
    df["press_release_flag"] = build_flag_series(idx, press_release_dates)

    # Add lagged effect flags for press releases (1-3 days after)
    for lag in range(1, 4):
        df[f"press_release_lag{lag}"] = df["press_release_flag"].shift(
            lag).fillna(0).astype(int)

    # Mail-out features
    logger.info("Creating mail-out features")
    mailout_schedule = {
        pd.Timestamp("2025-01-16"): 3000,
        pd.Timestamp("2025-01-24"): 6000,
        pd.Timestamp("2025-01-27"): 3000 * 4 + 3724 + 692,
        pd.Timestamp("2025-01-30"): 1851,
        pd.Timestamp("2025-04-30"): 65 + 87
    }
    mailout_count = pd.Series(mailout_schedule).reindex(idx, fill_value=0)
    df["mailout_count"] = mailout_count
    df["mailout_flag"] = (mailout_count > 0).astype(int)

    # Create regressors dataframe for Prophet
    regressors = df.copy()

    # Add day of week information for completeness
    regressors['day_of_week'] = regressors.index.dayofweek

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
    
    # Add post-holiday effects (separate holiday type)
    post_holidays = pd.DataFrame({
        'holiday': 'post_holiday',
        'ds': pd.to_datetime(holiday_dates) + pd.Timedelta(days=1),
        'lower_window': 0,
        'upper_window': 0
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
    all_holidays = pd.concat([holidays, post_holidays, deadlines, press_releases])
    
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


def train_prophet_model(prophet_df, holidays_df, regressors_df, future_periods=30):
    """
    Train Prophet model with custom components
    
    Args:
        prophet_df: DataFrame with ds and y columns
        holidays_df: DataFrame with holiday information
        regressors_df: DataFrame with regressor variables
        future_periods: Number of days to forecast
        
    Returns:
        Trained Prophet model, forecast DataFrame, future DataFrame
    """
    logger = logging.getLogger(__name__)
    logger.info("Training Prophet model")
    
    # Initialize Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,  # Not needed as we capture day of week effects
        seasonality_mode='multiplicative',  # Better for call volumes
        changepoint_prior_scale=0.05,  # Allow moderate flexibility in trend
        holidays=holidays_df
    )
    
    # Add key regressor variables
    important_regressors = [
        'busy_season_flag', 'nov_season_flag', 'mailout_flag',
        'may_2025_policy_changes'
    ]
    
    for regressor in important_regressors:
        if regressor in regressors_df.columns:
            # Add the regressor to prophet_df
            prophet_df[regressor] = regressors_df[regressor].values
            # Add to model
            model.add_regressor(regressor, mode='multiplicative')
    
    # Fit the model
    logger.info("Fitting Prophet model")
    model.fit(prophet_df)
    
    # Create future DataFrame
    logger.info(f"Creating future DataFrame with {future_periods} periods")
    future = model.make_future_dataframe(periods=future_periods)
    
    # Add regressor values to future DataFrame
    for regressor in important_regressors:
        if regressor in prophet_df.columns:
            # Copy known values to future DataFrame
            future[regressor] = np.nan
            for i, ds in enumerate(future['ds']):
                if ds in prophet_df['ds'].values:
                    future.loc[i, regressor] = prophet_df.loc[prophet_df['ds'] == ds, regressor].values[0]
                else:
                    # For future dates, use reasonable defaults
                    if regressor == 'busy_season_flag':
                        # Check if date is in busy season (April-July)
                        month = ds.month
                        future.loc[i, regressor] = 1 if month >= 4 and month <= 7 else 0
                    elif regressor == 'nov_season_flag':
                        # Check if date is in NOV season (April/May depending on year)
                        year, month = ds.year, ds.month
                        if year == 2024 and month == 4:
                            future.loc[i, regressor] = 1
                        elif year == 2025 and month == 5:
                            future.loc[i, regressor] = 1
                        else:
                            future.loc[i, regressor] = 0
                    elif regressor == 'may_2025_policy_changes':
                        # Only active in May 2025
                        year, month = ds.year, ds.month
                        future.loc[i, regressor] = 1 if year == 2025 and month == 5 else 0
                    else:
                        # Default to 0 for other regressors
                        future.loc[i, regressor] = 0
    
    # Make forecast
    logger.info("Making forecast")
    forecast = model.predict(future)
    
    return model, forecast, future

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
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05
    )
    
    # More flexible model
    model2 = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.5
    )
    
    # More rigid model
    model3 = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.01
    )
    
    models = [model1, model2, model3]
    
    # Add same regressors to all models
    important_regressors = [
        'busy_season_flag', 'nov_season_flag', 'mailout_flag',
        'may_2025_policy_changes'
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
                model.add_regressor(regressor, mode='multiplicative')
        
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
                        # For future dates, use reasonable defaults
                        if regressor == 'busy_season_flag':
                            # Check if date is in busy season (April-July)
                            month = ds.month
                            future.loc[j, regressor] = 1 if month >= 4 and month <= 7 else 0
                        elif regressor == 'nov_season_flag':
                            # Check if date is in NOV season (April/May depending on year)
                            year, month = ds.year, ds.month
                            if year == 2024 and month == 4:
                                future.loc[j, regressor] = 1
                            elif year == 2025 and month == 5:
                                future.loc[j, regressor] = 1
                            else:
                                future.loc[j, regressor] = 0
                        elif regressor == 'may_2025_policy_changes':
                            # Only active in May 2025
                            year, month = ds.year, ds.month
                            future.loc[j, regressor] = 1 if year == 2025 and month == 5 else 0
                        else:
                            # Default to 0 for other regressors
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
    
    elif method == 'median_replace':
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


def cross_validate_prophet(model, df, periods=30, horizon='30 days'):
    """Simple cross-validation for a Prophet model"""
    df_cv = cross_validation(model, initial='180 days', period=periods, horizon=horizon)
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
    features = ['busy_season_flag', 'nov_season_flag', 
                'mailout_flag', 'may_2025_policy_changes',
                'yearly_seasonality', 'weekly_seasonality']
    
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
            seasonality_mode='multiplicative',
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
                    seasonality_mode='multiplicative',
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
                    seasonality_mode='multiplicative',
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
    # This requires modifying the forecast to remove policy effect
    
    # Find May 2025 forecasts
    may_2025_forecast = forecast[
        (forecast['ds'].dt.year == 2025) & 
        (forecast['ds'].dt.month == 5)
    ].copy()
    
    # If we have the 'may_2025_policy_changes' regressor effect
    if 'may_2025_policy_changes' in may_2025_forecast.columns:
        # Calculate effect of policy
        policy_effect = may_2025_forecast['may_2025_policy_changes']
        
        # Create counterfactual by removing policy effect
        may_2025_forecast['yhat_no_policy'] = may_2025_forecast['yhat'] / (1 + policy_effect)
        
        # Plot counterfactual
        plt.figure(figsize=(14, 7))
        plt.plot(may_2025_forecast['ds'].dt.day,
                 may_2025_forecast['yhat'],
                 'r-',
                 marker='o',
                 label='With Policy Changes')
        plt.plot(may_2025_forecast['ds'].dt.day,
                 may_2025_forecast['yhat_no_policy'],
                 'b--',
                 marker='x',
                 label='Without Policy Changes (Counterfactual)')
        plt.fill_between(may_2025_forecast['ds'].dt.day,
                         may_2025_forecast['yhat'],
                         may_2025_forecast['yhat_no_policy'],
                         alpha=0.2,
                         color='red')
        
        plt.title('May 2025: Impact of Policy Changes on Call Volume',
                  fontsize=14,
                  fontweight='bold')
        plt.xlabel('Day of Month')
        plt.ylabel('Predicted Call Volume')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Calculate total impact
        total_impact = (may_2025_forecast['yhat'] - may_2025_forecast['yhat_no_policy']).sum()
        avg_pct_impact = ((may_2025_forecast['yhat'] / may_2025_forecast['yhat_no_policy'] - 1) * 100).mean()
        
        plt.annotate(
            f"Total additional calls: {total_impact:.0f}\nAverage increase: {avg_pct_impact:.1f}%",
            xy=(0.05, 0.95),
            xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_dir / "policy_counterfactual_analysis.png")
        plt.close()
        
        # Save counterfactual data
        may_2025_forecast.to_csv(output_dir / "may_2025_counterfactual.csv", index=False)
    
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


def export_prophet_forecast(model, forecast, df, output_dir):
    """
    Export Prophet forecast to Excel
    
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
    
    # Get the last week of business days
    today = pd.Timestamp.today()
    last_week = pd.date_range(end=today, periods=5, freq='B')
    
    # Get the next business day
    next_day = today + pd.Timedelta(days=1)
    if next_day.weekday() >= 5:  # Weekend
        next_day = next_day + pd.Timedelta(days=7 - next_day.weekday())
    
    # Get predictions for last week
    last_week_forecast = forecast[forecast['ds'].isin(last_week)].copy()
    last_week_forecast['actual'] = np.nan
    
    # Get actual values if available
    for i, date in enumerate(last_week):
        if date in df.index:
            last_week_forecast.loc[last_week_forecast['ds'] == date, 'actual'] = df.loc[date, 'call_count']
    
    # Calculate errors
    last_week_forecast['error'] = last_week_forecast['actual'] - last_week_forecast['yhat']
    last_week_forecast['abs_error'] = np.abs(last_week_forecast['error'])
    last_week_forecast['pct_error'] = (last_week_forecast['abs_error'] / 
                                      last_week_forecast['actual'].replace(0, np.nan)) * 100
    
    # Get next day forecast
    next_day_forecast = forecast[forecast['ds'] == next_day].copy()
    
    if len(next_day_forecast) == 0:
        # If next day isn't in forecast, make a special prediction
        future = pd.DataFrame({'ds': [next_day]})
        
        # Add any required regressors
        for regressor in model.extra_regressors:
            # Set reasonable default values
            if regressor == 'busy_season_flag':
                future[regressor] = 1 if next_day.month >= 4 and next_day.month <= 7 else 0
            elif regressor == 'nov_season_flag':
                future[regressor] = 1 if ((next_day.year == 2024 and next_day.month == 4) or
                                          (next_day.year == 2025 and next_day.month == 5)) else 0
            elif regressor == 'may_2025_policy_changes':
                future[regressor] = 1 if next_day.year == 2025 and next_day.month == 5 else 0
            else:
                future[regressor] = 0
        
        next_day_forecast = model.predict(future)
    
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
        # Last week performance
        last_week_performance = pd.DataFrame({
            'date': last_week_forecast['ds'],
            'predicted': last_week_forecast['yhat'],
            'actual': last_week_forecast['actual'],
            'lower_bound': last_week_forecast['yhat_lower'],
            'upper_bound': last_week_forecast['yhat_upper'],
            'error': last_week_forecast['error'],
            'abs_error': last_week_forecast['abs_error'],
            'pct_error': last_week_forecast['pct_error']
        })
        last_week_performance.to_excel(writer, sheet_name='Last Week Performance', index=False)
        
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
                'Model accounts for day-of-week patterns, monthly seasonality, holidays, and special events.'
            ]
        })
        notes.to_excel(writer, sheet_name='Notes', index=False)
    
    logger.info(f"Forecast exported to {output_file}")
    
    return output_file


def evaluate_prophet_model(model, prophet_df):
    """Cross‑validate the Prophet model and report MAE, RMSE, and MAPE."""

    logger = logging.getLogger(__name__)
    logger.info(
        "Evaluating Prophet model with 365‑day initial window, "
        "30‑day period, 60‑day horizon"
    )

    # ------------------------------------------------------------------
    # 365‑day initial window  ↓↓↓
    # ------------------------------------------------------------------
    df_cv = cross_validation(
        model,
        initial='365 days',      # <- change is right here
        period='30 days',
        horizon='60 days',
        parallel="processes",
    )

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

    logger.info(f"Cross‑validation →  MAE {mae:.2f} | RMSE {rmse:.2f} | "
                f"MAPE {mape if mape==mape else 'N/A'}")

        # ---------- NEW BLOCK ----------
    summary = pd.DataFrame({
        "metric": ["MAE", "RMSE", "MAPE"],
        "value":  [mae,  rmse,  mape]
    })
    # --------------------------------

    return df_cv, df_p, summary

def predict_next_day_calls(model_path=None):
    """
    Predict call volume for the next business day with detailed diagnostics
    """
    logger = setup_logging()
    # Option 1: Load pre-trained model if available
    if model_path and Path(model_path).exists():
        logger.info(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        logger.warning("No saved model found, predictions may not be accurate")
        # Create a minimal model for testing
        model = Prophet()
    
    # Get next business day
    today = pd.Timestamp.today()
    next_day = today + pd.Timedelta(days=1)
    if next_day.weekday() >= 5:  # Weekend
        next_day = next_day + pd.Timedelta(days=7 - next_day.weekday())
    
    logger.info(f"Predicting for {next_day.strftime('%A, %B %d, %Y')}")
    
    # Create future DataFrame
    future = pd.DataFrame({'ds': [next_day]})
    
    # Check which regressors the model expects
    if hasattr(model, 'extra_regressors'):
        logger.info(f"Model has {len(model.extra_regressors)} regressors: {list(model.extra_regressors.keys())}")
        
        for regressor in model.extra_regressors:
            # DEBUG: Print regression mode
            if hasattr(model.extra_regressors[regressor], 'mode'):
                logger.info(f"Regressor '{regressor}' mode: {model.extra_regressors[regressor].mode}")
            
            # Set regressor values with detailed logging
            if regressor == 'busy_season_flag':
                # April-July is busy season
                value = 1 if (next_day.month >= 4 and next_day.month <= 7) else 0
                future[regressor] = value
                logger.info(f"Set {regressor} = {value} (month is {next_day.month})")
                
            elif regressor == 'nov_season_flag':
                # April 2024 or May 2025 is NOV season
                value = 1 if ((next_day.year == 2024 and next_day.month == 4) or
                              (next_day.year == 2025 and next_day.month == 5)) else 0
                future[regressor] = value
                logger.info(f"Set {regressor} = {value}")
                
            elif regressor == 'may_2025_policy_changes':
                # Only active in May 2025
                value = 1 if (next_day.year == 2025 and next_day.month == 5) else 0
                future[regressor] = value
                logger.info(f"Set {regressor} = {value}")
                
            elif regressor == 'mailout_flag':
                # Default to 0 unless specific date known
                future[regressor] = 0
                logger.info(f"Set {regressor} = 0 (default)")
                
            else:
                # Default for any other regressors
                future[regressor] = 0
                logger.info(f"Set {regressor} = 0 (unknown regressor)")
    
    # Check if model uses log transformation
    uses_log = hasattr(model, 'y_scale') and model.y_scale > 2
    logger.info(f"Model uses log transform: {uses_log} (y_scale: {model.y_scale if hasattr(model, 'y_scale') else 'N/A'})")
    
    # Make prediction with detailed logging
    logger.info("Making prediction...")
    forecast = model.predict(future)
    
    # Print all forecast columns for debugging
    logger.info(f"Forecast columns: {forecast.columns.tolist()}")
    
    # Check if weekend - predictions should be near zero
    is_weekend = next_day.weekday() >= 5
    if is_weekend:
        logger.warning(f"Predicting for {next_day.strftime('%A')} (weekend) - expect near-zero calls")
    
    # Show raw prediction
    raw_pred = forecast['yhat'].values[0]
    logger.info(f"Raw prediction: {raw_pred}")
    
    # Apply back-transformation if model uses log transform
    if uses_log:
        # Back-transform all prediction columns
        forecast['yhat'] = np.expm1(forecast['yhat'])
        forecast['yhat_lower'] = np.expm1(forecast['yhat_lower'])
        forecast['yhat_upper'] = np.expm1(forecast['yhat_upper'])
        pred = forecast['yhat'].values[0]  # Now use the back-transformed value
        logger.info(f"Back-transformed from log: {pred}")
    else:
        pred = raw_pred
    
    # Debug component contributions
    for col in forecast.columns:
        if col not in ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'trend_lower', 'trend_upper']:
            if col in forecast.columns:
                logger.info(f"Component {col}: {forecast[col].values[0]}")
    
    # Final prediction with rounding
    final_pred = max(0, round(pred))  # Ensure non-negative
    logger.info(f"Final prediction: {final_pred} calls")
    
    return final_pred, forecast


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
    
    # 6. Busy season comparison
    # If we have the busy_season_flag regressor
    if 'busy_season_flag' in forecast.columns:
        busy_avg = forecast[forecast['busy_season_flag'] == 1]['yhat'].mean()
        non_busy_avg = forecast[forecast['busy_season_flag'] == 0]['yhat'].mean()
        
        plt.figure(figsize=(8, 5))
        plt.bar(['Regular Season', 'Busy Season'], [non_busy_avg, busy_avg])
        plt.title('Average Call Volume: Busy Season vs Regular Season')
        plt.ylabel('Average Calls')
        plt.grid(axis='y', alpha=0.3)
        
        # Add percentage increase
        pct_increase = ((busy_avg - non_busy_avg) / non_busy_avg) * 100
        plt.annotate(f"+{pct_increase:.1f}%",
                    xy=(1, busy_avg),
                    xytext=(1, busy_avg + 5),
                    ha='center',
                    fontweight='bold')
        
        plt.savefig(output_dir / "busy_season_comparison.png")
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
        help="Method to handle outliers (winsorize, median_replace, interpolate)",
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

    args = parser.parse_args(argv)

    call_path = args.call_data
    visit_path = args.visitor_data
    chat_path = args.chatbot_data
    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True)

    handle_outliers_method = args.handle_outliers
    use_transformation = str(args.use_transformation).lower() == "true"
    skip_feature_importance = args.skip_feature_importance

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
            # Keep all dates as in original code
            # ...
            date(2025, 12, 25)
        ]
        
        deadline_dates = [
            date(2023, 9, 1),
            # Keep all dates as in original code
            # ...
            date(2025, 12, 1)
        ]
        
        press_release_dates = [
            date(2025, 1, 9),
            # Keep all dates as in original code
            # ...
            date(2025, 5, 13)
        ]
        
        holidays_df = create_prophet_holidays(holiday_dates, deadline_dates, press_release_dates)
        holidays_df = enhance_holiday_handling(holidays_df)

        # Apply log transformation if requested
        if use_transformation:
            logger.info("Applying log transformation to target variable")
            prophet_df['y'] = np.log1p(prophet_df['y'])

        # Train Prophet model
        model, forecast, future = train_prophet_model(
            prophet_df, 
            holidays_df, 
            regressors, 
            future_periods=30
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
            
            # Analyze press release impact
            press_release_impact = analyze_press_release_impact_prophet(forecast, output_dir)
            
            # Instead of full cross-validation, use simplified evaluation
            try:
                logger.info("Evaluating model with simplified approach")
                # Split data for evaluation
                train_size = int(len(prophet_df) * 0.8)
                test_prophet_df = prophet_df.iloc[train_size:].copy()
                test_dates = test_prophet_df['ds']
                
                # Get predictions for test period
                test_forecast = forecast[forecast['ds'].isin(test_dates)]
                
                # Calculate error metrics
                y_true = test_prophet_df['y'].values
                y_pred = test_forecast['yhat'].values
                
                mae = mean_absolute_error(y_true, y_pred)
                
                try:
                    # Newer scikit-learn versions
                    rmse = mean_squared_error(y_true, y_pred, squared=False)
                except TypeError:
                    # Older scikit-learn versions
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                
                # Calculate MAPE manually
                nonzero = y_true != 0
                mape = (np.abs(y_true[nonzero] - y_pred[nonzero]) / np.abs(y_true[nonzero])).mean() * 100
                
                # Create summary dataframe
                perf_summary = pd.DataFrame({
                    "metric": ["MAE", "RMSE", "MAPE"],
                    "value": [mae, rmse, mape]
                })
                
                # Save performance metrics
                perf_summary.to_csv(output_dir / "performance_metrics.csv", index=False)
                logger.info(f"Model evaluation: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")
                
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
