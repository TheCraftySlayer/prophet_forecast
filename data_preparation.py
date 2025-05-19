"""Utility functions for preparing data for Prophet models."""
from pathlib import Path
import pandas as pd

from prophet_analysis import (
    load_time_series as _load_time_series,
    verify_date_formats as _verify_date_formats,
    build_flag_series as _build_flag_series,
    prepare_data as _prepare_data,
    create_prophet_holidays as _create_prophet_holidays,
    prepare_prophet_data as _prepare_prophet_data,
)


def load_time_series(*args, **kwargs):
    """Proxy to :func:`prophet_analysis.load_time_series`."""
    return _load_time_series(*args, **kwargs)


def verify_date_formats(*args, **kwargs):
    """Proxy to :func:`prophet_analysis.verify_date_formats`."""
    return _verify_date_formats(*args, **kwargs)


def build_flag_series(*args, **kwargs):
    """Proxy to :func:`prophet_analysis.build_flag_series`."""
    return _build_flag_series(*args, **kwargs)


def prepare_data(*args, **kwargs):
    """Proxy to :func:`prophet_analysis.prepare_data`."""
    return _prepare_data(*args, **kwargs)


def create_prophet_holidays(*args, **kwargs):
    """Proxy to :func:`prophet_analysis.create_prophet_holidays`."""
    return _create_prophet_holidays(*args, **kwargs)


def prepare_prophet_data(*args, **kwargs):
    """Proxy to :func:`prophet_analysis.prepare_prophet_data`."""
    return _prepare_prophet_data(*args, **kwargs)
