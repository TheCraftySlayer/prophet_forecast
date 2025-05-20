from datetime import datetime

from ... import DataFrame, DatetimeIndex, to_datetime


class USFederalHolidayCalendar:
    """Very small subset of the real holiday calendar."""

    def holidays(self, start=None, end=None):
        """Return DatetimeIndex of holidays between start and end."""
        try:
            from holidays_calendar import get_holidays_dataframe
        except Exception:  # pragma: no cover - fallback if calendar missing
            return DatetimeIndex([])

        df = get_holidays_dataframe()
        start_dt = to_datetime(start) if start else df['date'].iloc[0]
        end_dt = to_datetime(end) if end else df['date'].iloc[-1]
        mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
        return DatetimeIndex(df[mask]['date'].data)

