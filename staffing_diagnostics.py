"""Compute staffing diagnostics for call volume forecasts."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Tuple


DEFAULT_STAFF = 13
OPEN_HOURS_PER_DAY = 9  # 08:00-16:59


def mean_call_volumes(times: Iterable, counts: Iterable[float]) -> Tuple[float, float]:
    """Return mean daily and hourly call volumes.

    Parameters
    ----------
    times : Iterable
        Sequence of datetime-like objects.
    counts : Iterable[float]
        Sequence of call counts matching ``times``.
    """
    daily_totals: dict = defaultdict(float)
    total = 0.0
    n = 0
    for t, c in zip(times, counts):
        try:
            day = t.date()
        except Exception:
            # Fallback if ``t`` is not datetime-like
            continue
        try:
            val = float(c)
        except Exception:
            continue
        daily_totals[day] += val
        total += val
        n += 1
    mean_daily = sum(daily_totals.values()) / len(daily_totals) if daily_totals else float("nan")
    mean_hourly = total / n if n else float("nan")
    return mean_daily, mean_hourly


def relative_mae(abs_errors: Iterable[float], mean_daily: float) -> float:
    """Return MAE as percentage of ``mean_daily``."""
    total = 0.0
    n = 0
    for e in abs_errors:
        try:
            total += float(e)
            n += 1
        except Exception:
            continue
    if n == 0 or mean_daily != mean_daily or mean_daily == 0:
        return float("nan")
    return (total / n) / mean_daily * 100


def staffing_cost(
    actual: Iterable[float],
    predicted: Iterable[float],
    mean_daily: float,
    understaff_penalty: float,
    overstaff_penalty: float,
    staff_count: int = DEFAULT_STAFF,
    open_hours: int = OPEN_HOURS_PER_DAY,
) -> float:
    """Compute service level cost for forecast errors."""
    calls_per_agent_hour = mean_daily / (staff_count * open_hours) if staff_count and open_hours else 1.0
    cost = 0.0
    n = 0
    for a, p in zip(actual, predicted):
        try:
            a_val = float(a)
            p_val = float(p)
        except Exception:
            continue
        error = p_val - a_val
        if error >= 0:
            over_hours = error / calls_per_agent_hour
            cost += over_hours * overstaff_penalty
        else:
            cost += (-error) * understaff_penalty
        n += 1
    return cost / n if n else float("nan")
