# ruff: noqa: E402
from datetime import date
from pathlib import Path

import os
for var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[var] = "1"

import pandas as pd


def get_holidays_dataframe() -> pd.DataFrame:
    """Return DataFrame with county holidays, tax deadlines and press release dates.

    Additional assessor-related events are loaded from ``assessor_events.csv`` if
    present in the same directory and appended to the calendar.
    """
    county_holidays = [
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
        date(2025, 12, 25),
    ]

    tax_deadlines = [
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
        date(2025, 12, 1),
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
        date(2025, 5, 13),
    ]

    notice_mailout_dates = [
        date(2023, 3, 1),
        date(2024, 3, 1),
        date(2025, 3, 1),
    ]

    records = []
    for d in county_holidays:
        records.append({"date": pd.to_datetime(d), "event": "county_holiday"})
    for d in tax_deadlines:
        records.append({"date": pd.to_datetime(d), "event": "tax_deadline"})
    for d in press_release_dates:
        records.append({"date": pd.to_datetime(d), "event": "press_release"})
    for d in notice_mailout_dates:
        records.append({"date": pd.to_datetime(d), "event": "notice_mailout"})

    events_path = Path(__file__).with_name("assessor_events.csv")
    if events_path.exists():
        df_events = pd.read_csv(events_path)
        for row in df_events.itertuples(index=False):
            start = pd.to_datetime(row.start_date)
            end = pd.to_datetime(row.end_date) if isinstance(row.end_date, str) and row.end_date else start
            dates = pd.date_range(start, end)
            for d in dates:
                records.append({"date": d, "event": row.feature})

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date", "event"])

    # Gap-fill event ranges so regressors remain aligned
    filled = []
    for event, grp in df.groupby("event"):
        grp = grp.sort_values("date")
        start = grp["date"].iloc[0]
        prev = start
        for cur in grp["date"].iloc[1:]:
            if (cur - prev).days > 1:
                dates = pd.date_range(start, prev)
                for d in dates:
                    filled.append({"date": d, "event": event})
                start = cur
            prev = cur
        dates = pd.date_range(start, prev)
        for d in dates:
            filled.append({"date": d, "event": event})

    df = pd.DataFrame(filled).drop_duplicates(subset=["date", "event"])\
        .sort_values("date")
    df.reset_index(drop=True, inplace=True)
    return df


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()

    df = get_holidays_dataframe()
    print(df.head())
