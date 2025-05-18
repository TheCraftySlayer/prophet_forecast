import pandas as pd
from datetime import date


def get_holidays_dataframe() -> pd.DataFrame:
    """Return DataFrame with county holidays, tax deadlines and press release dates."""
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

    records = []
    for d in county_holidays:
        records.append({"date": pd.to_datetime(d), "event": "county_holiday"})
    for d in tax_deadlines:
        records.append({"date": pd.to_datetime(d), "event": "tax_deadline"})
    for d in press_release_dates:
        records.append({"date": pd.to_datetime(d), "event": "press_release"})

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = get_holidays_dataframe()
    print(df.head())
