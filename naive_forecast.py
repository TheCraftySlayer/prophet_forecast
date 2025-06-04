"""Seasonal naive forecast using the same hour one week earlier."""

import csv
from datetime import datetime
from math import sqrt


def main() -> None:
    rows = []
    with open("hourly_call_data.csv") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            ts_str, val_str = row[:2]
            try:
                dt = datetime.strptime(ts_str.strip(), "%m/%d/%Y %H:%M")
            except ValueError:
                continue
            rows.append((dt, float(val_str)))

    rows.sort(key=lambda x: x[0])
    if len(rows) < 24 * 21:
        raise SystemExit("Need at least 21 days of hourly data")

    preds = []
    acts = []
    dates = []
    hours = 24 * 7
    for i in range(hours, len(rows)):
        preds.append(rows[i - hours][1])
        acts.append(rows[i][1])
        dates.append(rows[i][0])

    preds = preds[-24 * 14:]
    acts = acts[-24 * 14:]
    dates = dates[-24 * 14:]

    filtered = [
        (d, p, a)
        for d, p, a in zip(dates, preds, acts)
        if d.weekday() < 5 and 8 <= d.hour < 17
    ]

    daily = {}
    for d, p, a in filtered:
        day = d.date()
        pred, act = daily.get(day, (0.0, 0.0))
        daily[day] = (pred + p, act + a)

    dates_out = list(daily.keys())
    preds_out = [v[0] for v in daily.values()]
    acts_out = [v[1] for v in daily.values()]

    n = len(preds_out)
    mae = sum(abs(a - p) for a, p in zip(acts_out, preds_out)) / n
    rmse = sqrt(sum((a - p) ** 2 for a, p in zip(acts_out, preds_out)) / n)

    print("date,predicted,actual")
    for d, p, a in zip(dates_out, preds_out, acts_out):
        print(f"{d},{p},{a}")

    print(f"MAE,{mae:.2f}")
    print(f"RMSE,{rmse:.2f}")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
