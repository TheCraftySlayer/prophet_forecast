"""Seasonal naive forecast using last week's value.

This script reads ``calls.csv`` and produces predictions for the last
14 days using the value from the same weekday in the prior week as the
prediction. It reports MAE and RMSE to compare predictions to actual values.
"""
import csv
from datetime import datetime
from math import sqrt

# Load call data
rows = []
with open('calls.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row:
            continue
        date_str, value_str = row
        parsed = None
        for fmt in ("%m/%d/%Y", "%m/%d/%y"):
            try:
                parsed = datetime.strptime(date_str.strip(), fmt)
                break
            except ValueError:
                continue
        if not parsed:
            # skip header or malformed lines
            continue
        date = parsed
        rows.append((date.strftime('%Y-%m-%d'), float(value_str)))

# Ensure data is sorted by date
rows.sort(key=lambda x: x[0])

# Need one week of extra history for the seasonal naive baseline
recent = rows[-21:]

preds = []
acts = []
dates = []
for i in range(7, len(recent)):
    pred = recent[i - 7][1]
    actual = recent[i][1]
    preds.append(pred)
    acts.append(actual)
    dates.append(recent[i][0])

# Compute metrics
n = len(preds)
mae = sum(abs(a - p) for a, p in zip(acts, preds)) / n
rmse = sqrt(sum((a - p) ** 2 for a, p in zip(acts, preds)) / n)

print("date,predicted,actual")
for d, p, a in zip(dates, preds, acts):
    print(f"{d},{p},{a}")

print(f"MAE,{mae:.2f}")
print(f"RMSE,{rmse:.2f}")
