"""Naive forecast for call volumes using previous day's value.

This script reads `calls.csv` and produces predictions for the last
14 days using the value from the prior day as the prediction. It also
reports MAE, RMSE and MAPE to compare predictions to actual values.
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
        try:
            date = datetime.strptime(date_str.strip(), '%m/%d/%Y')
        except ValueError:
            # skip header or malformed lines
            continue
        rows.append((date.strftime('%Y-%m-%d'), float(value_str)))

# Ensure data is sorted by date
rows.sort(key=lambda x: x[0])

# Need one extra day for the first prediction
recent = rows[-15:]

preds = []
acts = []
dates = []
for i in range(1, len(recent)):
    pred = recent[i - 1][1]
    actual = recent[i][1]
    preds.append(pred)
    acts.append(actual)
    dates.append(recent[i][0])

# Compute metrics
n = len(preds)
mae = sum(abs(a - p) for a, p in zip(acts, preds)) / n
rmse = sqrt(sum((a - p) ** 2 for a, p in zip(acts, preds)) / n)
mape = (
    sum(abs(a - p) / a for a, p in zip(acts, preds) if a != 0) / n * 100
)

print("date,predicted,actual")
for d, p, a in zip(dates, preds, acts):
    print(f"{d},{p},{a}")

print(f"MAE,{mae:.2f}")
print(f"RMSE,{rmse:.2f}")
print(f"MAPE,{mape:.2f}%")
