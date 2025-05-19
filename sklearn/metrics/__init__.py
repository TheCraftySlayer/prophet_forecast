from math import sqrt

def mean_squared_error(y_true, y_pred, squared=True):
    if not y_true:
        return 0.0
    mse = sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)
    if squared:
        return mse
    return sqrt(mse)

def mean_absolute_error(y_true, y_pred):
    if not y_true:
        return 0.0
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)
