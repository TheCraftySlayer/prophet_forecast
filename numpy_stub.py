import math

def sqrt(x):
    return math.sqrt(x)

def log1p(x):
    return math.log1p(x)

def expm1(x):
    return math.expm1(x)

def abs(x):
    return math.fabs(x)

def isinf(x):
    return math.isinf(x)

def argmin(seq):
    return min(range(len(seq)), key=lambda i: seq[i])

def mean(seq):
    if not seq:
        return float('nan')
    return sum(seq) / len(seq)

nan = float('nan')
