import csv
import math
from datetime import date, datetime, timedelta
from types import SimpleNamespace

__version__ = "1.5.3"

__all__ = [
    'DataFrame',
    'Series',
    'DatetimeIndex',
    'to_datetime',
    'read_csv',
    'date_range',
    'Timedelta',
    'api',
]

class DatetimeIndex:
    def __init__(self, data):
        self.data = list(data)

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __iter__(self):
        return iter(self.data)

    def __iter__(self):
        return iter(self.data)

    def __iter__(self):
        return iter(self.data)

    def __iter__(self):
        return iter(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return DatetimeIndex(self.data[item])
        return self.data[item]

    def normalize(self):
        norm = []
        for d in self.data:
            if isinstance(d, datetime):
                norm.append(d.replace(hour=0, minute=0, second=0, microsecond=0))
            else:
                norm.append(datetime.combine(d, datetime.min.time()))
        return DatetimeIndex(norm)

    @property
    def dayofweek(self):
        return Series([d.weekday() for d in self.data], self.data)

def _ensure_index(index, length):
    if index is None:
        return DatetimeIndex(range(length))
    if isinstance(index, DatetimeIndex):
        return index
    return DatetimeIndex(index)

class Series:
    def __init__(self, data, index=None, name=None):
        self.data = list(data)
        self.index = _ensure_index(index, len(self.data))
        self.name = name

    class _ILoc:
        def __init__(self, series):
            self.series = series
        def __getitem__(self, idx):
            if isinstance(idx, slice):
                return Series(self.series.data[idx], self.series.index.data[idx], self.series.name)
            return self.series.data[idx]

    @property
    def iloc(self):
        return Series._ILoc(self)

    def __len__(self):
        return len(self.data)
    def __iter__(self):
        return iter(self.data)

    class _DatetimeAccessor:
        def __init__(self, series):
            self.series = series

        @property
        def dayofweek(self):
            return Series([d.weekday() if d is not None else None for d in self.series.data], self.series.index)

    @property
    def dt(self):
        return Series._DatetimeAccessor(self)

    @property
    def values(self):
        return self.data

    def abs(self):
        return Series([abs(v) if v is not None else None for v in self.data], self.index, self.name)

    def mean(self):
        # Ignore ``None`` and ``NaN`` values when computing the mean to mimic
        # pandas behaviour.
        vals = [
            v for v in self.data
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        ]
        return sum(vals) / len(vals) if vals else float("nan")

    def shift(self, periods=1):
        if periods > 0:
            shifted = [None]*periods + self.data[:-periods]
        elif periods < 0:
            shifted = self.data[-periods:] + [None]*(-periods)
        else:
            shifted = list(self.data)
        return Series(shifted, self.index, self.name)

    def dropna(self):
        new_data = []
        new_index = []
        for d,i in zip(self.data, self.index.data):
            if d is not None and not (isinstance(d,float) and math.isnan(d)):
                new_data.append(d)
                new_index.append(i)
        return Series(new_data, new_index, self.name)

    def replace(self, old, new):
        return Series([new if v==old else v for v in self.data], self.index, self.name)

    def astype(self, typ):
        if typ is int:
            return Series([int(v) if v is not None else None for v in self.data], self.index, self.name)
        elif typ is float:
            return Series([float(v) if v is not None else None for v in self.data], self.index, self.name)
        return self

    def __sub__(self, other):
        if isinstance(other, Series):
            other_data = other.data
        else:
            other_data = [other] * len(self.data)
        return Series([(a - b) if a is not None and b is not None else None for a,b in zip(self.data, other_data)], self.index, self.name)

    def __add__(self, other):
        if isinstance(other, Series):
            other_data = other.data
        else:
            other_data = [other]*len(self.data)
        return Series([(a + b) if a is not None and b is not None else None for a,b in zip(self.data, other_data)], self.index, self.name)

    def __mul__(self, other):
        if isinstance(other, Series):
            other_data = other.data
        else:
            other_data = [other]*len(self.data)
        return Series([(a * b) if a is not None and b is not None else None for a,b in zip(self.data, other_data)], self.index, self.name)

    def __truediv__(self, other):
        if isinstance(other, Series):
            other_data = other.data
        else:
            other_data = [other]*len(self.data)
        return Series([(a / b) if a is not None and b not in (0,None) else None for a,b in zip(self.data, other_data)], self.index, self.name)

    def __pow__(self, power):
        return Series([ (a ** power) if a is not None else None for a in self.data], self.index, self.name)

    def isin(self, values):
        vset = set(values)
        return Series([v in vset for v in self.data], self.index)

    def sort_index(self):
        order = sorted(range(len(self.index.data)), key=lambda i: self.index.data[i])
        return Series([self.data[i] for i in order], [self.index.data[i] for i in order], self.name)

    # comparison operators returning boolean Series
    def _compare(self, other, op):
        return Series([op(v, other) for v in self.data], self.index)

    def __lt__(self, other):
        return self._compare(other, lambda a, b: a < b)

    def __le__(self, other):
        return self._compare(other, lambda a, b: a <= b)

    def __gt__(self, other):
        return self._compare(other, lambda a, b: a > b)

    def __ge__(self, other):
        return self._compare(other, lambda a, b: a >= b)

    def __eq__(self, other):
        return self._compare(other, lambda a, b: a == b)

    def __ne__(self, other):
        return self._compare(other, lambda a, b: a != b)

class DataFrame:
    def __init__(self, data):
        if isinstance(data, dict):
            self.columns = list(data.keys())
            length = len(next(iter(data.values()), []))
            self.data = {}
            for c, v in data.items():
                if isinstance(v, Series):
                    self.data[c] = v
                else:
                    self.data[c] = Series(v, range(length), c)
            self.index = _ensure_index(self.data[self.columns[0]].index, length)
            for s in self.data.values():
                s.index = self.index
        elif isinstance(data, list):
            if data:
                self.columns = list(data[0].keys())
            else:
                self.columns = []
            cols = {c: [] for c in self.columns}
            for row in data:
                for c in self.columns:
                    cols[c].append(row.get(c))
            length = len(data)
            self.data = {c: Series(vals, range(length), c) for c, vals in cols.items()}
            self.index = _ensure_index(None, length)
        else:
            raise ValueError('Invalid data type for DataFrame')

    def __getitem__(self, key):
        if isinstance(key, Series):
            # Boolean indexing
            mask = [bool(v) for v in key.data]
            new_data = {c: [s.data[i] for i,m in enumerate(mask) if m] for c,s in self.data.items()}
            new_index = [self.index.data[i] for i,m in enumerate(mask) if m]
            result = DataFrame(new_data)
            result.index = DatetimeIndex(new_index)
            for c in result.columns:
                result.data[c].index = result.index
            return result
        return self.data[key]

    class _ILoc:
        def __init__(self, df):
            self.df = df
        def __getitem__(self, idx):
            if isinstance(idx, slice):
                order = range(*idx.indices(len(self.df.index.data)))
                new_data = {c:[s.data[i] for i in order] for c,s in self.df.data.items()}
                new_index = [self.df.index.data[i] for i in order]
                result = DataFrame(new_data)
                result.index = DatetimeIndex(new_index)
                for c in result.columns:
                    result.data[c].index = result.index
                return result
            else:
                i = idx
                new_data = {c:[s.data[i]] for c,s in self.df.data.items()}
                result = DataFrame(new_data)
                result.index = DatetimeIndex([self.df.index.data[i]])
                for c in result.columns:
                    result.data[c].index = result.index
                return result

    @property
    def iloc(self):
        return DataFrame._ILoc(self)

    def __setitem__(self, key, value):
        if not isinstance(value, Series):
            value = Series(value, self.index, key)
        self.data[key] = value
        if key not in self.columns:
            self.columns.append(key)

    def sort_index(self):
        order = sorted(range(len(self.index.data)), key=lambda i: self.index.data[i])
        return self._take(order)

    def sort_values(self, column):
        order = sorted(range(len(self.index.data)), key=lambda i: self.data[column].data[i])
        return self._take(order)

    def _take(self, order):
        new_data = {c: [s.data[i] for i in order] for c,s in self.data.items()}
        new_index = [self.index.data[i] for i in order]
        result = DataFrame(new_data)
        result.index = DatetimeIndex(new_index)
        for col in result.columns:
            result.data[col].index = result.index
        return result

    def reset_index(self, drop=False):
        if drop:
            self.index = DatetimeIndex(range(len(self.index.data)))
            for s in self.data.values():
                s.index = self.index
            return self
        else:
            data = {'index': self.index.data}
            for c in self.columns:
                data[c] = self.data[c].data
            return DataFrame(data)

    def dropna(self, subset):
        keep = []
        for i in range(len(self.index.data)):
            ok = True
            for col in subset:
                val = self.data[col].data[i]
                if val is None or (isinstance(val,float) and math.isnan(val)):
                    ok = False
                    break
            if ok:
                keep.append(i)
        new_data = {c: [s.data[i] for i in keep] for c,s in self.data.items()}
        new_index = [self.index.data[i] for i in keep]
        self.data = {c: Series(vals, new_index, c) for c, vals in new_data.items()}
        self.index = DatetimeIndex(new_index)
        return self

    def rename(self, columns, inplace=False):
        new_data = {}
        for c,s in self.data.items():
            new_name = columns.get(c,c)
            new_data[new_name] = Series(s.data, s.index, new_name)
        if inplace:
            self.data = new_data
            self.columns = list(new_data.keys())
            return self
        else:
            df = DataFrame({k:v.data for k,v in new_data.items()})
            df.index = self.index
            for col in df.columns:
                df.data[col].index = df.index
            return df

    def set_index(self, column):
        self.index = DatetimeIndex(self.data[column].data)
        del self.data[column]
        self.columns.remove(column)
        for s in self.data.values():
            s.index = self.index
        return self

    def to_csv(self, path, index=False):
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['index'] + self.columns if index else self.columns
            writer.writerow(header)
            for i in range(len(self.index.data)):
                row = []
                if index:
                    row.append(self.index.data[i])
                for c in self.columns:
                    row.append(self.data[c].data[i])
                writer.writerow(row)

    @property
    def empty(self):
        return len(self.index.data) == 0

    def __len__(self):
        return len(self.index.data)

# CSV reader

def read_csv(path, header='infer', names=None):
    with open(path, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
    if header == 'infer':
        headers = [h.strip('\ufeff') for h in rows[0]]
        data_rows = rows[1:]
    elif header is None:
        headers = names if names is not None else [f'col{i}' for i in range(len(rows[0]))]
        data_rows = rows
    else:
        headers = rows[header]
        data_rows = rows[header+1:]
    records = []
    for row in data_rows:
        if not row:
            continue
        record = {}
        for h, v in zip(headers, row):
            v_strip = v.strip()
            if v_strip.replace('.', '', 1).isdigit():
                if '.' in v_strip:
                    record[h] = float(v_strip)
                else:
                    record[h] = int(v_strip)
            else:
                record[h] = v_strip
        records.append(record)
    return DataFrame(records)

# Datetime conversion

def _parse_date(x):
    if isinstance(x, datetime):
        return x
    if isinstance(x, date):
        return datetime.combine(x, datetime.min.time())
    for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%m/%d/%y'):
        try:
            return datetime.strptime(x, fmt)
        except Exception:
            continue
    return None

def to_datetime(obj, errors='raise'):
    if isinstance(obj, Series):
        data = [_parse_date(x) for x in obj.data]
        if errors == 'raise' and any(d is None for d in data):
            raise ValueError('could not parse some dates')
        return Series(data, obj.index, obj.name)
    elif isinstance(obj, list):
        data = [_parse_date(x) for x in obj]
        if errors == 'raise' and any(d is None for d in data):
            raise ValueError('could not parse some dates')
        return [d for d in data]
    else:
        d = _parse_date(obj)
        if d is None and errors == 'raise':
            raise ValueError(f'could not parse date {obj}')
        return d

# ---------------------------------------------------------------------------
# Additional helpers used by the forecast pipeline

class Timedelta(timedelta):
    """Minimal Timedelta implementation wrapping datetime.timedelta."""


def date_range(start=None, end=None, periods=None, freq='D'):
    """Generate a simple DatetimeIndex between start and end."""
    if freq not in ['D', 'B', 'MS']:
        raise ValueError('unsupported freq')
    if periods is not None and end is None:
        if start is None:
            raise ValueError('start must be provided if end is None')
        if freq == 'B':
            dates = []
            current = start
            while len(dates) < periods:
                if current.weekday() < 5:
                    dates.append(current)
                current += timedelta(days=1)
        else:
            step = timedelta(days=1)
            if freq == 'MS':
                current = start.replace(day=1)
                step = None
            dates = []
            current_date = current
            while len(dates) < periods:
                dates.append(current_date)
                if freq == 'MS':
                    month = current_date.month + 1
                    year = current_date.year + (month - 1) // 12
                    month = (month - 1) % 12 + 1
                    current_date = current_date.replace(year=year, month=month, day=1)
                else:
                    current_date += step
        return DatetimeIndex(dates)

    if start is None or end is None:
        raise ValueError('start and end must be provided if periods is None')

    dates = []
    current = start
    while current <= end:
        if freq == 'B' and current.weekday() >= 5:
            current += timedelta(days=1)
            continue
        dates.append(current)
        if freq == 'MS':
            month = current.month + 1
            year = current.year + (month - 1) // 12
            month = (month - 1) % 12 + 1
            current = current.replace(year=year, month=month, day=1)
        else:
            current += timedelta(days=1)
    return DatetimeIndex(dates)

class _Types:
    @staticmethod
    def is_datetime64_any_dtype(series):
        data = series.data if isinstance(series, Series) else series
        return all(isinstance(x, (datetime, date)) for x in data)

api = SimpleNamespace(types=_Types())
