__version__ = "0.0"


class DataFrame:
    pass

class Series:
    pass

class DatetimeIndex(list):
    pass

def date_range(*args, **kwargs):
    return DatetimeIndex()

def read_csv(*args, **kwargs):
    return DataFrame()

def read_excel(*args, **kwargs):
    return DataFrame()

def to_datetime(*args, **kwargs):
    return Series()

class api:
    class types:
        @staticmethod
        def is_datetime64_any_dtype(_obj):
            return True
