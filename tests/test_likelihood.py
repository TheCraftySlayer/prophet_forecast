import pytest
pytest.importorskip("pandas")

import pandas as pd
from prophet_analysis import select_likelihood


def test_select_likelihood_poisson():
    s = pd.Series([1, 2, 2, 1, 2])
    assert select_likelihood(s) == "poisson"


def test_select_likelihood_neg_binomial():
    s = pd.Series([1, 10, 1, 10, 1])
    assert select_likelihood(s) == "neg_binomial"
