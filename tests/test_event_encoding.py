import pytest
pytest.importorskip("pandas")

import pandas as pd
from holidays_calendar import get_holidays_dataframe
from prophet_analysis import encode_assessor_events


def test_encode_outage_event():
    df = get_holidays_dataframe()
    idx = pd.date_range("2024-07-18", periods=5, freq="D")
    events = df[df["event"].isin(["portal_down", "site_outage"])]
    flags = encode_assessor_events(idx, events)
    assert "portal_down" in flags.columns
    assert flags.loc[pd.Timestamp("2024-07-19"), "portal_down"] == 1
    assert flags["portal_down"].sum() == 1


def test_encode_policy_step():
    df = get_holidays_dataframe()
    idx = pd.date_range("2025-03-19", periods=3, freq="D")
    events = df[df["event"] == "hb47_effective"]
    flags = encode_assessor_events(idx, events)
    assert list(flags["hb47_effective"]) == [0, 1, 1]
