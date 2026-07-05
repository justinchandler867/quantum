"""
Regression test for the dividend_yield double-conversion bug.

Pre-fix, screener.py applied `dy*100 if dy < 1 else dy` then `min(..., 25)`.
The current yfinance contract returns dividendYield as a percent number
already, so any yield < 1% was multiplied by 100 and clamped to the 25
sentinel (AAPL 0.35% -> 35 -> 25). This test pins the corrected behavior:
sub-1% yields are stored as-is, and the winsorization ceiling only trips
above 25.
"""
from __future__ import annotations

import logging

from app.screener import normalize_dividend_yield, DIVIDEND_YIELD_CEILING


def test_raw_percent_values_stored_directly():
    # These are the exact raw yfinance values observed in the shakedown.
    assert normalize_dividend_yield(0.35) == 0.35   # AAPL — was clamped to 25 pre-fix
    assert normalize_dividend_yield(2.04) == 2.04   # JNJ
    assert normalize_dividend_yield(3.01) == 3.01   # XOM


def test_sub_one_percent_is_not_multiplied():
    # The heuristic that caused the bug: anything < 1 got * 100.
    for raw in (0.35, 0.51, 0.68, 0.74, 0.89):
        assert normalize_dividend_yield(raw) == round(raw, 2)


def test_none_and_zero_yield_to_zero():
    assert normalize_dividend_yield(None) == 0.0
    assert normalize_dividend_yield(0) == 0.0


def test_ceiling_winsorizes_and_warns(caplog):
    with caplog.at_level(logging.WARNING):
        out = normalize_dividend_yield(40.0, "BADTICKER")
    assert out == DIVIDEND_YIELD_CEILING
    assert any("winsoriz" in r.message.lower() for r in caplog.records)
    assert any("BADTICKER" in r.message for r in caplog.records)


def test_value_below_ceiling_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING):
        out = normalize_dividend_yield(3.01, "XOM")
    assert out == 3.01
    assert not caplog.records
