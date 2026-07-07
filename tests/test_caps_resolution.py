"""
Three-tier resolution order for _get_market_caps (equilibrium-weight sizes):

    1. Live fetch (yfinance marketCap / ETF totalAssets)
    2. caps.json snapshot  (curated, for tickers live can't supply)
    3. Basket-median       (a ticker missing from BOTH live AND snapshot)

Availability is mocked at each tier so the resolution order is asserted without
touching the network. The contract under test:
  - caps_source == "snapshot <asof>" iff tier 2 supplied at least one ticker
  - the constraints note fires ONLY on the true median fallback (tier 3)
"""
from __future__ import annotations

import numpy as np
import pytest

import app.main as main


class _FakeObj:
    def __init__(self, info: dict):
        self._info = info

    @property
    def info(self):
        return self._info


class _FakeTickers:
    """Stand-in for yfinance.Tickers — resolves caps from a fixed mapping.

    A cap of 0 (or a ticker absent from the mapping) yields empty `info`, i.e.
    the live tier could not supply that ticker.
    """
    def __init__(self, live: dict[str, float]):
        self._live = live

    def __call__(self, _space_joined: str):
        self.tickers = {
            t: _FakeObj({"marketCap": v} if v else {})
            for t, v in self._live.items()
        }
        return self


@pytest.fixture
def caps_env(monkeypatch):
    """Isolate _get_market_caps: empty live cache, controllable live+snapshot."""
    monkeypatch.setitem(main._store, "fundamentals", {})
    monkeypatch.setattr(main, "_CAPS_ASOF", "2026-07-06")

    def configure(live: dict[str, float], snapshot: dict[str, float]):
        import yfinance
        monkeypatch.setattr(yfinance, "Tickers", _FakeTickers(live))
        monkeypatch.setattr(main, "_CAPS_SNAPSHOT", dict(snapshot))

    return configure


def test_tier1_live_fetch_wins(caps_env):
    """Every ticker resolvable live: caps come from live, no snapshot, no note."""
    caps_env(live={"AAPL": 100.0, "MSFT": 50.0}, snapshot={"AAPL": 1.0, "MSFT": 1.0})

    caps, note, source = main._get_market_caps(["AAPL", "MSFT"])

    assert np.allclose(caps, [100.0, 50.0])   # live values, not the snapshot's 1.0
    assert source is None                      # tier 2 never consulted
    assert note is None                        # nothing hit the median


def test_tier2_snapshot_fills_live_gap(caps_env):
    """A ticker live can't supply falls through to the snapshot, not the median."""
    caps_env(live={"AAPL": 100.0, "MSFT": 0.0}, snapshot={"MSFT": 50.0})

    caps, note, source = main._get_market_caps(["AAPL", "MSFT"])

    assert np.allclose(caps, [100.0, 50.0])    # MSFT from snapshot
    assert source == "snapshot 2026-07-06"     # tier 2 supplied a ticker
    assert note is None                        # median fallback did NOT fire


def test_tier3_median_fallback_only_when_absent_from_both(caps_env):
    """Missing from live AND snapshot -> basket-median, and the note names it."""
    caps_env(live={"AAPL": 100.0, "MSFT": 50.0, "ZZZ": 0.0}, snapshot={})

    caps, note, source = main._get_market_caps(["AAPL", "MSFT", "ZZZ"])

    assert np.allclose(caps[:2], [100.0, 50.0])
    assert caps[2] == pytest.approx(np.median([100.0, 50.0]))  # 75.0
    assert source is None                      # snapshot supplied nothing
    assert note is not None and "ZZZ" in note  # honest median note fires for ZZZ
