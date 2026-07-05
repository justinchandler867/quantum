"""
Minimal construction test for ScreenedAsset.

This is the regression test that would have caught the `earnings_date`
cascade: when a new optional field is added to the ScreenedAsset dataclass
but a construction site is missed, the app fails at runtime. Constructing
one asset here with *all* optional fields populated exercises every keyword
and fails loudly at collection time if the dataclass signature drifts.
"""
from __future__ import annotations

from dataclasses import fields

from app.screener import ScreenedAsset


def _make_full_asset() -> ScreenedAsset:
    """Construct a ScreenedAsset with every field — required and optional — set."""
    return ScreenedAsset(
        # Identity + liquidity
        ticker="AAPL",
        name="Apple Inc.",
        sector="Technology",
        industry="Consumer Electronics",
        market_cap=3.0e12,
        price=225.0,
        avg_volume=5.0e7,
        # Return metrics
        return_1y=0.24,
        return_6m=0.11,
        return_3m=0.06,
        volatility=0.28,
        sharpe=0.9,
        beta=1.2,
        max_drawdown=-0.18,
        dividend_yield=0.005,
        # Valuation
        pe_ratio=32.0,
        forward_pe=28.0,
        pb_ratio=45.0,
        earnings_yield=0.031,
        # Calendar + growth/leverage extras (the fields the cascade touched)
        earnings_date="2026-08-01",
        dividend_growth_5y=0.07,
        debt_to_equity=1.5,
        revenue_growth=0.08,
        # Factor z-scores
        z_momentum=0.5,
        z_quality=0.3,
        z_value=-0.2,
        z_low_vol=0.1,
        z_yield=-0.4,
        # Composite scores
        factor_composite=0.42,
        fit_score=78,
        rank=1,
    )


def test_screened_asset_constructs_with_all_optional_fields():
    asset = _make_full_asset()
    # The field that triggered the cascade must round-trip.
    assert asset.earnings_date == "2026-08-01"
    # And the other optional extras must be settable.
    assert asset.dividend_growth_5y == 0.07
    assert asset.debt_to_equity == 1.5
    assert asset.revenue_growth == 0.08


def test_screened_asset_optionals_default_to_none_or_zero():
    """A minimal asset (no optional fields) must still construct — defaults hold."""
    minimal = ScreenedAsset(
        ticker="MSFT", name="Microsoft", sector="Technology",
        industry="Software", market_cap=2.5e12, price=400.0, avg_volume=2.0e7,
        return_1y=0.2, return_6m=0.1, return_3m=0.05, volatility=0.25,
        sharpe=0.8, beta=1.0, max_drawdown=-0.15, dividend_yield=0.008,
        pe_ratio=30.0, forward_pe=27.0, pb_ratio=12.0, earnings_yield=0.033,
    )
    assert minimal.earnings_date is None
    assert minimal.z_momentum == 0.0
    assert minimal.fit_score == 50


def test_screened_asset_has_no_unexpected_required_fields():
    """
    Guard against a new *required* field silently appearing: every field
    without a default is covered by _make_full_asset above. If someone adds a
    required field, this constructs-everything path breaks — which is the point.
    """
    _make_full_asset()  # would raise TypeError on a missing required field
    names = {f.name for f in fields(ScreenedAsset)}
    assert "earnings_date" in names
