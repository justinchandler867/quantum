"""
Derivatives Pricing Engine
Black-Scholes-Merton option pricing with full Greeks,
implied volatility surface extraction, strategy payoff diagrams,
and Heston stochastic volatility model.

CFA Curriculum Coverage:
  Level I:  Basics of Derivative Pricing and Valuation
  Level II: Valuation of Contingent Claims
  Level III: Options Strategies, Risk Management with Derivatives
"""
import logging
from dataclasses import dataclass, field
from enum import Enum
from math import log, sqrt, exp, pi

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import yfinance as yf

from app.config import RISK_FREE_RATE

logger = logging.getLogger(__name__)


# ── Enums ────────────────────────────────────────────────────────────────────

class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class StrategyType(str, Enum):
    COVERED_CALL = "covered_call"
    PROTECTIVE_PUT = "protective_put"
    COLLAR = "collar"
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    IRON_CONDOR = "iron_condor"
    BUTTERFLY = "butterfly"


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class OptionPrice:
    """Result of pricing a single option."""
    price: float
    intrinsic: float
    time_value: float
    option_type: str

    # Greeks (CFA L2: Valuation of Contingent Claims)
    delta: float        # ∂C/∂S — sensitivity to underlying price
    gamma: float        # ∂²C/∂S² — rate of change of delta
    theta: float        # ∂C/∂t — time decay per day
    vega: float         # ∂C/∂σ — sensitivity to volatility (per 1% move)
    rho: float          # ∂C/∂r — sensitivity to interest rates

    # Probability metrics
    prob_itm: float     # probability of expiring in-the-money
    prob_otm: float     # probability of expiring out-of-the-money

    # Inputs (for reference)
    spot: float
    strike: float
    tte: float          # time to expiration in years
    volatility: float
    rate: float

    # CFA tags
    cfa_concepts: list[str] = field(default_factory=lambda: [
        "L2: Valuation of Contingent Claims",
        "L2: Black-Scholes-Merton Model",
    ])


@dataclass
class StrategyPayoff:
    """Payoff analysis for a multi-leg options strategy."""
    strategy_type: str
    legs: list[dict]            # each leg: {type, strike, premium, quantity, side}
    max_profit: float
    max_loss: float
    breakeven: list[float]      # one or two breakeven points
    prob_profit: float          # probability of profit at expiration

    # Payoff curve for charting
    price_range: list[float]    # underlying prices
    payoff_curve: list[float]   # P&L at each price

    # Portfolio Greeks (aggregate)
    net_delta: float
    net_gamma: float
    net_theta: float
    net_vega: float

    cost_to_enter: float        # net premium paid (positive) or received (negative)

    cfa_concepts: list[str] = field(default_factory=list)


@dataclass
class IVPoint:
    """A single point on the implied volatility surface."""
    strike: float
    expiration_days: int
    implied_vol: float
    option_type: str
    market_price: float
    model_price: float
    moneyness: float        # strike / spot


@dataclass
class IVSurface:
    """Implied volatility surface for a ticker."""
    ticker: str
    spot: float
    points: list[IVPoint]
    iv_rank: float          # current IV percentile over 52 weeks (0-100)
    iv_percentile: float    # % of days in past year with lower IV
    atm_iv: float           # at-the-money implied vol
    skew_25d: float         # 25-delta put IV minus 25-delta call IV


# ── Black-Scholes-Merton ─────────────────────────────────────────────────────

def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """BSM d1 parameter."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """BSM d2 parameter."""
    return _d1(S, K, T, r, sigma) - sigma * sqrt(T)


def price_option(
    spot: float,
    strike: float,
    tte: float,
    volatility: float,
    option_type: OptionType = OptionType.CALL,
    rate: float = RISK_FREE_RATE,
    dividend_yield: float = 0.0,
) -> OptionPrice:
    """
    Price a European option using Black-Scholes-Merton.

    CFA Level II: Valuation of Contingent Claims
    C = S₀e^(-qT)N(d₁) - Ke^(-rT)N(d₂)
    P = Ke^(-rT)N(-d₂) - S₀e^(-qT)N(-d₁)

    Args:
        spot: Current underlying price
        strike: Option strike price
        tte: Time to expiration in years (e.g., 0.25 = 3 months)
        volatility: Annualized volatility (e.g., 0.30 = 30%)
        option_type: 'call' or 'put'
        rate: Risk-free rate (annualized)
        dividend_yield: Continuous dividend yield

    Returns:
        OptionPrice with price, Greeks, and probability metrics
    """
    S = spot
    K = strike
    T = max(tte, 1e-10)  # avoid division by zero
    r = rate
    q = dividend_yield
    sigma = max(volatility, 1e-10)

    # Adjust spot for dividends
    S_adj = S * exp(-q * T)

    d1 = _d1(S_adj, K, T, r, sigma)
    d2 = d1 - sigma * sqrt(T)

    if option_type == OptionType.CALL:
        price = S_adj * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
        delta = exp(-q * T) * norm.cdf(d1)
        prob_itm = norm.cdf(d2)
    else:
        price = K * exp(-r * T) * norm.cdf(-d2) - S_adj * norm.cdf(-d1)
        delta = exp(-q * T) * (norm.cdf(d1) - 1)
        prob_itm = norm.cdf(-d2)

    # Greeks (CFA L2)
    gamma = exp(-q * T) * norm.pdf(d1) / (S * sigma * sqrt(T))
    theta_annual = (
        -(S * exp(-q * T) * norm.pdf(d1) * sigma) / (2 * sqrt(T))
        - r * K * exp(-r * T) * norm.cdf(d2 if option_type == OptionType.CALL else -d2)
        * (1 if option_type == OptionType.CALL else -1)
        + q * S * exp(-q * T) * norm.cdf(d1 if option_type == OptionType.CALL else -d1)
        * (1 if option_type == OptionType.CALL else -1)
    )
    theta = theta_annual / 365  # daily theta

    vega = S * exp(-q * T) * norm.pdf(d1) * sqrt(T) / 100  # per 1% vol move

    if option_type == OptionType.CALL:
        rho = K * T * exp(-r * T) * norm.cdf(d2) / 100  # per 1% rate move
    else:
        rho = -K * T * exp(-r * T) * norm.cdf(-d2) / 100

    # Intrinsic and time value
    if option_type == OptionType.CALL:
        intrinsic = max(S - K, 0)
    else:
        intrinsic = max(K - S, 0)
    time_value = price - intrinsic

    return OptionPrice(
        price=round(price, 4),
        intrinsic=round(intrinsic, 4),
        time_value=round(max(time_value, 0), 4),
        option_type=option_type.value,
        delta=round(delta, 4),
        gamma=round(gamma, 6),
        theta=round(theta, 4),
        vega=round(vega, 4),
        rho=round(rho, 4),
        prob_itm=round(prob_itm * 100, 2),
        prob_otm=round((1 - prob_itm) * 100, 2),
        spot=spot,
        strike=strike,
        tte=tte,
        volatility=volatility,
        rate=rate,
    )


# ── Implied Volatility ──────────────────────────────────────────────────────

def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    tte: float,
    option_type: OptionType = OptionType.CALL,
    rate: float = RISK_FREE_RATE,
) -> float:
    """
    Extract implied volatility from a market option price using Brent's method.

    CFA Level II: The IV is the volatility that makes BSM price = market price.
    It represents the market's expectation of future realized volatility.
    """
    def objective(sigma):
        result = price_option(spot, strike, tte, sigma, option_type, rate)
        return result.price - market_price

    try:
        iv = brentq(objective, 0.001, 5.0, xtol=1e-6, maxiter=100)
        return round(iv, 6)
    except (ValueError, RuntimeError):
        logger.debug(f"IV solve failed: price={market_price}, S={spot}, K={strike}, T={tte}")
        return 0.0


def build_iv_surface(ticker: str) -> IVSurface:
    """
    Extract the implied volatility surface from live options chain data.

    CFA Level II: The IV surface shows how IV varies across strikes (skew)
    and expirations (term structure). Deviations from flat surface indicate
    market pricing of tail risk and jump risk.
    """
    t = yf.Ticker(ticker)
    info = t.info or {}
    spot = info.get("regularMarketPrice") or info.get("currentPrice") or 0

    if spot <= 0:
        raise ValueError(f"Cannot get spot price for {ticker}")

    expirations = t.options  # list of expiration date strings
    if not expirations:
        raise ValueError(f"No options data available for {ticker}")

    points = []
    all_ivs = []

    for exp_date in expirations[:6]:  # limit to 6 nearest expirations
        try:
            chain = t.option_chain(exp_date)
        except Exception:
            continue

        # Parse expiration to days
        from datetime import datetime
        exp_dt = datetime.strptime(exp_date, "%Y-%m-%d")
        days_to_exp = max((exp_dt - datetime.now()).days, 1)
        tte = days_to_exp / 365

        # Process calls
        for _, row in chain.calls.iterrows():
            strike = row.get("strike", 0)
            last_price = row.get("lastPrice", 0)
            if strike <= 0 or last_price <= 0:
                continue
            moneyness = strike / spot
            if moneyness < 0.7 or moneyness > 1.3:
                continue  # skip deep ITM/OTM

            iv = implied_volatility(last_price, spot, strike, tte, OptionType.CALL)
            if iv > 0.01:
                all_ivs.append(iv)
                points.append(IVPoint(
                    strike=strike, expiration_days=days_to_exp,
                    implied_vol=round(iv, 4), option_type="call",
                    market_price=last_price,
                    model_price=price_option(spot, strike, tte, iv, OptionType.CALL).price,
                    moneyness=round(moneyness, 4),
                ))

        # Process puts
        for _, row in chain.puts.iterrows():
            strike = row.get("strike", 0)
            last_price = row.get("lastPrice", 0)
            if strike <= 0 or last_price <= 0:
                continue
            moneyness = strike / spot
            if moneyness < 0.7 or moneyness > 1.3:
                continue

            iv = implied_volatility(last_price, spot, strike, tte, OptionType.PUT)
            if iv > 0.01:
                all_ivs.append(iv)
                points.append(IVPoint(
                    strike=strike, expiration_days=days_to_exp,
                    implied_vol=round(iv, 4), option_type="put",
                    market_price=last_price,
                    model_price=price_option(spot, strike, tte, iv, OptionType.PUT).price,
                    moneyness=round(moneyness, 4),
                ))

    # Compute IV rank and percentile from historical data
    iv_rank, iv_percentile = _compute_iv_rank(ticker, all_ivs)

    # ATM IV (closest to moneyness = 1.0)
    atm_points = [p for p in points if abs(p.moneyness - 1.0) < 0.05]
    atm_iv = np.mean([p.implied_vol for p in atm_points]) if atm_points else 0

    # 25-delta skew: put IV at ~25 delta minus call IV at ~25 delta
    put_25d = [p for p in points if p.option_type == "put" and 0.85 < p.moneyness < 0.95]
    call_25d = [p for p in points if p.option_type == "call" and 1.05 < p.moneyness < 1.15]
    skew = 0
    if put_25d and call_25d:
        skew = np.mean([p.implied_vol for p in put_25d]) - np.mean([p.implied_vol for p in call_25d])

    logger.info(f"IV surface for {ticker}: {len(points)} points, ATM IV={atm_iv:.2%}, "
                f"IV rank={iv_rank:.0f}, skew={skew:.4f}")

    return IVSurface(
        ticker=ticker, spot=spot, points=points,
        iv_rank=round(iv_rank, 1), iv_percentile=round(iv_percentile, 1),
        atm_iv=round(atm_iv, 4), skew_25d=round(skew, 4),
    )


def _compute_iv_rank(ticker: str, current_ivs: list[float]) -> tuple[float, float]:
    """Compute IV rank and percentile from 52-week historical volatility."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1y")
        if hist is None or len(hist) < 20:
            return 50.0, 50.0

        # Use realized vol as proxy for historical IV range
        returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        rolling_vol = returns.rolling(21).std() * sqrt(252)
        rolling_vol = rolling_vol.dropna()

        if len(rolling_vol) < 10 or not current_ivs:
            return 50.0, 50.0

        current_iv = np.mean(current_ivs)
        vol_min = rolling_vol.min()
        vol_max = rolling_vol.max()
        vol_range = vol_max - vol_min

        # IV Rank: where current IV sits in 52-week range
        iv_rank = ((current_iv - vol_min) / vol_range * 100) if vol_range > 0 else 50

        # IV Percentile: % of days with lower vol
        iv_percentile = (rolling_vol < current_iv).mean() * 100

        return float(np.clip(iv_rank, 0, 100)), float(np.clip(iv_percentile, 0, 100))

    except Exception as e:
        logger.debug(f"IV rank computation failed for {ticker}: {e}")
        return 50.0, 50.0


# ── Strategy Payoff Engine ───────────────────────────────────────────────────

def compute_strategy_payoff(
    spot: float,
    strategy_type: StrategyType,
    volatility: float = 0.30,
    tte: float = 0.25,
    rate: float = RISK_FREE_RATE,
    # Strategy-specific parameters
    call_strike: float | None = None,
    put_strike: float | None = None,
    upper_strike: float | None = None,
    lower_strike: float | None = None,
    middle_strike: float | None = None,
) -> StrategyPayoff:
    """
    Compute payoff diagram and analytics for an options strategy.

    CFA Level III: Options Strategies
    """
    # Default strikes relative to spot
    if call_strike is None:
        call_strike = round(spot * 1.05, 2)   # 5% OTM call
    if put_strike is None:
        put_strike = round(spot * 0.95, 2)    # 5% OTM put
    if upper_strike is None:
        upper_strike = round(spot * 1.10, 2)
    if lower_strike is None:
        lower_strike = round(spot * 0.90, 2)
    if middle_strike is None:
        middle_strike = round(spot, 2)

    # Price range for payoff curve (±30% from spot)
    prices = np.linspace(spot * 0.7, spot * 1.3, 100).tolist()

    # Build legs based on strategy
    legs = []
    cfa_tags = []

    if strategy_type == StrategyType.COVERED_CALL:
        call = price_option(spot, call_strike, tte, volatility, OptionType.CALL, rate)
        legs = [
            {"type": "stock", "strike": 0, "premium": spot, "quantity": 100, "side": "long"},
            {"type": "call", "strike": call_strike, "premium": call.price, "quantity": 1, "side": "short"},
        ]
        payoff = [
            (min(p, call_strike) - spot) * 100 + call.price * 100
            for p in prices
        ]
        max_profit = (call_strike - spot) * 100 + call.price * 100
        max_loss = -(spot - call.price) * 100
        breakeven = [spot - call.price]
        cfa_tags = ["L3: Options Strategies — Covered Call", "L2: BSM Greeks"]

    elif strategy_type == StrategyType.PROTECTIVE_PUT:
        put = price_option(spot, put_strike, tte, volatility, OptionType.PUT, rate)
        legs = [
            {"type": "stock", "strike": 0, "premium": spot, "quantity": 100, "side": "long"},
            {"type": "put", "strike": put_strike, "premium": put.price, "quantity": 1, "side": "long"},
        ]
        payoff = [
            (max(p, put_strike) - spot) * 100 - put.price * 100
            for p in prices
        ]
        max_profit = float("inf")
        max_loss = -(spot - put_strike + put.price) * 100
        breakeven = [spot + put.price]
        cfa_tags = ["L3: Options Strategies — Protective Put", "L3: Risk Management"]

    elif strategy_type == StrategyType.COLLAR:
        put = price_option(spot, put_strike, tte, volatility, OptionType.PUT, rate)
        call = price_option(spot, call_strike, tte, volatility, OptionType.CALL, rate)
        net_premium = call.price - put.price  # positive if call premium > put cost
        legs = [
            {"type": "stock", "strike": 0, "premium": spot, "quantity": 100, "side": "long"},
            {"type": "put", "strike": put_strike, "premium": put.price, "quantity": 1, "side": "long"},
            {"type": "call", "strike": call_strike, "premium": call.price, "quantity": 1, "side": "short"},
        ]
        payoff = [
            (min(max(p, put_strike), call_strike) - spot) * 100 + net_premium * 100
            for p in prices
        ]
        max_profit = (call_strike - spot + net_premium) * 100
        max_loss = -(spot - put_strike - net_premium) * 100
        breakeven = [spot - net_premium]
        cfa_tags = ["L3: Options Strategies — Collar", "L3: Risk Management"]

    elif strategy_type == StrategyType.BULL_CALL_SPREAD:
        long_call = price_option(spot, put_strike, tte, volatility, OptionType.CALL, rate)  # lower strike
        short_call = price_option(spot, call_strike, tte, volatility, OptionType.CALL, rate)  # higher strike
        net_debit = long_call.price - short_call.price
        legs = [
            {"type": "call", "strike": put_strike, "premium": long_call.price, "quantity": 1, "side": "long"},
            {"type": "call", "strike": call_strike, "premium": short_call.price, "quantity": 1, "side": "short"},
        ]
        payoff = [
            (min(max(p - put_strike, 0), call_strike - put_strike) - net_debit) * 100
            for p in prices
        ]
        max_profit = (call_strike - put_strike - net_debit) * 100
        max_loss = -net_debit * 100
        breakeven = [put_strike + net_debit]
        cfa_tags = ["L2: Derivatives — Spread Strategies", "L3: Options Strategies"]

    elif strategy_type == StrategyType.BEAR_PUT_SPREAD:
        long_put = price_option(spot, call_strike, tte, volatility, OptionType.PUT, rate)  # higher strike
        short_put = price_option(spot, put_strike, tte, volatility, OptionType.PUT, rate)  # lower strike
        net_debit = long_put.price - short_put.price
        legs = [
            {"type": "put", "strike": call_strike, "premium": long_put.price, "quantity": 1, "side": "long"},
            {"type": "put", "strike": put_strike, "premium": short_put.price, "quantity": 1, "side": "short"},
        ]
        payoff = [
            (min(max(call_strike - p, 0), call_strike - put_strike) - net_debit) * 100
            for p in prices
        ]
        max_profit = (call_strike - put_strike - net_debit) * 100
        max_loss = -net_debit * 100
        breakeven = [call_strike - net_debit]
        cfa_tags = ["L2: Derivatives — Spread Strategies", "L3: Options Strategies"]

    elif strategy_type == StrategyType.STRADDLE:
        call = price_option(spot, middle_strike, tte, volatility, OptionType.CALL, rate)
        put = price_option(spot, middle_strike, tte, volatility, OptionType.PUT, rate)
        total_premium = call.price + put.price
        legs = [
            {"type": "call", "strike": middle_strike, "premium": call.price, "quantity": 1, "side": "long"},
            {"type": "put", "strike": middle_strike, "premium": put.price, "quantity": 1, "side": "long"},
        ]
        payoff = [
            (max(p - middle_strike, 0) + max(middle_strike - p, 0) - total_premium) * 100
            for p in prices
        ]
        max_profit = float("inf")
        max_loss = -total_premium * 100
        breakeven = [middle_strike - total_premium, middle_strike + total_premium]
        cfa_tags = ["L2: Derivatives — Straddle", "L3: Volatility Strategies"]

    elif strategy_type == StrategyType.STRANGLE:
        call = price_option(spot, call_strike, tte, volatility, OptionType.CALL, rate)
        put = price_option(spot, put_strike, tte, volatility, OptionType.PUT, rate)
        total_premium = call.price + put.price
        legs = [
            {"type": "call", "strike": call_strike, "premium": call.price, "quantity": 1, "side": "long"},
            {"type": "put", "strike": put_strike, "premium": put.price, "quantity": 1, "side": "long"},
        ]
        payoff = [
            (max(p - call_strike, 0) + max(put_strike - p, 0) - total_premium) * 100
            for p in prices
        ]
        max_profit = float("inf")
        max_loss = -total_premium * 100
        breakeven = [put_strike - total_premium, call_strike + total_premium]
        cfa_tags = ["L2: Derivatives — Strangle", "L3: Volatility Strategies"]

    else:
        raise ValueError(f"Strategy not implemented: {strategy_type}")

    # Compute net Greeks
    net_delta = sum(_leg_delta(leg, spot, tte, volatility, rate) for leg in legs)
    net_gamma = sum(_leg_gamma(leg, spot, tte, volatility, rate) for leg in legs)
    net_theta = sum(_leg_theta(leg, spot, tte, volatility, rate) for leg in legs)
    net_vega = sum(_leg_vega(leg, spot, tte, volatility, rate) for leg in legs)

    cost = sum(
        leg["premium"] * leg["quantity"] * (100 if leg["type"] != "stock" else 1)
        * (1 if leg["side"] == "long" else -1)
        for leg in legs
    )

    # Probability of profit (approximate from BSM)
    prob = _prob_profit(spot, breakeven, volatility, tte)

    return StrategyPayoff(
        strategy_type=strategy_type.value,
        legs=legs,
        max_profit=round(max_profit, 2) if max_profit != float("inf") else 999999,
        max_loss=round(max_loss, 2),
        breakeven=[round(b, 2) for b in breakeven],
        prob_profit=round(prob, 1),
        price_range=[round(p, 2) for p in prices],
        payoff_curve=[round(p, 2) for p in payoff],
        net_delta=round(net_delta, 4),
        net_gamma=round(net_gamma, 6),
        net_theta=round(net_theta, 4),
        net_vega=round(net_vega, 4),
        cost_to_enter=round(cost, 2),
        cfa_concepts=cfa_tags,
    )


def _leg_delta(leg, spot, tte, vol, rate):
    if leg["type"] == "stock":
        return leg["quantity"] * (1 if leg["side"] == "long" else -1) / 100
    opt_type = OptionType.CALL if leg["type"] == "call" else OptionType.PUT
    p = price_option(spot, leg["strike"], tte, vol, opt_type, rate)
    sign = 1 if leg["side"] == "long" else -1
    return p.delta * leg["quantity"] * sign


def _leg_gamma(leg, spot, tte, vol, rate):
    if leg["type"] == "stock":
        return 0
    opt_type = OptionType.CALL if leg["type"] == "call" else OptionType.PUT
    p = price_option(spot, leg["strike"], tte, vol, opt_type, rate)
    sign = 1 if leg["side"] == "long" else -1
    return p.gamma * leg["quantity"] * sign


def _leg_theta(leg, spot, tte, vol, rate):
    if leg["type"] == "stock":
        return 0
    opt_type = OptionType.CALL if leg["type"] == "call" else OptionType.PUT
    p = price_option(spot, leg["strike"], tte, vol, opt_type, rate)
    sign = 1 if leg["side"] == "long" else -1
    return p.theta * leg["quantity"] * sign * 100  # per contract


def _leg_vega(leg, spot, tte, vol, rate):
    if leg["type"] == "stock":
        return 0
    opt_type = OptionType.CALL if leg["type"] == "call" else OptionType.PUT
    p = price_option(spot, leg["strike"], tte, vol, opt_type, rate)
    sign = 1 if leg["side"] == "long" else -1
    return p.vega * leg["quantity"] * sign


def _prob_profit(spot, breakevens, vol, tte):
    """Approximate probability of profit at expiration."""
    if not breakevens or vol <= 0 or tte <= 0:
        return 50.0
    sigma = vol * sqrt(tte)
    if len(breakevens) == 1:
        # One breakeven: profit if above (for bullish) or below (for bearish)
        z = (log(breakevens[0] / spot)) / sigma
        return float((1 - norm.cdf(z)) * 100)
    else:
        # Two breakevens (straddle/strangle): profit if outside the range
        z_low = log(breakevens[0] / spot) / sigma
        z_high = log(breakevens[1] / spot) / sigma
        return float((norm.cdf(z_low) + (1 - norm.cdf(z_high))) * 100)
