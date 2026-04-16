"""
Data Ingestion Layer
Fetches adjusted close prices from yfinance, computes log returns,
and identifies stress regime windows.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from app.config import (
    PRICE_HISTORY_YEARS,
    STRESS_DRAWDOWN_THRESHOLD,
    MIN_STRESS_DAYS,
    REFERENCE_HEDGES,
)

logger = logging.getLogger(__name__)


def fetch_prices(
    tickers: list[str],
    years: int = PRICE_HISTORY_YEARS,
    include_hedges: bool = True,
) -> pd.DataFrame:
    """
    Fetch adjusted close prices for a list of tickers.
    Returns a DataFrame with DatetimeIndex and one column per ticker.
    Missing tickers are dropped with a warning.
    """
    all_tickers = list(set(tickers))
    if include_hedges:
        all_tickers = list(set(all_tickers + REFERENCE_HEDGES))

    end = datetime.now()
    start = end - timedelta(days=years * 365)

    logger.info(f"Fetching {len(all_tickers)} tickers from {start.date()} to {end.date()}")

    # Use Ticker.history() per ticker (more reliable across yfinance versions)
    frames = {}
    for ticker in all_tickers:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
            if hist is not None and not hist.empty and "Close" in hist.columns:
                frames[ticker] = hist["Close"]
        except Exception as ex:
            logger.debug(f"Skipping {ticker}: {ex}")

    if not frames:
        raise ValueError("No price data returned from yfinance")

    prices = pd.DataFrame(frames)

    # Drop tickers with insufficient data (< 60% of expected trading days)
    min_rows = int(years * 252 * 0.6)
    valid = prices.columns[prices.count() >= min_rows]
    dropped = set(prices.columns) - set(valid)
    if dropped:
        logger.warning(f"Dropped tickers with insufficient data: {dropped}")
    prices = prices[valid]

    # Forward-fill gaps (up to 5 days for holidays/halts), then drop remaining NaNs
    prices = prices.ffill(limit=5).dropna()

    logger.info(f"Price matrix: {prices.shape[0]} days × {prices.shape[1]} tickers")
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from price DataFrame.
    log return = ln(P_t / P_{t-1})
    """
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns


def identify_stress_windows(
    returns: pd.DataFrame,
    benchmark: str = "QQQ",
    threshold: float = STRESS_DRAWDOWN_THRESHOLD,
) -> pd.Series:
    """
    Identify stress regime days based on benchmark drawdown.

    A day is in 'stress' if the benchmark is in a drawdown exceeding
    the threshold from its rolling 252-day high.

    Returns a boolean Series aligned with the returns index.
    True = stress day.
    """
    if benchmark not in returns.columns:
        # Fetch QQQ separately if not in the universe
        qqq_prices = fetch_prices([benchmark], include_hedges=False)
        qqq_returns = compute_log_returns(qqq_prices)
        bench_cumret = qqq_returns[benchmark].cumsum()
    else:
        bench_cumret = returns[benchmark].cumsum()

    # Align to returns index
    bench_cumret = bench_cumret.reindex(returns.index).ffill()

    # Rolling 252-day max of cumulative return (proxy for trailing high)
    rolling_high = bench_cumret.rolling(window=252, min_periods=20).max()

    # Drawdown from rolling high
    drawdown = bench_cumret - rolling_high

    # Stress = drawdown exceeds threshold (threshold is negative, e.g., -0.15)
    stress_mask = drawdown < threshold

    stress_days = stress_mask.sum()
    logger.info(
        f"Stress regime: {stress_days} days ({stress_days / len(returns) * 100:.1f}%) "
        f"using {benchmark} with threshold {threshold:.0%}"
    )

    if stress_days < MIN_STRESS_DAYS:
        logger.warning(
            f"Only {stress_days} stress days found (minimum {MIN_STRESS_DAYS}). "
            f"Stress estimates may be unstable. Consider lowering threshold."
        )

    return stress_mask


def split_returns_by_regime(
    returns: pd.DataFrame,
    stress_mask: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split returns into normal and stress regimes.
    Returns (normal_returns, stress_returns).
    """
    # Ensure alignment
    mask = stress_mask.reindex(returns.index).fillna(False)

    normal = returns[~mask]
    stress = returns[mask]

    logger.info(f"Normal regime: {len(normal)} days | Stress regime: {len(stress)} days")
    return normal, stress


def get_trailing_returns(
    returns: pd.DataFrame,
    window: int = 252,
) -> pd.DataFrame:
    """
    Get the most recent `window` trading days of returns.
    Used for the normal-regime correlation matrix.
    """
    return returns.tail(window)


def annualized_volatility(returns: pd.DataFrame) -> pd.Series:
    """
    Compute annualized volatility (σ) for each ticker.
    σ_annual = σ_daily × √252
    """
    return returns.std() * np.sqrt(252)


def compute_return_stats(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics per ticker for screening.
    Returns DataFrame with columns: ann_return, ann_vol, sharpe, beta, max_dd
    """
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol.replace(0, np.nan)

    # Beta vs QQQ (or first ticker as proxy)
    bench = "QQQ" if "QQQ" in returns.columns else returns.columns[0]
    bench_var = returns[bench].var()
    beta = returns.apply(lambda col: col.cov(returns[bench]) / bench_var if bench_var > 0 else 1.0)

    # Max drawdown per ticker
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = drawdown.min()

    stats = pd.DataFrame({
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "beta": beta,
        "max_dd": max_dd,
    })

    return stats
