"""
Correlation Engine
Computes normal, stress, and blended correlation/covariance matrices
with Ledoit-Wolf shrinkage estimation.
"""
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from app.config import (
    DEFAULT_LAMBDA,
    LAMBDA_BASE,
    LAMBDA_RANGE,
    SHRINKAGE_ALWAYS_STRESS,
    SHRINKAGE_NORMAL_THRESHOLD,
    HIGH_CORR_THRESHOLD,
    HIGH_CORR_NORMAL_THRESHOLD,
    NEGATIVE_CORR_THRESHOLD,
    REFERENCE_HEDGES,
)

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Container for a correlation matrix plus metadata."""
    matrix: np.ndarray               # n × n correlation matrix
    tickers: list[str]               # ticker labels, same order as matrix axes
    regime: str                      # 'normal', 'stress', or 'blended'
    observation_days: int            # how many trading days were used
    shrinkage_applied: bool = False
    shrinkage_coefficient: float = 0.0


@dataclass
class CovarianceResult:
    """Container for optimization-ready covariance matrix."""
    matrix: np.ndarray               # n × n covariance matrix
    tickers: list[str]
    lambda_used: float
    vol_normal: np.ndarray           # annualized vol vector (normal regime)
    vol_stress: np.ndarray           # annualized vol vector (stress regime)


@dataclass
class CorrelationDiagnostics:
    """Summary diagnostics for the frontend."""
    avg_corr_normal: float
    avg_corr_stress: float
    corr_spike_pct: float            # percentage increase from normal → stress
    high_corr_pairs_normal: list[tuple[str, str, float]] = field(default_factory=list)
    high_corr_pairs_stress: list[tuple[str, str, float]] = field(default_factory=list)
    hedging_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    diversification_ratio_normal: float = 0.0
    diversification_ratio_stress: float = 0.0
    reference_correlations: dict = field(default_factory=dict)


def _correlation_from_returns(
    returns: pd.DataFrame,
    use_shrinkage: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Compute correlation matrix from returns DataFrame.
    Optionally applies Ledoit-Wolf shrinkage.

    Returns (correlation_matrix, shrinkage_coefficient).
    Shrinkage coefficient is 0.0 if shrinkage not applied.
    """
    if use_shrinkage and len(returns) >= 10:
        # Ledoit-Wolf operates on covariance; we extract correlation from it
        lw = LedoitWolf().fit(returns.values)
        cov = lw.covariance_
        shrink_coef = lw.shrinkage_

        # Convert covariance → correlation
        d = np.sqrt(np.diag(cov))
        d[d == 0] = 1e-10  # avoid division by zero
        corr = cov / np.outer(d, d)

        # Clamp to [-1, 1] (numerical precision)
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, -1.0, 1.0)

        return corr, shrink_coef
    else:
        corr = returns.corr().values
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, -1.0, 1.0)
        return corr, 0.0


def compute_normal_correlation(
    returns: pd.DataFrame,
    tickers: list[str],
    window: int = 252,
) -> CorrelationResult:
    """
    Compute normal-regime correlation matrix from trailing window.
    """
    # Filter to requested tickers (+ any reference hedges present)
    available = [t for t in tickers if t in returns.columns]
    if len(available) < 2:
        raise ValueError(f"Need at least 2 tickers with data, got {len(available)}")

    subset = returns[available].tail(window).dropna()
    use_shrinkage = len(available) > SHRINKAGE_NORMAL_THRESHOLD

    corr, shrink_coef = _correlation_from_returns(subset, use_shrinkage)

    logger.info(
        f"Normal correlation: {len(available)} tickers, {len(subset)} days"
        + (f", shrinkage={shrink_coef:.4f}" if use_shrinkage else "")
    )

    return CorrelationResult(
        matrix=corr,
        tickers=available,
        regime="normal",
        observation_days=len(subset),
        shrinkage_applied=use_shrinkage,
        shrinkage_coefficient=shrink_coef,
    )


def compute_stress_correlation(
    stress_returns: pd.DataFrame,
    tickers: list[str],
) -> CorrelationResult:
    """
    Compute stress-regime correlation matrix.
    Always applies Ledoit-Wolf shrinkage due to smaller sample size.
    """
    available = [t for t in tickers if t in stress_returns.columns]
    if len(available) < 2:
        raise ValueError(f"Need at least 2 tickers with stress data, got {len(available)}")

    subset = stress_returns[available].dropna()

    if len(subset) < 20:
        raise ValueError(
            f"Only {len(subset)} stress observations — too few for stable estimate. "
            f"Consider lowering the drawdown threshold."
        )

    use_shrinkage = SHRINKAGE_ALWAYS_STRESS
    corr, shrink_coef = _correlation_from_returns(subset, use_shrinkage)

    logger.info(
        f"Stress correlation: {len(available)} tickers, {len(subset)} days, "
        f"shrinkage={shrink_coef:.4f}"
    )

    return CorrelationResult(
        matrix=corr,
        tickers=available,
        regime="stress",
        observation_days=len(subset),
        shrinkage_applied=use_shrinkage,
        shrinkage_coefficient=shrink_coef,
    )


def compute_blended_correlation(
    normal: CorrelationResult,
    stress: CorrelationResult,
    lam: float = DEFAULT_LAMBDA,
) -> CorrelationResult:
    """
    Blend normal and stress correlation matrices.
    Σ_blend = (1 - λ) × Σ_normal + λ × Σ_stress

    Matrices must share the same ticker ordering.
    """
    if normal.tickers != stress.tickers:
        raise ValueError("Ticker ordering mismatch between normal and stress matrices")

    blended = (1 - lam) * normal.matrix + lam * stress.matrix

    # Re-enforce diagonal = 1 and symmetry
    np.fill_diagonal(blended, 1.0)
    blended = (blended + blended.T) / 2
    blended = np.clip(blended, -1.0, 1.0)

    return CorrelationResult(
        matrix=blended,
        tickers=normal.tickers,
        regime="blended",
        observation_days=normal.observation_days,  # informational
        shrinkage_applied=normal.shrinkage_applied or stress.shrinkage_applied,
    )


def lambda_from_risk_score(risk_score: int) -> float:
    """
    Map investor risk score (0-100) to stress blending weight λ.
    Score 0   → λ = 0.45 (conservative, heavy stress weight)
    Score 100 → λ = 0.20 (aggressive, lighter stress weight)
    """
    score = max(0, min(100, risk_score))
    return LAMBDA_BASE - (score / 100) * LAMBDA_RANGE


def build_covariance_matrix(
    normal_returns: pd.DataFrame,
    stress_returns: pd.DataFrame,
    tickers: list[str],
    lam: float = DEFAULT_LAMBDA,
    window: int = 252,
) -> CovarianceResult:
    """
    Build the full blended covariance matrix ready for optimization.

    Σ_cov = D_blend × Σ_corr_blend × D_blend

    where D_blend is a diagonal matrix of blended volatilities:
    σ_blend = (1 - λ) × σ_normal + λ × σ_stress
    """
    # Compute both correlation matrices
    normal_corr = compute_normal_correlation(normal_returns, tickers, window)
    stress_corr = compute_stress_correlation(stress_returns, tickers)

    # Align tickers — use only those present in both
    common = [t for t in normal_corr.tickers if t in stress_corr.tickers]
    if len(common) < 2:
        raise ValueError("Fewer than 2 tickers overlap between normal and stress data")

    # Re-index matrices to common tickers
    normal_idx = [normal_corr.tickers.index(t) for t in common]
    stress_idx = [stress_corr.tickers.index(t) for t in common]

    n_mat = normal_corr.matrix[np.ix_(normal_idx, normal_idx)]
    s_mat = stress_corr.matrix[np.ix_(stress_idx, stress_idx)]

    normal_corr_aligned = CorrelationResult(
        matrix=n_mat, tickers=common, regime="normal",
        observation_days=normal_corr.observation_days,
        shrinkage_applied=normal_corr.shrinkage_applied,
    )
    stress_corr_aligned = CorrelationResult(
        matrix=s_mat, tickers=common, regime="stress",
        observation_days=stress_corr.observation_days,
        shrinkage_applied=stress_corr.shrinkage_applied,
    )

    # Blend correlation
    blended_corr = compute_blended_correlation(normal_corr_aligned, stress_corr_aligned, lam)

    # Compute volatilities per regime
    normal_tail = normal_returns[common].tail(window).dropna()
    vol_normal = (normal_tail.std() * np.sqrt(252)).values
    vol_stress = (stress_returns[common].dropna().std() * np.sqrt(252)).values

    # Blend volatilities
    vol_blended = (1 - lam) * vol_normal + lam * vol_stress

    # Build covariance: Σ = D × R × D
    D = np.diag(vol_blended)
    cov_matrix = D @ blended_corr.matrix @ D

    # Enforce symmetry
    cov_matrix = (cov_matrix + cov_matrix.T) / 2

    logger.info(
        f"Covariance matrix: {len(common)} tickers, λ={lam:.2f}, "
        f"avg vol normal={np.mean(vol_normal):.4f}, stress={np.mean(vol_stress):.4f}"
    )

    return CovarianceResult(
        matrix=cov_matrix,
        tickers=common,
        lambda_used=lam,
        vol_normal=vol_normal,
        vol_stress=vol_stress,
    )


def compute_diagnostics(
    normal_corr: CorrelationResult,
    stress_corr: CorrelationResult,
    portfolio_tickers: list[str],
    weights: np.ndarray | None = None,
) -> CorrelationDiagnostics:
    """
    Compute summary diagnostics for the frontend correlation view.
    """
    n = len(portfolio_tickers)

    # Average pairwise correlation (upper triangle only, excluding diagonal)
    def _avg_upper(mat):
        upper = mat[np.triu_indices(n, k=1)]
        return float(np.mean(upper)) if len(upper) > 0 else 0.0

    # Filter to portfolio tickers only (exclude reference hedges for avg calc)
    port_idx_n = [normal_corr.tickers.index(t) for t in portfolio_tickers if t in normal_corr.tickers]
    port_idx_s = [stress_corr.tickers.index(t) for t in portfolio_tickers if t in stress_corr.tickers]

    n_mat = normal_corr.matrix[np.ix_(port_idx_n, port_idx_n)]
    s_mat = stress_corr.matrix[np.ix_(port_idx_s, port_idx_s)]

    avg_n = _avg_upper(n_mat)
    avg_s = _avg_upper(s_mat)
    spike = ((avg_s - avg_n) / max(abs(avg_n), 0.001)) * 100

    # High-correlation pairs
    tickers_n = [t for t in portfolio_tickers if t in normal_corr.tickers]
    tickers_s = [t for t in portfolio_tickers if t in stress_corr.tickers]

    high_normal = []
    for i in range(len(tickers_n)):
        for j in range(i + 1, len(tickers_n)):
            val = n_mat[i, j]
            if val > HIGH_CORR_NORMAL_THRESHOLD:
                high_normal.append((tickers_n[i], tickers_n[j], round(float(val), 3)))

    high_stress = []
    hedging = []
    for i in range(len(tickers_s)):
        for j in range(i + 1, len(tickers_s)):
            val = s_mat[i, j]
            if val > HIGH_CORR_THRESHOLD:
                high_stress.append((tickers_s[i], tickers_s[j], round(float(val), 3)))
            if val < NEGATIVE_CORR_THRESHOLD:
                hedging.append((tickers_s[i], tickers_s[j], round(float(val), 3)))

    # Diversification ratio: (Σ w_i σ_i) / σ_portfolio
    # Higher = more diversification benefit
    # (simplified — uses equal weights if none provided)
    def _div_ratio(corr_mat, vols):
        if weights is not None:
            w = weights
        else:
            w = np.ones(len(vols)) / len(vols)
        weighted_vol_sum = np.dot(w, vols)
        port_var = w @ (np.diag(vols) @ corr_mat @ np.diag(vols)) @ w
        port_vol = np.sqrt(max(port_var, 0))
        return float(weighted_vol_sum / max(port_vol, 1e-10))

    # Reference hedge correlations (avg correlation of each hedge with portfolio)
    ref_corrs = {}
    for hedge in REFERENCE_HEDGES:
        if hedge in normal_corr.tickers and hedge not in portfolio_tickers:
            h_idx = normal_corr.tickers.index(hedge)
            corrs_with_port = [
                normal_corr.matrix[h_idx, normal_corr.tickers.index(t)]
                for t in portfolio_tickers
                if t in normal_corr.tickers
            ]
            if corrs_with_port:
                ref_corrs[hedge] = {
                    "avg_corr_normal": round(float(np.mean(corrs_with_port)), 3),
                }
                # Add stress correlation if available
                if hedge in stress_corr.tickers:
                    h_idx_s = stress_corr.tickers.index(hedge)
                    corrs_stress = [
                        stress_corr.matrix[h_idx_s, stress_corr.tickers.index(t)]
                        for t in portfolio_tickers
                        if t in stress_corr.tickers
                    ]
                    if corrs_stress:
                        ref_corrs[hedge]["avg_corr_stress"] = round(float(np.mean(corrs_stress)), 3)

    return CorrelationDiagnostics(
        avg_corr_normal=round(avg_n, 4),
        avg_corr_stress=round(avg_s, 4),
        corr_spike_pct=round(spike, 1),
        high_corr_pairs_normal=high_normal,
        high_corr_pairs_stress=high_stress,
        hedging_pairs=hedging,
        reference_correlations=ref_corrs,
    )
