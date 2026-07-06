"""
Return models — Black-Litterman-lite equilibrium returns + Ledoit-Wolf
constant-correlation shrinkage covariance.

Two orthogonal knobs feed the optimizer:

  * return_model = "historical"       -> sample covariance, historical means
                                         (the regression anchor; UNCHANGED).
  * return_model = "black_litterman"  -> shrunk covariance + market-implied
                                         equilibrium prior, with the Co-CIO
                                         Outlook sector tilts recast as VIEWS.

This module is pure numpy: `ledoit_wolf_constant_correlation` and
`black_litterman`. No I/O, no yfinance, no global state — the endpoint layer
sources the returns window and market caps and hands them in. That keeps the
sealed audit (tests/test_black_litterman.py) network-free.

References
----------
Ledoit, O. & Wolf, M. (2004). "Honey, I Shrunk the Sample Covariance Matrix."
    Journal of Portfolio Management, 30(4). Constant-correlation target.
Black, F. & Litterman, R. (1992). "Global Portfolio Optimization."
    Financial Analysts Journal.
"""
from __future__ import annotations

import numpy as np

TRADING_DAYS = 252


def _as_matrix(returns) -> np.ndarray:
    """Coerce a returns table (DataFrame or array) to a dense (T, N) float array."""
    if hasattr(returns, "values"):
        returns = returns.values
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"returns must be 2-D (T, N); got shape {x.shape}")
    return x


def ledoit_wolf_constant_correlation(
    returns,
    annualize: bool = True,
    periods_per_year: int = TRADING_DAYS,
) -> tuple[np.ndarray, float, int]:
    """
    Ledoit-Wolf (2004) shrinkage toward the constant-correlation target.

    Target F:  F_ii = S_ii,  F_ij = rbar * sqrt(S_ii S_jj),
    where rbar is the mean of the sample pairwise correlations. The shrinkage
    intensity alpha in [0, 1] is the closed-form optimal constant
    delta* = kappa / T, kappa = (pi - rho) / gamma, clipped to [0, 1]:

        Sigma_shrunk = (1 - alpha) * S + alpha * F

    S is the MLE sample covariance (divisor T, on demeaned returns).

    Args:
        returns: (T, N) periodic (daily) returns — DataFrame or ndarray.
        annualize: multiply the shrunk covariance by `periods_per_year`.
        periods_per_year: annualization factor (252 trading days).

    Returns:
        (Sigma_shrunk, alpha, n_days)
          Sigma_shrunk : (N, N) shrunk covariance, annualized if requested.
          alpha        : shrinkage intensity in [0, 1].
          n_days       : number of observations T used.

    Notes:
        * alpha is scale-invariant, so it is computed on the raw (daily) S and
          F; annualization is applied afterwards and does not change alpha.
        * For N == 2 the target equals the sample matrix identically (a single
          off-diagonal correlation means rbar == r_12), so the estimator is a
          provable no-op: Sigma_shrunk == S. See Case D1.
    """
    x = _as_matrix(returns)
    t, n = x.shape
    if t < 2:
        raise ValueError(f"need at least 2 observations; got {t}")
    if n < 1:
        raise ValueError("need at least 1 asset")

    # Demeaned returns.
    x = x - x.mean(axis=0, keepdims=True)

    # Sample covariance (MLE, divisor T).
    s = (x.T @ x) / t
    var = np.diag(s).copy()
    std = np.sqrt(var)
    outer_std = np.outer(std, std)

    # Constant-correlation target.
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(outer_std > 0, s / outer_std, 0.0)
    if n > 1:
        rbar = (corr.sum() - n) / (n * (n - 1))
    else:
        rbar = 0.0
    f = rbar * outer_std
    np.fill_diagonal(f, var)

    # gamma = ||F - S||_F^2 (misspecification of the target).
    gamma = float(np.sum((f - s) ** 2))

    if gamma <= 1e-18:
        # Target coincides with the sample matrix (e.g. N == 2). No-op shrink.
        alpha = 0.0
    else:
        # pi-hat: sum of asymptotic variances of the sample-covariance entries.
        x2 = x ** 2
        phi_mat = (x2.T @ x2) / t - s ** 2
        pi_hat = float(phi_mat.sum())

        # rho-hat: pi diagonal + cross terms between S and the target's rbar.
        term1 = ((x ** 3).T @ x) / t              # E[y_i^3 y_j]
        theta = term1 - var[:, None] * s          # theta_{ii,ij}
        np.fill_diagonal(theta, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            mult = np.where(std[:, None] > 0, std[None, :] / std[:, None], 0.0)
        rho_hat = float(np.diag(phi_mat).sum() + rbar * np.sum(mult * theta))

        kappa = (pi_hat - rho_hat) / gamma
        alpha = float(max(0.0, min(1.0, kappa / t)))

    sigma_shrunk = alpha * f + (1.0 - alpha) * s
    # Numerical symmetry hygiene.
    sigma_shrunk = (sigma_shrunk + sigma_shrunk.T) / 2.0

    if annualize:
        sigma_shrunk = sigma_shrunk * periods_per_year

    return sigma_shrunk, alpha, int(t)


def equilibrium_prior(
    sigma: np.ndarray,
    mkt_caps: np.ndarray,
    delta: float = 2.5,
) -> np.ndarray:
    """
    Reverse-optimization equilibrium prior: pi = delta * Sigma * w_mkt,
    where w_mkt is the market-cap weight vector (normalized to sum to 1).
    """
    sigma = np.asarray(sigma, dtype=float)
    caps = np.asarray(mkt_caps, dtype=float)
    total = caps.sum()
    if total <= 0:
        raise ValueError("market caps must sum to a positive value")
    w_mkt = caps / total
    return delta * (sigma @ w_mkt)


def black_litterman(
    sigma: np.ndarray,
    mkt_caps: np.ndarray,
    tilts: np.ndarray,
    view_confidence: float,
    tau: float = 0.05,
    delta: float = 2.5,
) -> np.ndarray:
    """
    Black-Litterman-lite posterior expected returns.

    Equilibrium prior:
        pi = delta * Sigma * w_mkt

    Views: for each asset i whose tilt t_i is nonzero, one ABSOLUTE view
        Q_i = pi_i + t_i
    with P the row-selector over tilted assets. The view-uncertainty is scaled
    by the conviction c = view_confidence:
        Omega = (1/c) * P (tau Sigma) P^T

    Posterior (standard update form):
        mu_BL = pi + tau Sigma P^T (P tau Sigma P^T + Omega)^{-1} (Q - P pi)

    Consequences (proved in the sealed audit):
      (1) Views on ALL assets  ->  mu_BL = pi + [c/(1+c)] * t   exactly.
      (2) tau cancels identically in this Omega parameterization — the posterior
          is tau-invariant.
      (3) An untilted asset j spills over by beta_ji * realized_tilt_i,
          beta_ji = Sigma_ji / Sigma_ii.

    Args:
        sigma: (N, N) covariance (annualized shrunk matrix).
        mkt_caps: (N,) market caps (effective size). Normalized internally.
        tilts: (N,) per-asset absolute view tilt; 0 where the sector is untilted.
        view_confidence: c > 0. Higher = tighter views = more pass-through.
        tau, delta: BL scalars (documented constants).

    Returns:
        mu_BL: (N,) posterior expected returns. Equals pi exactly when no asset
        is tilted (empty view set).
    """
    sigma = np.asarray(sigma, dtype=float)
    tilts = np.asarray(tilts, dtype=float)
    n = sigma.shape[0]

    pi = equilibrium_prior(sigma, mkt_caps, delta)

    if view_confidence <= 0:
        raise ValueError("view_confidence must be > 0")

    view_idx = np.flatnonzero(np.abs(tilts) > 0)
    if view_idx.size == 0:
        # No views -> posterior collapses to the equilibrium prior exactly.
        return pi

    # P: selector over tilted assets (k x n).
    k = view_idx.size
    p = np.zeros((k, n))
    p[np.arange(k), view_idx] = 1.0

    # Absolute views Q_i = pi_i + t_i  ->  innovation (Q - P pi) = t restricted.
    innovation = tilts[view_idx]

    tau_sigma = tau * sigma
    p_ts_pt = p @ tau_sigma @ p.T            # (k x k)
    omega = (1.0 / view_confidence) * p_ts_pt
    middle = p_ts_pt + omega                 # = (1 + 1/c) * P tau Sigma P^T

    adjustment = tau_sigma @ p.T @ np.linalg.solve(middle, innovation)
    return pi + adjustment
