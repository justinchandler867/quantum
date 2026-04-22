"""
Portfolio Optimizer
Scipy SLSQP optimization using the blended covariance matrix from the
correlation engine. Supports four objectives with profile-based constraints.
"""
import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.optimize import minimize

from app.config import RISK_FREE_RATE

logger = logging.getLogger(__name__)


class Objective(str, Enum):
    MAX_SHARPE = "max_sharpe"
    MAX_RETURN = "max_return"
    MIN_VOLATILITY = "min_vol"
    RISK_PARITY = "risk_parity"
    MAX_DIVERSIFICATION = "max_diversification"


@dataclass
class ProfileConstraints:
    """Investor profile constraints for the optimizer."""
    max_position_pct: float = 0.30        # max weight per asset (0-1)
    min_position_pct: float = 0.02        # min weight if included (0-1)
    max_beta: float = 1.5                 # portfolio-level beta cap
    max_volatility: float | None = None   # portfolio-level vol cap (annualized)
    excluded_tickers: list[str] = field(default_factory=list)  # ineligible assets


@dataclass
class OptimizationResult:
    """Full result of a portfolio optimization."""
    weights: dict[str, float]             # ticker → weight (0-1)
    objective: str
    portfolio_return: float               # annualized
    portfolio_volatility: float           # annualized
    sharpe_ratio: float
    beta: float
    var_95: float                         # daily parametric VaR
    cvar_95: float
    risk_contributions: dict[str, float]  # ticker → % of total risk
    solver_success: bool
    solver_message: str
    iterations: int
    constraints_active: list[str]         # which constraints were binding


def _portfolio_return(weights: np.ndarray, expected_returns: np.ndarray) -> float:
    """Portfolio expected return: w · μ"""
    return float(weights @ expected_returns)


def _portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Portfolio volatility: √(wᵀΣw)"""
    var = weights @ cov_matrix @ weights
    return float(np.sqrt(max(var, 0)))


def _portfolio_sharpe(weights: np.ndarray, expected_returns: np.ndarray,
                      cov_matrix: np.ndarray, rf: float = RISK_FREE_RATE) -> float:
    """Sharpe ratio: (Rp - Rf) / σp"""
    ret = _portfolio_return(weights, expected_returns)
    vol = _portfolio_volatility(weights, cov_matrix)
    if vol < 1e-10:
        return 0.0
    return (ret - rf) / vol


def _risk_contributions(weights: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
    """
    Marginal risk contribution per asset.
    RC_i = w_i × (Σw)_i / σ_p
    Returns array of fractional contributions (sum to 1.0).
    """
    port_vol = _portfolio_volatility(weights, cov_matrix)
    if port_vol < 1e-10:
        return np.ones(len(weights)) / len(weights)
    marginal = cov_matrix @ weights
    rc = weights * marginal / port_vol
    total = rc.sum()
    if total < 1e-10:
        return np.ones(len(weights)) / len(weights)
    return rc / total


# ── Objective Functions (all minimized, so negate for maximization) ───────────

def _neg_sharpe(weights, expected_returns, cov_matrix, rf):
    """Negative Sharpe — minimize this to maximize Sharpe."""
    return -_portfolio_sharpe(weights, expected_returns, cov_matrix, rf)


def _neg_return(weights, expected_returns, cov_matrix, rf):
    """Negative return — minimize this to maximize return."""
    return -_portfolio_return(weights, expected_returns)


def _volatility(weights, expected_returns, cov_matrix, rf):
    """Portfolio volatility — minimize directly."""
    return _portfolio_volatility(weights, cov_matrix)


def _risk_parity_objective(weights, expected_returns, cov_matrix, rf):
    """
    Risk parity: minimize the sum of squared differences between
    each asset's risk contribution and the target (equal) contribution.

    Target: each asset contributes 1/n of total risk.
    """
    n = len(weights)
    target = 1.0 / n
    rc = _risk_contributions(weights, cov_matrix)
    return float(np.sum((rc - target) ** 2))


def _neg_diversification_ratio(weights, expected_returns, cov_matrix, rf):
    """
    Negative diversification ratio — minimize to maximize DR.
    DR(w) = (w · σ) / sqrt(wᵀ Σ w)
    where σ is the vector of individual asset volatilities
    (sqrt of diagonal of cov_matrix). DR ≥ 1, with higher values
    meaning more diversification benefit.
    """
    sigma = np.sqrt(np.maximum(np.diag(cov_matrix), 0))
    weighted_avg_vol = float(weights @ sigma)
    port_vol = _portfolio_volatility(weights, cov_matrix)
    if port_vol < 1e-10:
        return 0.0
    return -weighted_avg_vol / port_vol


_OBJECTIVE_FNS = {
    Objective.MAX_SHARPE: _neg_sharpe,
    Objective.MAX_RETURN: _neg_return,
    Objective.MIN_VOLATILITY: _volatility,
    Objective.RISK_PARITY: _risk_parity_objective,
    Objective.MAX_DIVERSIFICATION: _neg_diversification_ratio,
}


def optimize_portfolio(
    tickers: list[str],
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    betas: np.ndarray,
    objective: Objective = Objective.MAX_SHARPE,
    constraints: ProfileConstraints | None = None,
    rf: float = RISK_FREE_RATE,
) -> OptimizationResult:
    """
    Run constrained portfolio optimization using scipy SLSQP.

    Args:
        tickers: Ticker symbols (same order as matrix axes)
        expected_returns: Annualized expected return per asset (n,)
        cov_matrix: n×n annualized covariance matrix (from blended engine)
        betas: Beta per asset (n,)
        objective: Optimization objective
        constraints: Investor profile constraints
        rf: Risk-free rate

    Returns:
        OptimizationResult with optimal weights and portfolio metrics
    """
    if constraints is None:
        constraints = ProfileConstraints()

    n = len(tickers)
    assert expected_returns.shape == (n,), f"Returns shape mismatch: {expected_returns.shape} vs ({n},)"
    assert cov_matrix.shape == (n, n), f"Cov matrix shape mismatch: {cov_matrix.shape} vs ({n},{n})"
    assert betas.shape == (n,), f"Betas shape mismatch: {betas.shape} vs ({n},)"

    # ── Bounds: per-asset min/max weights ────────────────────────────────────
    bounds = []
    for t in tickers:
        if t in constraints.excluded_tickers:
            bounds.append((0.0, 0.0))  # force to zero
        else:
            bounds.append((constraints.min_position_pct, constraints.max_position_pct))

    # ── Constraints ──────────────────────────────────────────────────────────
    scipy_constraints = []
    active_constraints = []

    # Weights sum to 1 (equality constraint)
    scipy_constraints.append({
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0,
    })

    # Portfolio beta cap (inequality: max_beta - w·β ≥ 0)
    if constraints.max_beta is not None:
        scipy_constraints.append({
            "type": "ineq",
            "fun": lambda w, b=betas, mb=constraints.max_beta: mb - w @ b,
        })

    # Portfolio volatility cap
    if constraints.max_volatility is not None:
        scipy_constraints.append({
            "type": "ineq",
            "fun": lambda w, S=cov_matrix, mv=constraints.max_volatility: (
                mv - np.sqrt(max(w @ S @ w, 0))
            ),
        })

    # ── Initial guess: equal weight (feasible) ──────────────────────────────
    eligible = [i for i, t in enumerate(tickers) if t not in constraints.excluded_tickers]
    x0 = np.zeros(n)
    if eligible:
        w_each = min(1.0 / len(eligible), constraints.max_position_pct)
        for i in eligible:
            x0[i] = w_each
        # Normalize to sum to 1
        x0 = x0 / x0.sum()

    # ── Solve ────────────────────────────────────────────────────────────────
    obj_fn = _OBJECTIVE_FNS[objective]

    result = minimize(
        obj_fn,
        x0,
        args=(expected_returns, cov_matrix, rf),
        method="SLSQP",
        bounds=bounds,
        constraints=scipy_constraints,
        options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
    )

    if not result.success:
        logger.warning(f"Optimizer did not converge: {result.message}. Using best found solution.")

    w_opt = result.x

    # Clean up: zero out negligible weights, re-normalize
    w_opt[w_opt < 0.005] = 0.0
    w_sum = w_opt.sum()
    if w_sum > 0:
        w_opt = w_opt / w_sum
    else:
        # Fallback to equal weight if optimizer failed completely
        w_opt = x0
        logger.error("Optimizer produced all-zero weights, falling back to equal weight")

    # ── Compute portfolio metrics ────────────────────────────────────────────
    port_ret = _portfolio_return(w_opt, expected_returns)
    port_vol = _portfolio_volatility(w_opt, cov_matrix)
    port_sharpe = _portfolio_sharpe(w_opt, expected_returns, cov_matrix, rf)
    port_beta = float(w_opt @ betas)

    # Parametric VaR/CVaR (daily)
    daily_ret = port_ret / 252
    daily_vol = port_vol / np.sqrt(252)
    var_95 = -(daily_ret - 1.645 * daily_vol)
    cvar_95 = -(daily_ret - 2.063 * daily_vol)

    # Risk contributions
    rc = _risk_contributions(w_opt, cov_matrix)
    risk_contribs = {tickers[i]: round(float(rc[i]) * 100, 2) for i in range(n) if w_opt[i] > 0}

    # Check which constraints are binding
    if constraints.max_beta and abs(port_beta - constraints.max_beta) < 0.01:
        active_constraints.append(f"beta_cap ({constraints.max_beta})")
    if constraints.max_volatility and abs(port_vol - constraints.max_volatility) < 0.005:
        active_constraints.append(f"vol_cap ({constraints.max_volatility:.1%})")

    at_max = [tickers[i] for i in range(n) if abs(w_opt[i] - constraints.max_position_pct) < 0.005]
    if at_max:
        active_constraints.append(f"max_position ({constraints.max_position_pct:.0%}): {', '.join(at_max)}")

    at_min = [tickers[i] for i in range(n)
              if 0 < w_opt[i] < constraints.min_position_pct + 0.005 and w_opt[i] > 0.005]
    if at_min:
        active_constraints.append(f"min_position ({constraints.min_position_pct:.0%}): {', '.join(at_min)}")

    # Build weights dict (only non-zero)
    weights_dict = {}
    for i, t in enumerate(tickers):
        if w_opt[i] > 0.005:
            weights_dict[t] = round(float(w_opt[i]), 4)

    logger.info(
        f"Optimization complete: {objective.value}, {len(weights_dict)} positions, "
        f"Sharpe={port_sharpe:.3f}, Return={port_ret:.2%}, Vol={port_vol:.2%}, "
        f"Beta={port_beta:.2f}, converged={result.success}"
    )

    return OptimizationResult(
        weights=weights_dict,
        objective=objective.value,
        portfolio_return=round(port_ret, 6),
        portfolio_volatility=round(port_vol, 6),
        sharpe_ratio=round(port_sharpe, 4),
        beta=round(port_beta, 4),
        var_95=round(var_95, 6),
        cvar_95=round(cvar_95, 6),
        risk_contributions=risk_contribs,
        solver_success=result.success,
        solver_message=result.message,
        iterations=result.nit,
        constraints_active=active_constraints,
    )


def optimize_multi_objective(
    tickers: list[str],
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    betas: np.ndarray,
    constraints: ProfileConstraints | None = None,
    rf: float = RISK_FREE_RATE,
) -> dict[str, OptimizationResult]:
    """
    Run all four objectives and return results keyed by objective name.
    Useful for the Compare view and efficient frontier endpoints.
    """
    results = {}
    for obj in Objective:
        try:
            results[obj.value] = optimize_portfolio(
                tickers, expected_returns, cov_matrix, betas,
                objective=obj, constraints=constraints, rf=rf,
            )
        except Exception as e:
            logger.error(f"Optimization failed for {obj.value}: {e}")
    return results


def generate_efficient_frontier(
    tickers: list[str],
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    betas: np.ndarray,
    constraints: ProfileConstraints | None = None,
    n_points: int = 50,
    rf: float = RISK_FREE_RATE,
) -> list[dict]:
    """
    Trace the efficient frontier by optimizing for minimum volatility
    at each target return level.

    Returns a list of {return, volatility, sharpe, weights} points.
    """
    if constraints is None:
        constraints = ProfileConstraints()

    n = len(tickers)

    # First find the min-vol and max-return endpoints
    min_vol_result = optimize_portfolio(
        tickers, expected_returns, cov_matrix, betas,
        Objective.MIN_VOLATILITY, constraints, rf,
    )
    max_ret_result = optimize_portfolio(
        tickers, expected_returns, cov_matrix, betas,
        Objective.MAX_RETURN, constraints, rf,
    )

    ret_min = min_vol_result.portfolio_return
    ret_max = max_ret_result.portfolio_return

    if ret_max - ret_min < 0.001:
        # Degenerate case: all assets have similar returns
        return [{
            "return": min_vol_result.portfolio_return,
            "volatility": min_vol_result.portfolio_volatility,
            "sharpe": min_vol_result.sharpe_ratio,
        }]

    # Generate target return levels
    target_returns = np.linspace(ret_min, ret_max, n_points)

    frontier = []
    eligible = [i for i, t in enumerate(tickers) if t not in constraints.excluded_tickers]
    x0 = np.zeros(n)
    if eligible:
        w_each = min(1.0 / len(eligible), constraints.max_position_pct)
        for i in eligible:
            x0[i] = w_each
        x0 = x0 / x0.sum()

    for target_ret in target_returns:
        bounds = []
        for t in tickers:
            if t in constraints.excluded_tickers:
                bounds.append((0.0, 0.0))
            else:
                bounds.append((constraints.min_position_pct, constraints.max_position_pct))

        scipy_constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, tr=target_ret: w @ expected_returns - tr},
        ]

        if constraints.max_beta is not None:
            scipy_constraints.append({
                "type": "ineq",
                "fun": lambda w, b=betas, mb=constraints.max_beta: mb - w @ b,
            })

        result = minimize(
            lambda w: _portfolio_volatility(w, cov_matrix),
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=scipy_constraints,
            options={"maxiter": 500, "ftol": 1e-10, "disp": False},
        )

        if result.success or result.fun < 999:
            w = result.x
            vol = _portfolio_volatility(w, cov_matrix)
            ret = _portfolio_return(w, expected_returns)
            sh = (ret - rf) / vol if vol > 1e-10 else 0.0

            frontier.append({
                "return": round(ret, 6),
                "volatility": round(vol, 6),
                "sharpe": round(sh, 4),
            })

    logger.info(f"Efficient frontier: {len(frontier)} points from {ret_min:.2%} to {ret_max:.2%}")
    return frontier
