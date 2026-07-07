"""
2026 Co-CIO Outlook — sector-level expected-return tilts for the optimizer.

When the user toggles "2026 Co-CIO Outlook" mode, each ticker's expected
return gets adjusted by its sector's tilt before optimization runs. Same
constraints, same objectives — only the input μ vector changes.

These tilts encode a curated thesis. They are NOT investment advice; the UI
should surface them alongside the source attribution so users treat them as
one view among many. When the thesis changes, update SECTOR_TILTS_2026 and
OUTLOOK_METADATA.last_updated together.
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np


# Annualized expected-return adjustment per sector (decimal).
# +0.015 = +1.5 percentage points, -0.010 = -1.0 pp.
# Sectors not listed here get no adjustment.
SECTOR_TILTS_2026: dict[str, float] = {
    "Technology":          -0.010,
    "Financial Services":  +0.015,
    "Healthcare":          +0.010,
    "Industrials":         +0.005,
    "Consumer Cyclical":   -0.005,
    "Consumer Defensive":  +0.005,
    "Fixed Income":        +0.005,
}


OUTLOOK_METADATA = {
    "name": "2026 Co-CIO Outlook",
    "source": (
        "Based on commentary from Fort Washington Co-CIOs "
        "Christopher Shipley & Brendan White, 2025-2026"
    ),
    "thesis_summary": (
        "Late-cycle posture with a defensive rotation. Financials carry "
        "the largest upward tilt as banks benefit from the resettled rate "
        "environment, with healthcare close behind on structural demand. "
        "Technology and consumer cyclical are tilted down on extended "
        "multiples and softening consumer spending. Fixed income gets a "
        "modest upward tilt after the 2022-2024 yield reset made bonds "
        "reasonable again on a real-yield basis."
    ),
    "last_updated": "2026-05",
    "tilts": SECTOR_TILTS_2026,
}


def apply_outlook_tilts(
    expected_returns: np.ndarray,
    tickers: list[str],
    sectors: dict[str, str | None],
    tilts: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Apply sector-level return tilts to a vector of expected returns.

    Pure function — does not mutate `expected_returns`. Tickers with no
    sector tag (or whose sector has no tilt) receive no adjustment.

    Args:
        expected_returns: Annualized expected returns aligned with `tickers` (n,).
        tickers: Ticker symbols in the same order as `expected_returns`.
        sectors: {ticker: sector_string or None}.
        tilts: Sector → tilt mapping. Defaults to SECTOR_TILTS_2026.

    Returns:
        New ndarray with tilts applied.
    """
    if tilts is None:
        tilts = SECTOR_TILTS_2026

    out = expected_returns.astype(float, copy=True)
    for i, t in enumerate(tickers):
        sec = sectors.get(t)
        if sec is None:
            continue
        delta = tilts.get(sec, 0.0)
        if delta:
            out[i] += delta
    return out


# ── Full regime stress-test parameters ────────────────────────────────────────
#
# The regime is a forward-looking scenario, not a historical fit. Returns get
# the SECTOR_TILTS_2026 nudges; volatilities are scaled per sector; and the
# stock-bond correlation is overridden to reflect Shipley/White's commentary
# that the post-2022 regime has the two asset classes co-moving rather than
# diversifying. When all three combine, diversified portfolios should look
# meaningfully worse — that's the point of the stress test.

# The "EQUITY" token in a correlation override expands to this equity-sector
# list. Per spec: exclude Fixed Income, Commodity, Broad Market, International,
# Volatility, and Unknown. A pair whose two legs match {a-set, b-set} of an
# override takes that override's rho; all other correlations are preserved.
_EQUITY_TOKEN_SECTORS = frozenset({
    "Technology",
    "Financial Services",
    "Healthcare",
    "Industrials",
    "Energy",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Utilities",
    "Real Estate",
    "Communication Services",
    "Basic Materials",
})


def _expand_sector_token(token: str) -> frozenset:
    """"EQUITY" -> the equity-sector list; any other token -> just itself."""
    if token == "EQUITY":
        return _EQUITY_TOKEN_SECTORS
    return frozenset({token})


# The 2026 Co-CIO Outlook expressed as a regime instance: SECTOR_TILTS_2026 on
# returns, per-sector vol scaling, and the stock-bond correlation override recast
# as a single {Fixed Income × EQUITY -> 0.20} entry. Preset tilts are decimal
# (matching SECTOR_TILTS_2026); the sandbox API accepts tilts in percentage
# points and converts before building its regime.
REGIME_2026 = {
    "name": "2026 Co-CIO Outlook",
    "tilts": SECTOR_TILTS_2026,
    "vol_multipliers": {
        "Technology":             1.10,
        "Financial Services":     1.05,
        "Healthcare":             0.95,
        "Industrials":            1.00,
        "Energy":                 1.15,
        "Consumer Cyclical":      1.05,
        "Consumer Defensive":     0.95,
        "Utilities":              0.95,
        "Real Estate":            1.05,
        "Communication Services": 1.05,
        "Basic Materials":        1.05,
        "Fixed Income":           1.10,
        "Broad Market":           1.00,
        "International":          1.05,
        "Commodity":              1.10,
        "Volatility":             1.00,
        "Unknown":                1.00,
    },
    "correlation_overrides": [
        {"a": "Fixed Income", "b": "EQUITY", "rho": 0.20},
    ],
    "description": (
        "Equities and credit spreads under late-cycle pressure; "
        "stock-bond correlation positive (diversification weakened); "
        "energy vol elevated; defensives slightly damped."
    ),
}


def _psd_guard(
    corr: np.ndarray, tickers: list[str], requested: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Nearest-PSD repair of a (post-override) correlation matrix.

    If the min eigenvalue is < -1e-10, clip negative eigenvalues to 1e-8,
    reconstruct C' = V diag(λ_clipped) Vᵀ, and rescale to unit diagonal
    (D^-1/2 C' D^-1/2). Otherwise the matrix is returned UNCHANGED — no
    reconstruction — so a valid regime (e.g. the 2026 preset) stays byte-exact.

    Returns (corr_out, psd_adjustment) where psd_adjustment is
    {applied, max_correlation_adjustment, affected_pairs: top-5
     {tickers, requested, adjusted}} sorted by |adjustment| descending.
    """
    evals, evecs = np.linalg.eigh(corr)
    lam_min = float(evals.min())
    if lam_min >= -1e-10:
        return corr, {"applied": False, "max_correlation_adjustment": 0.0, "affected_pairs": []}

    clipped = np.clip(evals, 1e-8, None)
    recon = (evecs * clipped) @ evecs.T
    d = np.sqrt(np.clip(np.diag(recon), 1e-16, None))
    corr_out = recon / np.outer(d, d)
    np.fill_diagonal(corr_out, 1.0)
    corr_out = np.clip(corr_out, -1.0, 1.0)

    n = corr.shape[0]
    scored = []
    for i in range(n):
        for j in range(i + 1, n):
            scored.append((abs(corr_out[i, j] - requested[i, j]), i, j))
    scored.sort(reverse=True, key=lambda x: x[0])
    max_adj = scored[0][0] if scored else 0.0
    affected = [
        {
            "tickers": [tickers[i], tickers[j]],
            "requested": round(float(requested[i, j]), 6),
            "adjusted": round(float(corr_out[i, j]), 6),
        }
        for (adj, i, j) in scored[:5] if adj > 1e-12
    ]
    return corr_out, {
        "applied": True,
        "max_correlation_adjustment": round(float(max_adj), 6),
        "affected_pairs": affected,
    }


def apply_regime(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    tickers: list[str],
    sectors: dict[str, str | None],
    regime: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the full 2026 Co-CIO Outlook regime to (μ, Σ).

    Three transformations:
      1. Sector tilts on expected returns (regime['tilts']).
      2. Sector-specific vol scaling: σ'_i = σ_i × m(sector_i).
      3. Correlation overrides: each {a, b, rho} entry (with "EQUITY" expanded)
         replaces the correlation of every matching sector pair; the resulting
         matrix passes the PSD guard. All other correlations preserved.

    Reconstruction: Σ' = D' R' D' where D' = diag(σ'), R' is the modified
    (PSD-repaired) correlation matrix.

    Pure function — does not mutate inputs.
    """
    if regime is None:
        regime = REGIME_2026

    new_returns = apply_outlook_tilts(
        expected_returns, tickers, sectors, tilts=regime.get("tilts"))
    states = _regime_cov_states(cov_matrix, tickers, sectors, regime)
    return new_returns, states["Sigma_R"]


def _regime_cov_states(
    cov_matrix: np.ndarray,
    tickers: list[str],
    sectors: dict[str, str | None],
    regime: dict,
) -> dict:
    """
    Build the four covariance states used by attribution:

      Sigma_0 : the original covariance (returned verbatim, not reconstructed,
                so downstream vols/MC match the baseline exactly).
      Sigma_V : per-sector vol scaling applied, original correlations kept.
      Sigma_C : original vols, correlation overrides (PSD-repaired) applied.
      Sigma_R : both (== apply_regime's output).

    Also returns the decomposed pieces (σ vectors and ρ matrices) plus the
    psd_adjustment so the caller can surface the guard's outcome. The Shapley
    states are assembled from corr_C (the post-guard matrix), so attribution and
    the reported metrics share one correlation structure.
    """
    vols = np.sqrt(np.clip(np.diag(cov_matrix), 0.0, None))
    safe = np.where(vols > 1e-12, vols, 1e-12)
    corr = cov_matrix / np.outer(safe, safe)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)

    # Per-sector vol scaling.
    mults = regime.get("vol_multipliers") or {}
    new_vols = vols.copy()
    for i, t in enumerate(tickers):
        sec = sectors.get(t) or "Unknown"
        new_vols[i] *= mults.get(sec, 1.0)

    # Correlation overrides: each {a, b, rho} replaces every matching sector
    # pair's correlation ("EQUITY" expands to the equity-sector list).
    new_corr = corr.copy()
    n = len(tickers)
    secs = [sectors.get(t) or "Unknown" for t in tickers]
    for ov in (regime.get("correlation_overrides") or []):
        a_set = _expand_sector_token(ov["a"])
        b_set = _expand_sector_token(ov["b"])
        rho = float(ov["rho"])
        for i in range(n):
            for j in range(i + 1, n):
                si, sj = secs[i], secs[j]
                if (si in a_set and sj in b_set) or (si in b_set and sj in a_set):
                    new_corr[i, j] = rho
                    new_corr[j, i] = rho

    # PSD guard on the post-override correlation matrix (no-op when already PSD).
    corr_C, psd_adjustment = _psd_guard(new_corr, tickers, new_corr)

    return {
        "Sigma_0": cov_matrix,
        "Sigma_V": np.outer(new_vols, new_vols) * corr,
        "Sigma_C": np.outer(vols, vols) * corr_C,
        "Sigma_R": np.outer(new_vols, new_vols) * corr_C,
        "vols_0": vols,
        "vols_V": new_vols,
        "corr_0": corr,
        "corr_C": corr_C,
        "psd_adjustment": psd_adjustment,
    }


# ── Monte-Carlo max drawdown ───────────────────────────────────────────────────
# Centralized here so the stress endpoint AND the attribution waterfall share
# one implementation and one seed discipline — every MC state draws from the
# same RNG stream, so drawdown differences reflect the distribution change only.

def monte_carlo_max_drawdown(
    weights: np.ndarray,
    annual_mean: np.ndarray,
    annual_cov: np.ndarray,
    n_paths: int = 1000,
    days: int = 252,
    seed: int = 42,
) -> float:
    """
    Median max-drawdown across N simulated 1-year paths.

    Daily returns drawn from multivariate normal with (mean/252, cov/252).
    Portfolio log return per day = w · r; equity curve = exp(cumsum). Max
    drawdown per path = min((equity - running_max)/running_max). Returns the
    median (typical-case, not tail). Deterministic for a given seed.
    """
    rng = np.random.default_rng(seed)
    daily_mean = annual_mean / days
    daily_cov = annual_cov / days
    ridge = 1e-12 * np.eye(daily_cov.shape[0])
    samples = rng.multivariate_normal(daily_mean, daily_cov + ridge, size=(n_paths, days))
    port_daily = samples @ weights
    equity = np.empty((n_paths, days + 1))
    equity[:, 0] = 1.0
    equity[:, 1:] = np.exp(np.cumsum(port_daily, axis=1))
    run_max = np.maximum.accumulate(equity, axis=1)
    drawdown = (equity - run_max) / run_max
    return float(np.median(drawdown.min(axis=1)))


def _portfolio_vol(w: np.ndarray, sigma: np.ndarray) -> float:
    return float(np.sqrt(max(w @ sigma @ w, 0.0)))


def _variance_risk_contributions(w: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """RC_i = w_i (Σw)_i / σ_p² — variance shares that sum to 1."""
    denom = float(w @ sigma @ w)
    if denom <= 1e-18:
        return np.zeros_like(w)
    return (w * (sigma @ w)) / denom


def compute_attribution(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    weights: np.ndarray,
    tickers: list[str],
    sectors: dict[str, str | None],
    rf: float,
    regime: dict | None = None,
    n_paths: int = 1000,
    days: int = 252,
    seed: int = 42,
) -> tuple[dict, float, float]:
    """
    Decompose the baseline→regime deterioration into tilts (T) / vol scaling (V)
    / correlation flip (C), per metric and per position.

    Returns (attribution_dict, mdd_baseline, mdd_regime). The two MDD values are
    the [none] and [T,V,C] endpoints of the waterfall so the caller can reuse
    them as the response's baseline/regime drawdown — guaranteeing the MDD
    components sum exactly to the reported Δ.
    """
    if regime is None:
        regime = REGIME_2026

    w = np.asarray(weights, dtype=float)
    mu_0 = np.asarray(expected_returns, dtype=float)
    mu_R = apply_outlook_tilts(mu_0, tickers, sectors, tilts=regime.get("tilts"))
    tilt_vec = mu_R - mu_0
    n = len(tickers)

    st = _regime_cov_states(cov_matrix, tickers, sectors, regime)
    Sigma_0, Sigma_V, Sigma_C, Sigma_R = st["Sigma_0"], st["Sigma_V"], st["Sigma_C"], st["Sigma_R"]
    vols_0, vols_V, corr_0, corr_C = st["vols_0"], st["vols_V"], st["corr_0"], st["corr_C"]

    # ── Return: exact, 100% tilts (percentage points) ───────────────────────
    ret_tilts_pp = float(w @ tilt_vec) * 100.0

    # ── Volatility: Shapley over {V, C} (4 closed-form Σ evaluations) ────────
    s0 = _portfolio_vol(w, Sigma_0)
    sV = _portfolio_vol(w, Sigma_V)
    sC = _portfolio_vol(w, Sigma_C)
    sR = _portfolio_vol(w, Sigma_R)
    vol_scaling = ((sV - s0) + (sR - sC)) / 2.0
    vol_corr = ((sC - s0) + (sR - sV)) / 2.0

    # ── Sharpe: Shapley over {T, V, C} (2³ = 8 subset states) ───────────────
    def _sigma_of(subset: frozenset) -> np.ndarray:
        if "V" not in subset and "C" not in subset:
            return Sigma_0                       # original, matches baseline exactly
        vv = vols_V if "V" in subset else vols_0
        cc = corr_C if "C" in subset else corr_0
        return np.outer(vv, vv) * cc

    def _mu_of(subset: frozenset) -> np.ndarray:
        return mu_R if "T" in subset else mu_0

    _vcache: dict[frozenset, float] = {}

    def sharpe_v(subset: frozenset) -> float:
        if subset not in _vcache:
            sig = _portfolio_vol(w, _sigma_of(subset))
            ret = float(w @ _mu_of(subset))
            _vcache[subset] = ((ret - rf) / sig) if sig > 1e-12 else 0.0
        return _vcache[subset]

    levers = ("T", "V", "C")

    def shapley(lever: str) -> float:
        others = [l for l in levers if l != lever]
        total = 0.0
        for k in range(len(others) + 1):
            for combo in combinations(others, k):
                S = frozenset(combo)
                weight = (math.factorial(len(S)) * math.factorial(3 - len(S) - 1)
                          / math.factorial(3))
                total += weight * (sharpe_v(S | {lever}) - sharpe_v(S))
        return total

    sh_tilts, sh_vol, sh_corr = shapley("T"), shapley("V"), shapley("C")

    # ── Max drawdown: waterfall T -> V -> C, identical seed at every state ───
    mdd_none = monte_carlo_max_drawdown(w, mu_0, Sigma_0, n_paths, days, seed)
    mdd_t = monte_carlo_max_drawdown(w, mu_R, Sigma_0, n_paths, days, seed)
    mdd_tv = monte_carlo_max_drawdown(w, mu_R, Sigma_V, n_paths, days, seed)
    mdd_tvc = monte_carlo_max_drawdown(w, mu_R, Sigma_R, n_paths, days, seed)
    mdd_tilts = mdd_t - mdd_none
    mdd_vol = mdd_tv - mdd_t
    mdd_corr = mdd_tvc - mdd_tv

    # ── Per-position variance-share risk contributions ──────────────────────
    rc0 = _variance_risk_contributions(w, Sigma_0)
    rcV = _variance_risk_contributions(w, Sigma_V)
    rcR = _variance_risk_contributions(w, Sigma_R)
    positions = []
    for i, t in enumerate(tickers):
        positions.append({
            "ticker": t,
            "rc_baseline": round(float(rc0[i]), 6),
            "rc_regime": round(float(rcR[i]), 6),
            "drc_from_scaling": round(float(rcV[i] - rc0[i]), 6),
            "drc_from_flip": round(float(rcR[i] - rcV[i]), 6),
        })

    # ── Auto-generated headline ─────────────────────────────────────────────
    total_vol = sR - s0
    corr_share = int(round(100 * vol_corr / total_vol)) if abs(total_vol) > 1e-12 else 0
    flip_names = [
        p["ticker"] for p in sorted(positions, key=lambda p: p["drc_from_flip"], reverse=True)
        if p["drc_from_flip"] > 0.02
    ][:3]
    concentrated = ", ".join(flip_names) if flip_names else "none"
    headline = (
        f"{corr_share}% of your volatility increase comes from the correlation "
        f"regime change, concentrated in: {concentrated}."
    )

    attribution = {
        "return": {"tilts": round(ret_tilts_pp, 4)},
        "vol": {"vol_scaling": round(vol_scaling, 6), "correlation": round(vol_corr, 6)},
        "sharpe": {
            "tilts": round(sh_tilts, 6),
            "vol_scaling": round(sh_vol, 6),
            "correlation": round(sh_corr, 6),
        },
        "mdd": {
            "tilts": round(mdd_tilts, 6),
            "vol_scaling": round(mdd_vol, 6),
            "correlation": round(mdd_corr, 6),
            "method": "waterfall T->V->C",
        },
        "positions": positions,
        "headline": headline,
    }
    return attribution, mdd_none, mdd_tvc
