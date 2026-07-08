"""
Shared portfolio return-series helper.

The value-weighted portfolio daily-return series r_P,t = Σ w_i · r_i,t is needed
by BOTH the discovery correlation column (CORRELATION_COLUMN_SPEC.md §A) and the
portfolio beta tracker (BETA_TRACKER_SPEC.md). Lifted here from
`discovery_context._portfolio_returns` so the two consume one implementation
(BETA_TRACKER_SPEC.md "Prerequisite refactor"). Behaviour is byte-identical to
the original — the correlation tests must stay green after the lift.
"""
import numpy as np
import pandas as pd


def weighted_portfolio_returns(
    returns: pd.DataFrame,
    holdings: dict[str, float],
    exclude: str | None = None,
) -> pd.Series | None:
    """
    Value-weighted portfolio daily return series: r_P,t = sum(w_i * r_i,t).

    `holdings` maps ticker -> weight (need not already sum to 1; renormalized
    here). If `exclude` is given, that ticker is dropped and the remaining
    weights renormalized (§A1 held-candidate exclusion path).

    Returns None if fewer than 1 holding remains after exclusion/filtering,
    or none of the holdings have return data.
    """
    w = dict(holdings)
    if exclude is not None:
        w.pop(exclude, None)

    tickers = [t for t in w.keys() if t in returns.columns]
    if not tickers:
        return None

    total = sum(w[t] for t in tickers)
    if total <= 0:
        return None

    norm_w = {t: w[t] / total for t in tickers}

    sub = returns[tickers]
    weighted = sub.mul(pd.Series(norm_w))
    port_ret = weighted.sum(axis=1, skipna=False)
    return port_ret


def regress_beta(port, bench, window, min_overlap=60):
    """
    OLS beta (cov/var) of the portfolio return series on the benchmark over the
    trailing `window` days of their aligned overlap (BETA_TRACKER_SPEC.md method).
    Returns (beta, r2, n_obs); (None, None, n) when overlap < min_overlap or the
    benchmark has zero variance — never a fabricated 0.
    """
    p_al, b_al = port.tail(window).align(bench.tail(window), join="inner")
    mask = p_al.notna() & b_al.notna()
    p = p_al[mask].values
    b = b_al[mask].values
    n = int(len(p))
    if n < min_overlap:
        return None, None, n
    var_b = float(np.var(b))
    if var_b <= 0:
        return None, None, n
    beta = float(np.cov(p, b)[0, 1] / var_b)
    corr = float(np.corrcoef(p, b)[0, 1])
    return round(beta, 4), round(corr * corr, 4), n
