"""
Portfolio risk context layer: the §1 shared risk engine (PORTFOLIO_RISK_SPEC.md)
and the §8 Portfolio Health metrics (Amendment A).

One estimation path feeds every number, so they are mutually consistent:

  * Return series: the app's shared daily log-return frame (_store["returns"],
    from adjusted closes) and the shared value-weighted helper
    `weighted_portfolio_returns` — the same series the beta tracker regresses.
  * Window: the trailing 252 rows of that frame (the correlation column / beta
    tracker window).
  * Covariance: `ledoit_wolf_constant_correlation` from return_models.py, the
    estimator Black-Litterman ships (reused, not forked), annualized x252.
  * Portfolio vol sigma_P = sqrt(w' S w); MC_i = (S w)_i / sigma_P;
    RC_i = w_i * MC_i, with sum(RC_i) == sigma_P (Euler identity; tested).

Pure functions: no I/O, no yfinance, no global state. The endpoint layer hands
in the returns frame and (for yield) a dividend fetcher, so tests stay offline.
Facts only: nothing here grades, targets, or recommends.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from app.portfolio_series import weighted_portfolio_returns
from app.return_models import ledoit_wolf_constant_correlation

TRADING_DAYS = 252


def normalize_weights(holdings: dict[str, float]) -> dict[str, float]:
    """Holdings -> weights summing to 1 over tickers with positive weight."""
    pos = {t: float(w) for t, w in holdings.items() if w is not None and float(w) > 0}
    tot = sum(pos.values())
    return {t: w / tot for t, w in pos.items()} if tot > 0 else {}


def select_window(returns: pd.DataFrame | None, holdings: dict[str, float],
                  window: int = TRADING_DAYS, min_obs: int = 60):
    """
    Trailing `window` rows of the shared returns frame, split into included
    holdings and excluded ones (§1 insufficient-data rule): a holding is
    excluded when it is absent from the loaded frame or has < `min_obs`
    non-missing days in the window. Never imputed, never zero-filled.

    Returns (frame restricted to included tickers with complete rows,
             included tickers, excluded [{ticker, reason, n_obs}]).
    """
    tickers = [t for t in holdings if holdings[t] is not None and float(holdings[t]) > 0]
    excluded = []
    if returns is None or len(returns) == 0:
        return None, [], [{"ticker": t, "reason": "not_loaded", "n_obs": 0} for t in tickers]
    win = returns.tail(window)
    included = []
    for t in tickers:
        if t not in win.columns:
            excluded.append({"ticker": t, "reason": "not_loaded", "n_obs": 0})
            continue
        n = int(win[t].notna().sum())
        if n < min_obs:
            excluded.append({"ticker": t, "reason": "insufficient_history", "n_obs": n})
        else:
            included.append(t)
    if not included:
        return None, [], excluded
    sub = win[included].dropna()
    if len(sub) < min_obs:
        # Each holding clears the floor alone but their common overlap does not.
        excluded += [{"ticker": t, "reason": "insufficient_overlap", "n_obs": int(len(sub))} for t in included]
        return None, [], excluded
    return sub, included, excluded


def risk_decomposition(sub: pd.DataFrame, weights: dict[str, float]) -> dict:
    """
    §1 engine core on an aligned complete window. `weights` are renormalized
    over the columns of `sub`. Returns sigma_P, per-asset MC/RC/shares and the
    Ledoit-Wolf shrinkage intensity used.
    """
    tickers = list(sub.columns)
    w = np.array([weights.get(t, 0.0) for t in tickers], dtype=float)
    w = w / w.sum()
    cov, alpha, n_days = ledoit_wolf_constant_correlation(sub.values, annualize=True,
                                                          periods_per_year=TRADING_DAYS)
    sw = cov @ w
    var_p = float(w @ sw)
    sigma = math.sqrt(max(var_p, 0.0))
    mc = sw / sigma if sigma > 0 else np.full(len(w), np.nan)
    rc = w * mc
    rows = []
    for i, t in enumerate(tickers):
        rows.append({
            "ticker": t,
            "weight": float(w[i]),
            "mc": float(mc[i]),
            "rc": float(rc[i]),
            "risk_share": float(rc[i] / sigma) if sigma > 0 else None,
        })
    rows.sort(key=lambda r: (r["risk_share"] if r["risk_share"] is not None else -1e9), reverse=True)
    return {"sigma": sigma, "rows": rows, "shrinkage_alpha": float(alpha), "n_days": int(n_days),
            "cov": cov, "tickers": tickers, "w": w}


def effective_bets(rcs) -> float | None:
    """
    N_eff = 1 / sum(s_i^2), s_i = |RC_i| / sum_j |RC_j|  (|RC| normalization).
    Identical to the spec's 1/sum((RC_i/sigma_P)^2) whenever no RC_i < 0; with a
    negative RC the |RC| convention keeps s_i in [0, 1] (disclosed in the UI).
    """
    a = np.abs(np.asarray(list(rcs), dtype=float))
    tot = a.sum()
    if tot <= 0 or not np.isfinite(tot):
        return None
    s = a / tot
    return float(1.0 / np.sum(s ** 2))


def max_drawdown(port: pd.Series) -> dict | None:
    """Worst peak-to-trough of the value path exp(cumsum(r)) over the window."""
    r = port.dropna()
    if len(r) < 2:
        return None
    path = np.exp(np.concatenate([[0.0], np.cumsum(r.values)]))
    dates = [None] + list(r.index)
    peak_i, best, best_peak, best_trough = 0, 0.0, 0, 0
    for i in range(1, len(path)):
        if path[i] > path[peak_i]:
            peak_i = i
        dd = path[i] / path[peak_i] - 1.0
        if dd < best:
            best, best_peak, best_trough = dd, peak_i, i
    fmt = lambda d: None if d is None else pd.Timestamp(d).strftime("%Y-%m-%d")
    peak_date = fmt(dates[best_peak]) if best_peak > 0 else fmt(r.index[0])
    return {"max_dd": float(best), "peak": peak_date, "trough": fmt(dates[best_trough]) if best_trough > 0 else None}


def historical_var_cvar(port: pd.Series, level: float = 0.95) -> dict | None:
    """
    Historical 1-day VaR/CVaR from the empirical distribution (no parametric
    assumption). k = ceil((1-level) * n) worst days; VaR = loss on the k-th
    worst day, CVaR = mean loss over those k days, so CVaR >= VaR by construction.
    Reported as positive loss magnitudes (fractions).
    """
    r = np.sort(port.dropna().values)
    n = len(r)
    if n < 20:
        return None
    k = max(1, math.ceil((1.0 - level) * n))
    tail = r[:k]
    return {"var": float(-tail[-1]), "cvar": float(-tail.mean()), "k": int(k), "n": int(n)}


def capture_ratios(port: pd.Series, bench: pd.Series, min_obs: int) -> dict:
    """
    Up/down capture vs the benchmark on the same days: mean portfolio return on
    benchmark-up days / mean benchmark return on those days (and the same for
    down days). A side with < min_obs days renders None (never a noisy ratio).
    """
    p, b = port.align(bench, join="inner")
    m = p.notna() & b.notna()
    p, b = p[m], b[m]
    up, dn = b > 0, b < 0
    out = {"up_days": int(up.sum()), "down_days": int(dn.sum()), "min_obs": int(min_obs)}
    out["up"] = float(p[up].mean() / b[up].mean()) if up.sum() >= min_obs and b[up].mean() != 0 else None
    out["down"] = float(p[dn].mean() / b[dn].mean()) if dn.sum() >= min_obs and b[dn].mean() != 0 else None
    return out


def sharpe_1y(series: pd.Series, rf: float) -> dict | None:
    """
    The shipped per-ticker Sharpe method (app/main.py add_ticker): (total return
    over the window - rf) / annualized vol of daily log returns, rf from config.
    Total return compounds the log series: exp(sum r) - 1.
    """
    r = series.dropna()
    if len(r) < 20:
        return None
    vol = float(r.std(ddof=1) * math.sqrt(TRADING_DAYS))
    if not vol > 0:
        return None
    tot = float(math.exp(r.sum()) - 1.0)
    return {"sharpe": (tot - rf) / vol, "total_return": tot, "vol": vol, "n": int(len(r))}


def pairwise_correlations(sub: pd.DataFrame) -> dict:
    """Sample (Pearson) pairwise correlations over the window: mean, count, highest pair."""
    cols = list(sub.columns)
    if len(cols) < 2:
        return {"avg": None, "pairs": 0, "highest": None}
    c = sub.corr().values
    vals, best = [], None
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = float(c[i, j])
            vals.append(v)
            if best is None or v > best[2]:
                best = (cols[i], cols[j], v)
    return {"avg": float(np.mean(vals)), "pairs": len(vals),
            "highest": {"a": best[0], "b": best[1], "r": best[2]}}


def portfolio_health(returns: pd.DataFrame | None, holdings: dict[str, float], *,
                     benchmark: str = "SPY", rf: float = 0.043, window: int = TRADING_DAYS,
                     min_obs: int = 60, capture_min_obs: int = 60) -> dict:
    """
    §8 Portfolio Health metrics from the §1 engine. Every metric is computed on
    the included holdings (renormalized); excluded holdings are listed once.
    Missing inputs yield None for that metric only — never a fabricated value.
    """
    weights_all = normalize_weights(holdings)
    if not weights_all:
        return {"status": "empty"}
    sub, included, excluded = select_window(returns, weights_all, window, min_obs)
    base = {"status": "ok", "window": window, "excluded": excluded, "included": included,
            "n_holdings": len(weights_all)}
    if sub is None:
        base["status"] = "insufficient_data"
        return base
    w_inc = {t: weights_all[t] for t in included}
    dec = risk_decomposition(sub, w_inc)
    port = weighted_portfolio_returns(sub, w_inc)
    base.update({
        "first": sub.index[0].strftime("%Y-%m-%d"), "last": sub.index[-1].strftime("%Y-%m-%d"),
        "n_obs": int(len(sub)),
        "vol": dec["sigma"],
        "shrinkage_alpha": dec["shrinkage_alpha"],
        "risk_rows": [{k: r[k] for k in ("ticker", "weight", "mc", "rc", "risk_share")} for r in dec["rows"]],
        "rc_sum": float(sum(r["rc"] for r in dec["rows"])),
        "effective_bets": effective_bets(r["rc"] for r in dec["rows"]),
        "any_negative_rc": any(r["rc"] < 0 for r in dec["rows"]),
        "drawdown": max_drawdown(port),
        "var": historical_var_cvar(port),
        "pairwise": pairwise_correlations(sub),
    })
    bench = None
    if returns is not None and benchmark in returns.columns:
        bench = returns[benchmark].reindex(sub.index)
    base["capture"] = capture_ratios(port, bench, capture_min_obs) if bench is not None else None
    base["sharpe"] = sharpe_1y(port, rf)
    base["benchmark_sharpe"] = sharpe_1y(bench, rf) if bench is not None else None
    base["benchmark"] = benchmark
    base["rf"] = rf
    return base


def trailing_yield(dividends: pd.Series | None, raw_close: pd.Series | None) -> float | None:
    """
    J2 convention (BACKTEST_RUN_DECISIONS.md): trailing-12m cash dividends per
    share / latest RAW (unadjusted) close, as a fraction. None if either input
    is missing — never 0 for "unknown".
    """
    if raw_close is None or len(raw_close.dropna()) == 0:
        return None
    last_px = float(raw_close.dropna().iloc[-1])
    if not last_px > 0:
        return None
    if dividends is None:
        return None
    d = dividends.dropna()
    if len(d):
        end = raw_close.dropna().index[-1]
        d = d[d.index > end - pd.Timedelta(days=365)]
    return float(d.sum()) / last_px


def weighted_yield(yields: dict[str, float | None], weights: dict[str, float]) -> dict:
    """Weighted trailing yield over holdings with a known yield; lists unknowns."""
    known = {t: y for t, y in yields.items() if y is not None and t in weights}
    unknown = sorted(t for t in weights if yields.get(t) is None)
    wk = sum(weights[t] for t in known)
    if wk <= 0:
        return {"yield": None, "unknown": unknown, "coverage": 0.0}
    return {"yield": sum(weights[t] * known[t] for t in known) / wk, "unknown": unknown, "coverage": wk}


def redundancy_pairs(returns: pd.DataFrame | None, candidates: list[str], threshold: float,
                     window: int = TRADING_DAYS, min_obs: int = 60) -> dict:
    """
    §5: pairwise Pearson correlations among shortlist candidates over the same
    trailing window; for each candidate with a peer at r >= threshold, name its
    most-correlated peer. Candidates not in the loaded frame are reported, not
    guessed.
    """
    cands = list(dict.fromkeys(candidates))[:100]
    if returns is None or len(returns) == 0:
        return {"threshold": threshold, "window": window, "flags": [], "not_loaded": cands, "n_pairs": 0}
    win = returns.tail(window)
    present = [t for t in cands if t in win.columns and int(win[t].notna().sum()) >= min_obs]
    not_loaded = [t for t in cands if t not in present]
    flags = []
    n_pairs = len(present) * (len(present) - 1) // 2
    if len(present) >= 2:
        c = win[present].corr(min_periods=min_obs)
        for t in present:
            row = c[t].drop(labels=[t]).dropna()
            if len(row) == 0:
                continue
            peer = row.idxmax()
            r = float(row.loc[peer])
            if r >= threshold:
                days = int(win[[t, peer]].dropna().shape[0])
                flags.append({"ticker": t, "peer": peer, "r": r, "days": days})
    return {"threshold": threshold, "window": window, "flags": flags, "not_loaded": not_loaded, "n_pairs": n_pairs}
