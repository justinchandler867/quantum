"""
Point-in-time re-implementation of the screener Stage-2/Stage-3 ranking for the
walk-forward validation (VALIDATION_SPEC.md §5). Faithful to app/screener.py's
compute_factor_scores / rank_by_composite / compute_fit_scores as verified
2026-07-08, with the two sealed deviations:
  - value factor DROPPED (§L2 lookahead avoidance); §4 renormalized weights.
  - yield factor reconstructed point-in-time (§L2): trailing-12m dividends / raw
    close at freeze / *100, z-scored cross-sectionally like the others.

NOT imported by app/. Reads only the cached history (backtest/data/raw).
All computations at a freeze date T use only data with index <= T.
"""
import numpy as np
import pandas as pd

TRADING_YEAR = 252
MOM_SKIP = 21
MIN_HISTORY_DAYS = 273          # 252 + 21-day momentum skip (§9)
PRICE_GATE = 5.0                # §L3 (config SCREEN_MIN_PRICE)
VOLUME_GATE = 500_000           # §L3 (config SCREEN_MIN_AVG_VOLUME)
RF_ANNUAL = 0.043               # config.py:54

# §4 backtest weights (value dropped, renormalized). Used for BOTH the Stage-2
# composite and the Stage-3 goal component (same goal_weights, as in production).
GOAL_WEIGHTS = {
    "Growth":   {"momentum": 0.625, "quality": 0.250, "low_vol": 0.0625, "yield": 0.0625},
    "Income":   {"momentum": 0.000, "quality": 0.222, "low_vol": 0.333,  "yield": 0.444},
    "Balanced": {"momentum": 0.250, "quality": 0.250, "low_vol": 0.250,  "yield": 0.250},
}
# (goal, risk_score, horizon_years). Balanced horizon not given by spec -> logged
# judgment call (10.0), see BACKTEST_RUN_DECISIONS.md.
PROFILES = {
    "Growth-95":   ("Growth", 95, 30.0),
    "Income-20":   ("Income", 20, 5.0),
    "Balanced-50": ("Balanced", 50, 10.0),
}

# Stage-3 fit budget (screener.py:548-552)
FIT_GOAL_PTS, FIT_BAND_PTS, FIT_HORIZON_PTS, FIT_QUALITY_PTS = 55, 30, 8, 7
FIT_BAND_FALLOFF = 0.85


def _target_beta(risk_score):
    return 0.20 + (max(0, min(100, risk_score)) / 100.0) * 1.30


def _winsor_z(s):
    """Winsorize at 3σ then z-score, matching compute_factor_scores exactly."""
    s = s.dropna()
    if len(s) < 5:
        return pd.Series(0.0, index=s.index)
    mu, sigma = s.mean(), s.std()
    if sigma <= 0:
        return pd.Series(0.0, index=s.index)
    clipped = s.clip(mu - 3 * sigma, mu + 3 * sigma)
    return (clipped - clipped.mean()) / clipped.std()


def eligible_universe(T, adj, raw_close, volume):
    """Names with >=273 trading days of history <=T and passing PIT price+volume
    gates (§L3). Market-cap gate stands in as universe membership (already the
    universe)."""
    hist = adj.loc[:T]
    out = []
    for tk in adj.columns:
        col = hist[tk].dropna()
        if len(col) < MIN_HISTORY_DAYS:
            continue
        px = raw_close.loc[:T, tk].dropna()
        if len(px) == 0 or px.iloc[-1] < PRICE_GATE:
            continue
        vol = volume.loc[:T, tk].dropna()
        if len(vol) < TRADING_YEAR or vol.iloc[-TRADING_YEAR:].mean() < VOLUME_GATE:
            continue
        out.append(tk)
    return out


def compute_factors_at(T, tickers, adj, raw_close, dividends, benchmark="SPY"):
    """PIT factor frame for `tickers` at freeze T. Returns DataFrame indexed by
    ticker: z_momentum, z_quality, z_low_vol, z_yield, beta, volatility, sharpe."""
    logret = np.log(adj.loc[:T] / adj.loc[:T].shift(1))
    ret = logret[tickers]

    raw = pd.DataFrame(index=tickers)
    # momentum: skip most recent 21 days, then 12m/6m sums (screener.py:378-379,412)
    skip = ret.iloc[:-MOM_SKIP]
    ret_12m = skip.iloc[-TRADING_YEAR:].sum()
    ret_6m = skip.iloc[-126:].sum()
    raw["momentum"] = 0.6 * ret_12m + 0.4 * ret_6m

    trailing = ret.iloc[-TRADING_YEAR:]
    ann_vol = trailing.std() * np.sqrt(TRADING_YEAR)
    ann_ret = trailing.mean() * TRADING_YEAR
    sharpe = ann_ret / ann_vol.replace(0, np.nan)
    raw["quality"] = sharpe
    raw["low_vol"] = -ann_vol
    raw["volatility"] = ann_vol
    raw["sharpe"] = sharpe

    # beta vs benchmark over trailing 252 (screener.py:392-399)
    if benchmark in logret.columns:
        bench = logret[benchmark].iloc[-TRADING_YEAR:].reindex(trailing.index)
        bvar = bench.var()
        raw["beta"] = trailing.apply(lambda c: c.cov(bench) / bvar) if bvar > 0 else 1.0
    else:
        raw["beta"] = 1.0

    # PIT yield (§L2): trailing-12m dividends / raw close at T * 100
    yld = {}
    for tk in tickers:
        px = raw_close.loc[:T, tk].dropna()
        p = px.iloc[-1] if len(px) else np.nan
        d = dividends.get(tk)
        if d is None or p is None or p <= 0 or (isinstance(p, float) and np.isnan(p)):
            yld[tk] = 0.0
            continue
        window = d[(d.index > T - pd.Timedelta(days=365)) & (d.index <= T)]
        yld[tk] = float(window.sum()) / float(p) * 100.0
    raw["yield_factor"] = pd.Series(yld)

    raw["z_momentum"] = _winsor_z(raw["momentum"]).reindex(raw.index)
    raw["z_quality"] = _winsor_z(raw["quality"]).reindex(raw.index)
    raw["z_low_vol"] = _winsor_z(raw["low_vol"]).reindex(raw.index)
    raw["z_yield"] = _winsor_z(raw["yield_factor"]).reindex(raw.index)
    return raw.fillna(0.0)


def fit_rank(factors, goal, risk, horizon):
    """Stage-2 composite top-120 then Stage-3 fit score (screener.py:521-646)."""
    w = GOAL_WEIGHTS[goal]
    comp2 = (w["momentum"] * factors["z_momentum"] + w["quality"] * factors["z_quality"]
             + w["low_vol"] * factors["z_low_vol"] + w["yield"] * factors["z_yield"])
    stage2 = comp2.sort_values(ascending=False).head(120).index
    c = factors.loc[stage2].copy()

    composite = (w["momentum"] * c["z_momentum"] + w["quality"] * c["z_quality"]
                 + w["low_vol"] * c["z_low_vol"] + w["yield"] * c["z_yield"])
    cstd = float(composite.std(ddof=0))
    comp_z = (composite - composite.mean()) / cstd if cstd > 1e-9 else composite * 0.0
    goal_pts = ((comp_z + 2) / 4 * FIT_GOAL_PTS).clip(0, FIT_GOAL_PTS)

    target = _target_beta(risk)
    band_pts = (FIT_BAND_PTS * (1 - (c["beta"].abs() - target).abs() / FIT_BAND_FALLOFF)).clip(lower=0)

    vol = c["volatility"].abs()
    min_h = 1 + vol * 15
    horizon_pts = pd.Series(FIT_HORIZON_PTS * 0.15, index=c.index)
    horizon_pts[horizon >= min_h * 0.6] = FIT_HORIZON_PTS * 0.55
    horizon_pts[horizon >= min_h] = FIT_HORIZON_PTS

    sh = c["sharpe"]
    q = pd.Series(0.0, index=c.index)
    q[sh >= 0.2] = FIT_QUALITY_PTS * 0.3
    q[sh >= 0.5] = FIT_QUALITY_PTS * 0.6
    q[sh >= 1.0] = FIT_QUALITY_PTS

    total = (goal_pts + band_pts + horizon_pts + q).clip(upper=100)
    return total.sort_values(ascending=False)


def forward_metrics(names, T, horizon_days, adj, raw_close, dividends, benchmark="SPY"):
    """Equal-weight buy-and-hold forward metrics over (T, T+horizon_days]."""
    fut = adj.loc[adj.index > T].head(horizon_days)
    if len(fut) < 5:
        return None
    sub = fut[names]
    base = adj.loc[:T, names].iloc[-1]
    path = sub.div(base, axis=1).mean(axis=1, skipna=True)  # equal-weight value path
    daily = path.pct_change().dropna()
    total_return = float(path.iloc[-1] - 1.0)
    if len(daily) < 5 or daily.std() == 0:
        sharpe, vol = np.nan, np.nan
    else:
        vol = float(daily.std() * np.sqrt(TRADING_YEAR))
        sharpe = float((daily.mean() - RF_ANNUAL / TRADING_YEAR) / daily.std() * np.sqrt(TRADING_YEAR))
    mdd = float((path / path.cummax() - 1).min())

    bench_fut = adj.loc[adj.index > T, benchmark].head(horizon_days)
    bench_daily = bench_fut.pct_change().reindex(daily.index)
    bvar = bench_daily.var()
    beta = float(daily.cov(bench_daily) / bvar) if bvar and bvar > 0 else np.nan

    # realized dividend yield over the window: divs received / entry raw price
    ry = []
    for tk in names:
        d = dividends.get(tk)
        p0 = raw_close.loc[:T, tk].dropna()
        if d is None or len(p0) == 0 or p0.iloc[-1] <= 0:
            ry.append(0.0); continue
        w = d[(d.index > T) & (d.index <= fut.index[-1])]
        ry.append(float(w.sum()) / float(p0.iloc[-1]))
    realized_yield = float(np.mean(ry)) if ry else 0.0

    return {"total_return": total_return, "sharpe": sharpe, "max_dd": mdd, "vol": vol,
            "realized_yield": realized_yield, "beta": beta}


def percentile(test_val, null_vals):
    """Percentile of test_val within the null distribution (fraction strictly
    below), in [0,100]. NaN-safe: nulls with NaN are dropped."""
    arr = np.array([v for v in null_vals if v is not None and not (isinstance(v, float) and np.isnan(v))])
    if len(arr) == 0 or test_val is None or (isinstance(test_val, float) and np.isnan(test_val)):
        return np.nan
    return float((arr < test_val).sum()) / len(arr) * 100.0
