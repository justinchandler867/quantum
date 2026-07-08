"""
Quantex Backend — FastAPI Application
Correlation, covariance, and diagnostics API endpoints.
"""
import asyncio
import json
import logging
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.config import (
    CACHE_TTL_DAILY,
    CACHE_TTL_WEEKLY,
    DEFAULT_LAMBDA,
    REFERENCE_HEDGES,
    RISK_FREE_RATE,
    BETA_BENCHMARK,
)
from app.models import (
    CorrelationRequest,
    CorrelationResponse,
    CovarianceRequest,
    CovarianceResponse,
    DiagnosticsRequest,
    DiagnosticsResponse,
    OptimizeRequest,
    OptimizeResponse,
    MultiOptimizeRequest,
    MultiOptimizeResponse,
    StressTestRequest,
    StressTestResponse,
    StressMetrics,
    SandboxStressRequest,
    RegimeInput,
    FrontierRequest,
    FrontierResponse,
    ScreenRequest,
    ScreenResponse,
    ScreenedAssetResponse,
    AnalyzeRequest,
    AnalyzeSingleRequest,
    CompanyAnalysisResponse,
    BatchAnalysisResponse,
    SectorAnalysisResponse,
    TradeRequest,
    TradeResponse,
    PositionResponse,
    PerformanceResponse,
    AccountSummaryResponse,
    CfaTagResponse,
    PriceOptionRequest,
    OptionPriceResponse,
    IVRequest,
    StrategyRequest,
    PairInfo,
    ReferenceHedgeInfo,
    PricesRequest,
    PricesResponse,
    DiscoveryContextRequest,
    DiscoveryContextResponse,
    PortfolioBetaRequest,
    PortfolioBetaResponse,
    RollingCorrRequest,
    RollingCorrResponse,
    DiscoveryContextEntry,
    SearchRequest,
    SearchResult,
    TickerAddRequest,
    TickerAddResponse,
    NewsRequest,
    NewsItem,
    NewsResponse,
)
from app.data_ingest import (
    fetch_prices,
    compute_log_returns,
    identify_stress_windows,
    split_returns_by_regime,
    annualized_volatility,
)
from app.correlation_engine import (
    compute_normal_correlation,
    compute_stress_correlation,
    compute_blended_correlation,
    build_covariance_matrix,
    compute_diagnostics,
    lambda_from_risk_score,
)
from app.cache import cache_key, cache_get, cache_set
from app.discovery_context import compute_candidate_context, MIN_OVERLAP_DAYS
from app.portfolio_series import weighted_portfolio_returns, regress_beta, contiguous_true_ranges

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Data Store ───────────────────────────────────────────────────────────────
# Holds pre-fetched returns in memory after startup or first request.
# In production, replace with a proper database-backed store.
_store = {
    "prices": None,
    "volumes": None,
    "returns": None,
    "normal_returns": None,
    "stress_returns": None,
    "stress_mask": None,
    # Effective-size cache for Black-Litterman market weights: ticker -> float.
    # Populated lazily by _get_market_caps (marketCap, ETF fallback totalAssets).
    "fundamentals": {},
}
_store_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: log readiness. Data is loaded lazily on first request."""
    logger.info("Quantex backend starting — data will load on first request")
    yield
    logger.info("Quantex backend shutting down")


app = FastAPI(
    title="Quantex Correlation API",
    version="1.0.0",
    description="Regime-aware correlation and covariance matrices for portfolio optimization",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ensure_data(tickers: list[str]) -> None:
    """
    Lazy-load price data and compute returns + stress windows.
    Caches in memory for the process lifetime.
    Reloads if new tickers are requested that aren't in the current dataset.
    """
    all_needed = set(tickers + REFERENCE_HEDGES + [BETA_BENCHMARK])

    if _store["returns"] is not None:
        have = set(_store["returns"].columns)
        missing = all_needed - have
        if not missing:
            return
        logger.info(f"New tickers requested: {missing} — refetching")

    # Fetch everything
    logger.info(f"Loading price data for {len(all_needed)} tickers...")
    try:
        prices, volumes = fetch_prices(list(all_needed), include_hedges=True, return_volumes=True)
    except Exception as exc:
        logger.error(f"Price fetch failed: {exc}")
        raise HTTPException(status_code=502, detail=f"Failed to fetch price data: {exc}")

    returns = compute_log_returns(prices)
    stress_mask = identify_stress_windows(returns)
    normal_returns, stress_returns = split_returns_by_regime(returns, stress_mask)

    with _store_lock:
        _store["prices"] = prices
        _store["volumes"] = volumes
        _store["returns"] = returns
        _store["normal_returns"] = normal_returns
        _store["stress_returns"] = stress_returns
        _store["stress_mask"] = stress_mask

    logger.info(
        f"Data loaded: {returns.shape[1]} tickers, {returns.shape[0]} days, "
        f"stress days: {stress_mask.sum()}"
    )


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    has_data = _store["returns"] is not None
    n_tickers = _store["returns"].shape[1] if has_data else 0
    return {"status": "ok", "data_loaded": has_data, "tickers": n_tickers}


@app.post("/api/correlations", response_model=CorrelationResponse)
async def get_correlations(req: CorrelationRequest):
    """
    Returns the correlation matrix for the given tickers and regime.
    Regimes: 'normal' (trailing window), 'stress' (drawdown periods), 'blended'.
    """
    # Check cache
    ck = cache_key("corr", req.tickers, regime=req.regime, window=req.window)
    cached = cache_get(ck)
    if cached:
        return CorrelationResponse(**cached)

    _ensure_data(req.tickers)
    returns = _store["returns"]
    normal_returns = _store["normal_returns"]
    stress_returns = _store["stress_returns"]

    try:
        if req.regime == "normal":
            result = compute_normal_correlation(normal_returns, req.tickers, req.window)

        elif req.regime == "stress":
            result = compute_stress_correlation(stress_returns, req.tickers)

        elif req.regime == "blended":
            normal = compute_normal_correlation(normal_returns, req.tickers, req.window)
            stress = compute_stress_correlation(stress_returns, req.tickers)

            # Align to common tickers
            common = [t for t in normal.tickers if t in stress.tickers]
            if len(common) < 2:
                raise ValueError("Fewer than 2 tickers have both normal and stress data")

            n_idx = [normal.tickers.index(t) for t in common]
            s_idx = [stress.tickers.index(t) for t in common]

            from app.correlation_engine import CorrelationResult
            normal_aligned = CorrelationResult(
                matrix=normal.matrix[np.ix_(n_idx, n_idx)], tickers=common,
                regime="normal", observation_days=normal.observation_days,
                shrinkage_applied=normal.shrinkage_applied,
            )
            stress_aligned = CorrelationResult(
                matrix=stress.matrix[np.ix_(s_idx, s_idx)], tickers=common,
                regime="stress", observation_days=stress.observation_days,
                shrinkage_applied=stress.shrinkage_applied,
            )
            result = compute_blended_correlation(normal_aligned, stress_aligned)

        else:
            raise HTTPException(status_code=400, detail=f"Unknown regime: {req.regime}")

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    response = CorrelationResponse(
        matrix=result.matrix.tolist(),
        tickers=result.tickers,
        regime=result.regime,
        observation_days=result.observation_days,
        shrinkage_applied=result.shrinkage_applied,
        shrinkage_coefficient=result.shrinkage_coefficient,
    )

    # Cache: daily for normal, weekly for stress
    ttl = CACHE_TTL_WEEKLY if req.regime == "stress" else CACHE_TTL_DAILY
    cache_set(ck, response.model_dump(), ttl)

    return response


@app.post("/api/covariance", response_model=CovarianceResponse)
async def get_covariance(req: CovarianceRequest):
    """
    Returns the blended covariance matrix ready for optimization.
    Accepts either an explicit λ or a risk_score (which computes λ automatically).
    """
    lam = req.lam
    if req.risk_score is not None:
        lam = lambda_from_risk_score(req.risk_score)

    ck = cache_key("cov", req.tickers, lam=f"{lam:.3f}", window=req.window)
    cached = cache_get(ck)
    if cached:
        return CovarianceResponse(**cached)

    _ensure_data(req.tickers)
    normal_returns = _store["normal_returns"]
    stress_returns = _store["stress_returns"]

    try:
        result = build_covariance_matrix(
            normal_returns, stress_returns, req.tickers,
            lam=lam, window=req.window,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    vol_blended = ((1 - lam) * result.vol_normal + lam * result.vol_stress).tolist()

    response = CovarianceResponse(
        matrix=[row.tolist() for row in result.matrix],
        tickers=result.tickers,
        lambda_used=round(result.lambda_used, 4),
        vol_normal=result.vol_normal.tolist(),
        vol_stress=result.vol_stress.tolist(),
        vol_blended=vol_blended,
    )

    cache_set(ck, response.model_dump(), CACHE_TTL_DAILY)
    return response


@app.post("/api/correlation-diagnostics", response_model=DiagnosticsResponse)
async def get_diagnostics(req: DiagnosticsRequest):
    """
    Returns summary diagnostics for the frontend correlation view.
    Includes regime comparison, high-correlation warnings, hedging pairs,
    and reference hedge correlations.
    """
    ck = cache_key("diag", req.tickers)
    cached = cache_get(ck)
    if cached:
        return DiagnosticsResponse(**cached)

    _ensure_data(req.tickers)
    normal_returns = _store["normal_returns"]
    stress_returns = _store["stress_returns"]

    try:
        # Compute both matrices (including reference hedges)
        all_tickers = list(set(req.tickers + REFERENCE_HEDGES))
        normal_corr = compute_normal_correlation(normal_returns, all_tickers)
        stress_corr = compute_stress_correlation(stress_returns, all_tickers)

        weights_arr = None
        if req.weights and len(req.weights) == len(req.tickers):
            weights_arr = np.array(req.weights)

        diag = compute_diagnostics(normal_corr, stress_corr, req.tickers, weights_arr)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Build warnings
    warnings = []
    if diag.corr_spike_pct > 40:
        warnings.append(
            f"Correlation spikes {diag.corr_spike_pct:.0f}% in stress — "
            f"diversification benefit drops significantly in drawdowns"
        )
    if len(diag.high_corr_pairs_stress) > 0:
        pair_strs = [f"{a}/{b}" for a, b, _ in diag.high_corr_pairs_stress[:3]]
        warnings.append(
            f"Pairs with stress ρ > 0.85 (effectively no diversification in crisis): "
            + ", ".join(pair_strs)
        )
    if not diag.hedging_pairs and not any(
        v.get("avg_corr_normal", 1) < 0 for v in diag.reference_correlations.values()
    ):
        warnings.append(
            "No negative-correlation positions — consider adding bond or gold exposure "
            "for crisis protection"
        )
    if len(req.tickers) < 8:
        warnings.append(
            f"Only {len(req.tickers)} positions — unsystematic risk is not fully diversified. "
            f"Consider adding positions to at least 8."
        )

    # Diversification message
    div_msg = (
        f"Average pairwise ρ: {diag.avg_corr_normal:.2f} normal → "
        f"{diag.avg_corr_stress:.2f} stress "
        f"(+{diag.corr_spike_pct:.0f}%)"
    )

    # Reference hedges
    ref_list = []
    for ticker, info in diag.reference_correlations.items():
        ref_list.append(ReferenceHedgeInfo(
            ticker=ticker,
            avg_corr_normal=info["avg_corr_normal"],
            avg_corr_stress=info.get("avg_corr_stress"),
        ))

    response = DiagnosticsResponse(
        avg_corr_normal=diag.avg_corr_normal,
        avg_corr_stress=diag.avg_corr_stress,
        corr_spike_pct=diag.corr_spike_pct,
        diversification_message=div_msg,
        high_corr_pairs_normal=[
            PairInfo(ticker_a=a, ticker_b=b, correlation=c)
            for a, b, c in diag.high_corr_pairs_normal
        ],
        high_corr_pairs_stress=[
            PairInfo(ticker_a=a, ticker_b=b, correlation=c)
            for a, b, c in diag.high_corr_pairs_stress
        ],
        hedging_pairs=[
            PairInfo(ticker_a=a, ticker_b=b, correlation=c)
            for a, b, c in diag.hedging_pairs
        ],
        reference_hedges=ref_list,
        warnings=warnings,
    )

    cache_set(ck, response.model_dump(), CACHE_TTL_DAILY)
    return response


@app.post("/api/discovery/context", response_model=DiscoveryContextResponse)
async def discovery_context(req: DiscoveryContextRequest):
    """
    Discovery column: "Corr (calm / stress)" — CORRELATION_COLUMN_SPEC.md §A.

    For each candidate, computes 2-series Pearson correlation against the
    user's value-weighted active portfolio, in the trailing 252-day normal
    window and over data_ingest's identified stress windows (>= MIN_STRESS_DAYS
    for a stable estimate). Held candidates are compared against the
    portfolio excluding themselves (renormalized).

    Separate call from `/api/screen` by design (§E4): this must never add
    latency to screening. One batched call per shortlist — no per-candidate
    yfinance calls; reads the cached price/returns frame only.

    No Ledoit-Wolf shrinkage: each computation is a 2-series Pearson
    correlation, not a matrix — shrinkage does not apply (§A3).
    """
    if not req.holdings:
        # §A4: empty portfolio — caller should hide the column entirely and
        # never invoke this endpoint. If it is called anyway, return empty
        # results rather than guessing at a portfolio.
        return DiscoveryContextResponse(results=[])

    holdings = {h.ticker: h.weight for h in req.holdings}
    all_needed = list(set(list(holdings.keys()) + list(req.candidates)))

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _ensure_data, all_needed)

    returns = _store["returns"]
    stress_mask = _store["stress_mask"]

    results = []
    for ticker in req.candidates:
        ctx = compute_candidate_context(
            ticker=ticker,
            returns=returns,
            stress_mask=stress_mask,
            holdings=holdings,
            window=req.window,
        )
        results.append(DiscoveryContextEntry(
            ticker=ctx.ticker,
            corr_normal=ctx.corr_normal,
            corr_stress=ctx.corr_stress,
            band=ctx.band,
            flip_flag=ctx.flip_flag,
            days_normal=ctx.days_normal,
            days_stress=ctx.days_stress,
            status=ctx.status,
        ))

    return DiscoveryContextResponse(results=results)


@app.post("/api/portfolio/beta", response_model=PortfolioBetaResponse)
async def portfolio_beta(req: PortfolioBetaRequest):
    """
    Portfolio beta tracker (BETA_TRACKER_SPEC.md). Beta is the OLS regression
    (cov/var) of the value-weighted portfolio daily-return series on the
    benchmark, over the same trailing 252-day window as the correlation column —
    NOT the weighted average of per-asset betas. Discloses R² and observation
    count so a low R² is shown, not hidden. Accepts hypothetical/preview holdings
    for the marginal preview (β X → Y); the caller computes both.
    """
    holdings = {h.ticker: h.weight for h in req.holdings}
    needed = list(holdings.keys()) + [req.benchmark]

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _ensure_data, needed)
    returns = _store["returns"]

    def _insufficient(n):
        return PortfolioBetaResponse(beta=None, r2=None, n_obs=n, window=req.window,
                                     benchmark=req.benchmark, status="insufficient_data")

    port = weighted_portfolio_returns(returns, holdings)
    if port is None or returns is None or req.benchmark not in returns.columns:
        return _insufficient(0)

    beta, r2, n = regress_beta(port, returns[req.benchmark], req.window,
                               min_overlap=MIN_OVERLAP_DAYS)  # §A4: <60 -> "—", never 0
    if beta is None:
        return _insufficient(n)
    return PortfolioBetaResponse(beta=beta, r2=r2, n_obs=n,
                                 window=req.window, benchmark=req.benchmark, status="ok")


@app.post("/api/correlations/rolling", response_model=RollingCorrResponse)
async def correlations_rolling(req: RollingCorrRequest):
    """
    Rolling stock-bond correlation visual (CORRELATION_COLUMN_SPEC.md §B).
    Each sleeve is a value-weighted daily-return series (shared helper) or a
    fallback single ticker ("SPY"/"TLT"). Returns the rolling `window`-day Pearson
    correlation of the two sleeves plus data_ingest's stress windows (shaded by
    the frontend). Reuses _store["stress_mask"] — no stress recomputation.
    """
    def tickers_of(s):
        return [s] if isinstance(s, str) else [h.ticker for h in s]

    needed = tickers_of(req.series_a) + tickers_of(req.series_b)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _ensure_data, needed)
    returns = _store["returns"]
    stress_mask = _store["stress_mask"]

    def series_of(s):
        if isinstance(s, str):
            return returns[s] if returns is not None and s in returns.columns else None
        return weighted_portfolio_returns(returns, {h.ticker: h.weight for h in s})

    a = series_of(req.series_a)
    b = series_of(req.series_b)
    empty = RollingCorrResponse(dates=[], corr=[], stress_windows=[], n_obs=0, status="insufficient_data")
    if a is None or b is None:
        return empty

    a_al, b_al = a.align(b, join="inner")
    valid = a_al.notna() & b_al.notna()
    a_al, b_al = a_al[valid], b_al[valid]
    n = int(len(a_al))
    if n < req.window:                                  # not even one full window
        return RollingCorrResponse(dates=[], corr=[], stress_windows=[], n_obs=n, status="insufficient_data")

    rc = a_al.rolling(req.window).corr(b_al)
    dates = [d.strftime("%Y-%m-%d") for d in a_al.index]
    corr = [None if (v != v) else round(float(v), 4) for v in rc.values]  # v!=v -> NaN

    stress_windows = []
    if stress_mask is not None:
        sm = list(stress_mask.reindex(a_al.index).fillna(False).values)
        stress_windows = contiguous_true_ranges(sm, dates)

    return RollingCorrResponse(dates=dates, corr=corr, stress_windows=stress_windows,
                               n_obs=n, status="ok")


@app.post("/api/data/refresh")
async def refresh_data():
    """Force a data reload. Used by the scheduler or admin."""
    _store["prices"] = None
    _store["volumes"] = None
    _store["returns"] = None
    _store["normal_returns"] = None
    _store["stress_returns"] = None
    _store["stress_mask"] = None
    return {"status": "cleared", "message": "Next request will reload data"}


# ── Prices Endpoint ──────────────────────────────────────────────────────────

_RANGE_TRADING_DAYS = {
    "1W": 5, "1M": 21, "1Q": 63, "1Y": 252, "3Y": 756, "5Y": None,
}


@app.post("/api/prices", response_model=PricesResponse)
async def get_prices(req: PricesRequest):
    """
    Return historical close prices for the requested tickers, aligned to
    shared dates. Wide format: one array per ticker, all indexed by `dates`.
    Tickers with no price data are reported in `skipped`, not erroring.
    """
    ck = cache_key("prices", req.tickers, range=req.range, vol=int(req.include_volume))
    cached = cache_get(ck)
    if cached:
        return PricesResponse(**cached)

    store_prices = _store["prices"]
    needs_fresh = store_prices is None or any(
        t not in store_prices.columns for t in req.tickers
    )
    if len(req.tickers) <= 5 and needs_fresh:
        loop = asyncio.get_running_loop()
        prices_df, volumes_df = await loop.run_in_executor(
            None,
            lambda: fetch_prices(
                req.tickers, include_hedges=False, return_volumes=True
            ),
        )
    else:
        _ensure_data(req.tickers)
        prices_df = _store["prices"]
        volumes_df = _store["volumes"]

    available = [t for t in req.tickers if t in prices_df.columns]
    skipped = [t for t in req.tickers if t not in prices_df.columns]

    if not available:
        raise HTTPException(
            status_code=422,
            detail=f"None of the requested tickers have price data: {req.tickers}",
        )

    sliced = prices_df[available]

    if req.range == "common":
        sliced = sliced.dropna()
    else:
        days = _RANGE_TRADING_DAYS[req.range]
        if days is not None:
            sliced = sliced.tail(days)

    if sliced.empty:
        raise HTTPException(
            status_code=422,
            detail="No overlapping price data for requested tickers in this range",
        )

    volume_out = None
    if req.include_volume and volumes_df is not None:
        vol_sliced = volumes_df.reindex(index=sliced.index)[available]
        volume_out = {t: vol_sliced[t].fillna(0).tolist() for t in available}

    response = PricesResponse(
        dates=[d.strftime("%Y-%m-%d") for d in sliced.index],
        prices={t: sliced[t].tolist() for t in available},
        start_date=sliced.index[0].strftime("%Y-%m-%d"),
        end_date=sliced.index[-1].strftime("%Y-%m-%d"),
        skipped=skipped,
        volume=volume_out,
    )

    cache_set(ck, response.model_dump(), CACHE_TTL_DAILY)
    return response


# ── Ticker Search + On-Demand Add ────────────────────────────────────────────

_TICKERS_PATH = Path(__file__).parent / "data" / "tickers.json"
try:
    with open(_TICKERS_PATH) as _f:
        _TICKERS = json.load(_f)
    logger.info(f"Loaded {len(_TICKERS)} tickers from {_TICKERS_PATH.name}")
except (FileNotFoundError, json.JSONDecodeError) as _exc:
    logger.warning(f"Could not load tickers.json ({_exc}); search will return empty")
    _TICKERS = []

# Market-cap snapshot (generated by build_caps.py, which lives outside the repo).
# Middle tier of the equilibrium-weight cap resolution: live fetch -> this
# snapshot -> basket-median. Render's yfinance egress is unreliable, so this
# curated snapshot keeps equilibrium weights market-cap-weighted instead of
# collapsing to the flat median. Shape: {"asof": "YYYY-MM-DD", "caps": {t: n}}.
_CAPS_PATH = Path(__file__).parent / "data" / "caps.json"
try:
    with open(_CAPS_PATH) as _f:
        _caps_doc = json.load(_f)
    _CAPS_ASOF = _caps_doc.get("asof")
    _CAPS_SNAPSHOT = {t: float(v) for t, v in (_caps_doc.get("caps") or {}).items() if v}
    logger.info(f"Loaded caps.json: {len(_CAPS_SNAPSHOT)} caps, asof={_CAPS_ASOF}")
except (FileNotFoundError, json.JSONDecodeError) as _exc:
    logger.warning(f"Could not load caps.json ({_exc}); snapshot cap tier disabled")
    _CAPS_ASOF = None
    _CAPS_SNAPSHOT = {}


@app.post("/api/search", response_model=list[SearchResult])
async def search_tickers(req: SearchRequest):
    """Substring search across NASDAQ/NYSE universe. Top 20 matches, ranked by
    exact-symbol > symbol-prefix > name-contains."""
    q = req.q.strip().lower()
    if not q:
        return []

    ck = cache_key("search", [q])
    cached = cache_get(ck)
    if cached is not None:
        # in_universe may be stale but daily TTL is acceptable
        return [SearchResult(**r) for r in cached]

    exact, prefix, contains = [], [], []
    for r in _TICKERS:
        sym_l = r["symbol"].lower()
        if sym_l == q:
            exact.append(r)
        elif sym_l.startswith(q):
            prefix.append(r)
        elif q in r["name"].lower():
            contains.append(r)

    exact.sort(key=lambda x: x["symbol"])
    prefix.sort(key=lambda x: x["symbol"])
    contains.sort(key=lambda x: x["symbol"])
    ranked = (exact + prefix + contains)[:20]

    universe: set[str] = set()
    if _store["returns"] is not None:
        universe = set(_store["returns"].columns)

    results = [
        SearchResult(
            symbol=r["symbol"],
            name=r["name"],
            exchange=r["exchange"],
            in_universe=r["symbol"] in universe,
        )
        for r in ranked
    ]

    cache_set(ck, [r.model_dump() for r in results], CACHE_TTL_DAILY)
    return results


def _sharpe_band(sharpe: float) -> str:
    """Descriptive Sharpe (1Y) band — factual replacement for the retired
    verdict labels (INTERFACE_VERDICTS_SPEC.md A2 / Amendment 1). Closed set."""
    if sharpe >= 1.5:
        return "High"
    if sharpe >= 1.0:
        return "Moderate"
    if sharpe >= 0.5:
        return "Low"
    if sharpe >= 0.0:
        return "Minimal"
    return "Negative"


@app.post("/api/ticker/add", response_model=TickerAddResponse)
async def add_ticker(req: TickerAddRequest):
    """Fetch a single ticker via yfinance. Lenient validation — accepts any symbol
    so mutual funds (VTSAX etc.) and other securities not in tickers.json still work."""
    symbol = req.symbol.strip().upper()
    if not symbol:
        raise HTTPException(status_code=422, detail="Empty symbol")

    ck = cache_key("ticker_add", [symbol])
    cached = cache_get(ck)
    if cached is not None:
        return TickerAddResponse(**cached)

    import yfinance as yf

    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="1y")
    except Exception as exc:
        logger.error(f"yfinance history({symbol}) failed: {exc}")
        raise HTTPException(status_code=502, detail=f"Fetch failed for {symbol}: {exc}")

    if hist is None or hist.empty or "Close" not in hist.columns:
        raise HTTPException(status_code=502, detail=f"No price history returned for {symbol}")

    closes = hist["Close"].dropna()
    if len(closes) < 20:
        raise HTTPException(status_code=422, detail=f"Insufficient history for {symbol}: {len(closes)} days")

    ret_1y = float((closes.iloc[-1] / closes.iloc[0] - 1.0) * 100)
    returns = np.log(closes / closes.shift(1)).dropna()
    ann_vol = float(returns.std() * np.sqrt(252) * 100)

    beta = 1.0
    if _store["returns"] is not None and "QQQ" in _store["returns"].columns:
        qqq_ret = _store["returns"]["QQQ"]
        common_idx = returns.index.intersection(qqq_ret.index)
        if len(common_idx) > 30:
            r1 = returns.loc[common_idx].values
            r2 = qqq_ret.loc[common_idx].values
            var_qqq = float(np.var(r2))
            if var_qqq > 0:
                beta = float(np.cov(r1, r2)[0, 1] / var_qqq)

    rf_pct = RISK_FREE_RATE * 100
    sharpe = float((ret_1y - rf_pct) / ann_vol) if ann_vol > 0 else 0.0

    cum = (1.0 + returns).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = float(drawdown.min() * 100) if len(drawdown) > 0 else 0.0

    info: dict = {}
    try:
        info = t.info or {}
    except Exception as exc:
        logger.debug(f"yfinance info({symbol}) failed: {exc}")

    sector = info.get("sector") or info.get("quoteType") or "Unknown"
    name = info.get("longName") or info.get("shortName") or symbol
    div_yield = info.get("dividendYield") or 0
    try:
        div_yield = float(div_yield)
    except (TypeError, ValueError):
        div_yield = 0.0
    if div_yield > 25:
        div_yield = div_yield / 100

    pe = info.get("trailingPE")
    fpe = info.get("forwardPE")
    mkt_cap = info.get("marketCap")
    quote_type = (info.get("quoteType") or "").upper()
    t_type = "ETF" if quote_type == "ETF" else "Mutual Fund" if quote_type == "MUTUALFUND" else "Stock"

    sig = _sharpe_band(sharpe)  # Sharpe band, not a verdict (INTERFACE_VERDICTS_SPEC.md Amendment 1)

    response = TickerAddResponse(
        id=symbol,
        n=name,
        t=t_type,
        sec=sector,
        ret=round(ret_1y, 2),
        vol=round(ann_vol, 2),
        beta=round(beta, 2),
        sh=round(sharpe, 2),
        yld=round(div_yield, 2),
        sig=sig,
        goals=[], rb=[1, 5], hm=3,
        s08=0.0, s20=0.0, s22=0.0,
        fit=50,
        pe=float(pe) if pe else None,
        fpe=float(fpe) if fpe else None,
        mktCap=float(mkt_cap) if mkt_cap else None,
        maxDD=round(max_dd, 2),
    )

    cache_set(ck, response.model_dump(), CACHE_TTL_DAILY)
    return response


@app.post("/api/ticker/news", response_model=NewsResponse)
async def ticker_news(req: NewsRequest):
    """Return top 10 recent news items for a ticker via yfinance.
    Cached daily. Returns empty list on failure rather than erroring."""
    symbol = req.symbol.strip().upper()
    if not symbol:
        raise HTTPException(status_code=422, detail="Empty symbol")

    ck = cache_key("ticker_news", [symbol])
    cached = cache_get(ck)
    if cached is not None:
        return NewsResponse(**cached)

    import yfinance as yf
    items: list[NewsItem] = []
    try:
        raw = yf.Ticker(symbol).news or []
        for article in raw[:10]:
            title = None
            link = None
            publisher = None
            published = None
            summary = None

            # Newer yfinance wraps fields in a 'content' sub-dict
            content = article.get("content") if isinstance(article, dict) else None
            if isinstance(content, dict):
                title = content.get("title")
                summary = content.get("summary") or content.get("description")
                published = content.get("pubDate")
                provider_obj = content.get("provider") or {}
                if isinstance(provider_obj, dict):
                    publisher = provider_obj.get("displayName")
                url_obj = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
                if isinstance(url_obj, dict):
                    link = url_obj.get("url")

            # Older yfinance uses flat keys
            if not title:
                title = article.get("title")
            if not link:
                link = article.get("link")
            if not publisher:
                publisher = article.get("publisher")
            if not published:
                ts = article.get("providerPublishTime")
                if ts:
                    try:
                        published = datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
                    except (ValueError, TypeError, OSError, OverflowError):
                        published = None
            if not summary:
                summary = article.get("summary")

            if not title or not link:
                continue

            items.append(NewsItem(
                title=title,
                link=link,
                publisher=publisher or "Unknown",
                published=published,
                summary=summary,
            ))
    except Exception as exc:
        logger.debug(f"yfinance news({symbol}) failed: {exc}")

    response = NewsResponse(items=items)
    cache_set(ck, response.model_dump(), CACHE_TTL_DAILY)
    return response


# ── Optimizer Endpoints ──────────────────────────────────────────────────────

from app.optimizer import (
    Objective,
    ProfileConstraints,
    optimize_portfolio,
    optimize_multi_objective,
    generate_efficient_frontier,
)
from app.data_ingest import annualized_volatility as compute_ann_vol, compute_return_stats
from app.return_models import ledoit_wolf_constant_correlation, black_litterman
from app.config import PRICE_HISTORY_YEARS


def _build_profile_constraints(
    req_constraints, risk_score: int | None
) -> ProfileConstraints:
    """
    Build optimizer constraints from request input.
    If risk_score is provided, derive max_position and max_beta from it
    (matching the frontend's scoreProfile logic).
    """
    if req_constraints:
        pc = ProfileConstraints(
            max_position_pct=req_constraints.max_position_pct,
            min_position_pct=req_constraints.min_position_pct,
            max_beta=req_constraints.max_beta,
            max_volatility=req_constraints.max_volatility,
            excluded_tickers=req_constraints.excluded_tickers,
        )
    else:
        pc = ProfileConstraints()

    # Override from risk_score if provided and no explicit constraints given
    if risk_score is not None and req_constraints is None:
        score = max(0, min(100, risk_score))
        pc.max_position_pct = 0.30 if score >= 70 else 0.20 if score >= 50 else 0.15
        pc.max_beta = round(0.4 + score * 0.014, 2)

    return pc


def _get_expected_returns_and_betas(
    returns_df, tickers: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute annualized expected returns and betas from historical data.
    """
    available = [t for t in tickers if t in returns_df.columns]
    # Include the beta benchmark in the subset so compute_return_stats can
    # measure beta against it (it survives the per-request column subsetting).
    cols = available + ([BETA_BENCHMARK] if BETA_BENCHMARK in returns_df.columns
                        and BETA_BENCHMARK not in available else [])
    stats = compute_return_stats(returns_df[cols], benchmark=BETA_BENCHMARK)

    exp_ret = np.array([stats.loc[t, "ann_return"] if t in stats.index else 0.0 for t in tickers])
    betas = np.array([stats.loc[t, "beta"] if t in stats.index else 1.0 for t in tickers])

    return exp_ret, betas


def _get_market_caps(tickers: list[str]) -> tuple[np.ndarray, str | None, str | None]:
    """
    Effective-size market caps for the Black-Litterman market weights.

    Three-tier resolution per ticker:
      1. Live fetch — stored fundamentals cache, then a yfinance `info` fetch
         (marketCap, ETF fallback totalAssets).
      2. caps.json snapshot — curated market caps for the universe, generated
         locally by build_caps.py (Render's yfinance egress is unreliable).
      3. Basket-median — a ticker missing from BOTH live and snapshot gets the
         basket-median positive cap and is named in the constraints note.

    caps_source names the snapshot ("snapshot <asof>") whenever tier 2 supplied
    at least one ticker. The constraints note now fires ONLY on the true
    median fallback (tier 3).

    Returns (caps ndarray aligned to `tickers`, note-or-None, caps_source-or-None).
    """
    import yfinance as yf

    cache = _store.get("fundamentals") or {}
    caps: dict[str, float] = {}
    to_fetch = []
    for t in tickers:
        cached = cache.get(t)
        if cached and cached > 0:
            caps[t] = float(cached)
        else:
            to_fetch.append(t)

    if to_fetch:
        try:
            tobj = yf.Tickers(" ".join(to_fetch))
            for t in to_fetch:
                eff = 0.0
                try:
                    obj = tobj.tickers.get(t)
                    info = (obj.info if obj is not None else {}) or {}
                    eff = float(info.get("marketCap") or info.get("totalAssets") or 0.0)
                except Exception as exc:
                    logger.debug(f"market cap fetch failed for {t}: {exc}")
                caps[t] = eff
                if eff > 0:
                    cache[t] = eff
        except Exception as exc:
            logger.warning(f"batch market-cap fetch failed: {exc}")
            for t in to_fetch:
                caps.setdefault(t, 0.0)
    _store["fundamentals"] = cache

    # Tier 2: fill live gaps from the curated snapshot before touching the median.
    from_snapshot = []
    for t in tickers:
        if caps.get(t, 0.0) <= 0:
            snap = _CAPS_SNAPSHOT.get(t)
            if snap and snap > 0:
                caps[t] = snap
                from_snapshot.append(t)
    caps_source = f"snapshot {_CAPS_ASOF}" if from_snapshot else None

    positive = [caps[t] for t in tickers if caps.get(t, 0.0) > 0]
    median_cap = float(np.median(positive)) if positive else 1.0

    # Tier 3: only tickers missing from live AND snapshot hit the median fallback.
    missing = [t for t in tickers if caps.get(t, 0.0) <= 0]
    note = None
    if missing:
        for t in missing:
            caps[t] = median_cap
        note = (
            f"market cap unavailable for {', '.join(missing)} — "
            f"substituted basket-median size for equilibrium weights"
        )

    return np.array([caps[t] for t in tickers], dtype=float), note, caps_source


def _build_black_litterman_inputs(
    returns, tickers: list[str], apply_outlook: bool, view_confidence: float,
) -> dict:
    """
    Assemble the Black-Litterman (μ, Σ) for the optimizer.

    Σ = Ledoit-Wolf constant-correlation shrinkage of the sample covariance over
    the available return window (max 5 years, min 1 year), annualized.
    μ = equilibrium prior π (apply_outlook off) or the BL posterior with the
    Co-CIO Outlook sector tilts recast as views at `view_confidence`.

    Betas remain historical (they gate the profile beta cap, not the returns).
    Returns a dict with tickers, sigma, mu, betas, n_days, shrinkage_alpha, note.
    """
    from app.coc_io_outlook import SECTOR_TILTS_2026
    from app.screener import _SECTORS

    available = [t for t in tickers if t in returns.columns]
    if len(available) < 2:
        raise ValueError("Fewer than 2 requested tickers have return data for Black-Litterman")

    window_df = returns[available].dropna().tail(PRICE_HISTORY_YEARS * 252)
    if window_df.shape[0] < 60:
        raise ValueError("Insufficient overlapping return history for Black-Litterman shrinkage")

    sigma, alpha, n_days = ledoit_wolf_constant_correlation(window_df.values)

    caps, caps_note, caps_source = _get_market_caps(available)

    # Sector tilts as views (zeros when the outlook is off -> posterior == π).
    if apply_outlook:
        tilts = np.array([
            SECTOR_TILTS_2026.get((_SECTORS.get(t) or {}).get("sector"), 0.0)
            for t in available
        ])
    else:
        tilts = np.zeros(len(available))

    mu = black_litterman(sigma, caps, tilts, view_confidence=view_confidence)

    # Historical betas aligned to `available`.
    _, betas = _get_expected_returns_and_betas(returns, available)

    return {
        "tickers": available,
        "sigma": sigma,
        "mu": mu,
        "betas": betas,
        "n_days": n_days,
        "shrinkage_alpha": round(float(alpha), 6),
        "note": caps_note,
        "caps_source": caps_source,
    }


def _compute_current_portfolio_stats(
    weights_pct: dict[str, float],
    tickers: list[str],
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    betas: np.ndarray,
) -> dict:
    """Compute stats for the user's current weights for before/after comparison."""
    from app.optimizer import _portfolio_return, _portfolio_volatility, _portfolio_sharpe

    w = np.array([weights_pct.get(t, 0.0) / 100.0 for t in tickers])
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum
    else:
        return None

    ret = _portfolio_return(w, expected_returns)
    vol = _portfolio_volatility(w, cov_matrix)
    sh = _portfolio_sharpe(w, expected_returns, cov_matrix)
    beta = float(w @ betas)

    return {
        "portfolio_return": round(ret, 6),
        "portfolio_volatility": round(vol, 6),
        "sharpe_ratio": round(sh, 4),
        "beta": round(beta, 4),
    }


@app.post("/api/optimize", response_model=OptimizeResponse)
async def optimize(req: OptimizeRequest):
    """
    Optimize portfolio weights using scipy SLSQP with the blended
    covariance matrix. Returns optimal weights and full metrics.

    Accepts a risk_score to auto-derive both λ (stress blending) and
    profile constraints (max position, max beta).
    """
    _ensure_data(req.tickers)
    returns = _store["returns"]
    normal_returns = _store["normal_returns"]
    stress_returns = _store["stress_returns"]

    # Build constraints
    constraints = _build_profile_constraints(req.constraints, req.risk_score)

    # Compute λ for covariance blending
    lam = lambda_from_risk_score(req.risk_score) if req.risk_score is not None else DEFAULT_LAMBDA

    # Black-Litterman metadata (populated only under return_model="black_litterman").
    bl_n_days: int | None = None
    bl_alpha: float | None = None
    bl_caps_note: str | None = None
    bl_caps_source: str | None = None

    try:
        if req.return_model == "black_litterman":
            # Shrunk covariance + market-implied equilibrium prior (Outlook = views).
            bl = _build_black_litterman_inputs(
                returns, req.tickers, req.apply_outlook, req.view_confidence,
            )
            used_tickers = bl["tickers"]
            used_cov = bl["sigma"]
            exp_ret = bl["mu"]
            betas = bl["betas"]
            bl_n_days = bl["n_days"]
            bl_alpha = bl["shrinkage_alpha"]
            bl_caps_note = bl["note"]
            bl_caps_source = bl["caps_source"]
            logger.info(
                f"Black-Litterman: {len(used_tickers)} tickers, n_days={bl_n_days}, "
                f"alpha={bl_alpha}, outlook={req.apply_outlook}, c={req.view_confidence}"
            )
        else:
            # Historical (default) — byte-identical prior behavior.
            cov_result = build_covariance_matrix(
                normal_returns, stress_returns, req.tickers,
                lam=lam, window=req.window,
            )
            used_tickers = cov_result.tickers
            used_cov = cov_result.matrix

            # Get expected returns and betas from historical data
            exp_ret, betas = _get_expected_returns_and_betas(returns, used_tickers)

            # Apply 2026 Co-CIO Outlook sector tilts to expected returns if requested
            if req.apply_outlook:
                from app.coc_io_outlook import apply_outlook_tilts
                from app.screener import _SECTORS
                sectors_map = {t: (_SECTORS.get(t) or {}).get("sector") for t in used_tickers}
                exp_ret = apply_outlook_tilts(exp_ret, used_tickers, sectors_map)
                logger.info(f"Applied 2026 Co-CIO Outlook tilts to {len(used_tickers)} tickers")

        # Map objective string to enum
        obj = Objective(req.objective)

        # Run optimizer
        result = optimize_portfolio(
            tickers=used_tickers,
            expected_returns=exp_ret,
            cov_matrix=used_cov,
            betas=betas,
            objective=obj,
            constraints=constraints,
            apply_outlook=req.apply_outlook,
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")

    # Fold the equilibrium-weight fallback note into the response note honestly.
    constraints_note = result.constraints_note
    if bl_caps_note:
        constraints_note = (
            bl_caps_note if constraints_note is None
            else f"{constraints_note}; {bl_caps_note}"
        )

    # Optional: compute comparison with current weights
    comparison = None
    if req.include_current_weights:
        comparison = _compute_current_portfolio_stats(
            req.include_current_weights, used_tickers,
            exp_ret, used_cov, betas,
        )

    return OptimizeResponse(
        weights=result.weights,
        objective=result.objective,
        portfolio_return=result.portfolio_return,
        portfolio_volatility=result.portfolio_volatility,
        sharpe_ratio=result.sharpe_ratio,
        beta=result.beta,
        var_95=result.var_95,
        cvar_95=result.cvar_95,
        risk_contributions=result.risk_contributions,
        solver_success=result.solver_success,
        solver_message=result.solver_message,
        iterations=result.iterations,
        constraints_active=result.constraints_active,
        constraints_note=constraints_note,
        comparison=comparison,
        apply_outlook=req.apply_outlook,
        return_model_used=req.return_model,
        n_days=bl_n_days,
        shrinkage_alpha=bl_alpha,
        caps_source=bl_caps_source,
    )


# Sandbox regime bounds (spec section 1). Enforced here (API) and in the UI.
_TILT_BOUND_PP = 5.0          # tilts in percentage points
_VOL_MULT_BOUNDS = (0.5, 2.0)
_RHO_BOUNDS = (-0.95, 0.95)


def _validate_regime_bounds(regime: RegimeInput) -> None:
    """422 (naming the field) when any regime parameter is out of bounds."""
    for sec, pp in regime.tilts.items():
        if not (-_TILT_BOUND_PP <= pp <= _TILT_BOUND_PP):
            raise HTTPException(
                status_code=422,
                detail=f"tilts['{sec}'] = {pp}pp out of bounds [-5.0, +5.0]",
            )
    lo, hi = _VOL_MULT_BOUNDS
    for sec, m in regime.vol_multipliers.items():
        if not (lo <= m <= hi):
            raise HTTPException(
                status_code=422,
                detail=f"vol_multipliers['{sec}'] = {m} out of bounds [0.5, 2.0]",
            )
    rlo, rhi = _RHO_BOUNDS
    for k, ov in enumerate(regime.correlation_overrides):
        if not (rlo <= ov.rho <= rhi):
            raise HTTPException(
                status_code=422,
                detail=f"correlation_overrides[{k}].rho = {ov.rho} out of bounds [-0.95, +0.95]",
            )


def _build_internal_regime(regime: RegimeInput) -> dict:
    """RegimeInput (tilts in pp) -> engine regime dict (tilts in decimal)."""
    return {
        "name": regime.name,
        "tilts": {sec: pp / 100.0 for sec, pp in regime.tilts.items()},
        "vol_multipliers": dict(regime.vol_multipliers),
        "correlation_overrides": [
            {"a": ov.a, "b": ov.b, "rho": ov.rho} for ov in regime.correlation_overrides
        ],
    }


def _run_regime_stress(
    req_tickers: list[str],
    weights_pct: dict[str, float],
    window: int,
    include_attribution: bool,
    regime: dict,
    description: str,
) -> StressTestResponse:
    """
    Shared regime stress engine: baseline vs regime metrics + full T/V/C
    attribution + PSD-guard outcome. Both /api/stress_test_2026 (with the 2026
    preset) and /api/sandbox/stress (with a user regime) call through here, so a
    2026 regime posted to the sandbox is numerically identical to the preset run.
    """
    _ensure_data(req_tickers)
    returns = _store["returns"]
    normal_returns = _store["normal_returns"]
    stress_returns = _store["stress_returns"]

    from app.coc_io_outlook import (
        apply_regime, _regime_cov_states, compute_attribution, monte_carlo_max_drawdown,
    )
    from app.screener import _SECTORS

    cov_result = build_covariance_matrix(
        normal_returns, stress_returns, req_tickers,
        lam=DEFAULT_LAMBDA, window=window,
    )
    tickers = cov_result.tickers
    exp_ret, _betas = _get_expected_returns_and_betas(returns, tickers)

    w = np.array([weights_pct.get(t, 0.0) / 100.0 for t in tickers])
    if w.sum() <= 0:
        raise HTTPException(status_code=422, detail="Weights must sum to a positive value")
    w = w / w.sum()

    sectors_map = {t: (_SECTORS.get(t) or {}).get("sector") for t in tickers}

    # Baseline (historical) metrics
    base_ret = float(w @ exp_ret)
    base_vol = float(np.sqrt(max(w @ cov_result.matrix @ w, 0.0)))
    base_sh = ((base_ret - RISK_FREE_RATE) / base_vol) if base_vol > 1e-10 else 0.0

    # Regime-adjusted metrics (PSD-guarded correlation structure)
    regime_ret_vec, regime_cov = apply_regime(
        exp_ret, cov_result.matrix, tickers, sectors_map, regime)
    regime_ret = float(w @ regime_ret_vec)
    regime_vol = float(np.sqrt(max(w @ regime_cov @ w, 0.0)))
    regime_sh = ((regime_ret - RISK_FREE_RATE) / regime_vol) if regime_vol > 1e-10 else 0.0

    psd_adjustment = _regime_cov_states(
        cov_result.matrix, tickers, sectors_map, regime)["psd_adjustment"]

    # Max drawdown. When attribution is on, its waterfall computes the [none]
    # and [T,V,C] endpoints — reuse them so components sum exactly to the
    # reported Δ (shared seed). Otherwise run the two MC states directly.
    attribution = None
    if include_attribution:
        attribution, base_mdd, regime_mdd = compute_attribution(
            exp_ret, cov_result.matrix, w, tickers, sectors_map, RISK_FREE_RATE,
            regime=regime,
        )
    else:
        base_mdd = monte_carlo_max_drawdown(w, exp_ret, cov_result.matrix, seed=42)
        regime_mdd = monte_carlo_max_drawdown(w, regime_ret_vec, regime_cov, seed=42)

    return StressTestResponse(
        baseline=StressMetrics(
            expected_return=round(base_ret, 6),
            volatility=round(base_vol, 6),
            sharpe_ratio=round(base_sh, 4),
            max_drawdown=round(base_mdd, 6),
        ),
        regime=StressMetrics(
            expected_return=round(regime_ret, 6),
            volatility=round(regime_vol, 6),
            sharpe_ratio=round(regime_sh, 4),
            max_drawdown=round(regime_mdd, 6),
        ),
        description=description,
        attribution=attribution,
        psd_adjustment=psd_adjustment,
    )


@app.post("/api/stress_test_2026", response_model=StressTestResponse)
async def stress_test_2026(req: StressTestRequest):
    """
    Compare the user's current portfolio under baseline (historical) assumptions
    vs the 2026 Co-CIO Outlook regime — sector return tilts, sector vol scaling,
    and a stock-bond correlation override. Max drawdown estimated via 1000-path
    Monte Carlo over a 1-year horizon. Internally runs the shared sandbox engine
    with the REGIME_2026 preset (PSD guard verified per run, expected no-op).
    """
    from app.coc_io_outlook import REGIME_2026
    try:
        return _run_regime_stress(
            req.tickers, req.weights, req.window, req.include_attribution,
            REGIME_2026, REGIME_2026["description"],
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"Stress test failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Stress test failed: {exc}")


@app.post("/api/sandbox/stress", response_model=StressTestResponse)
async def sandbox_stress(req: SandboxStressRequest):
    """
    View Sandbox: run a user-defined regime (tilts / vol multipliers /
    correlation overrides) through the same stress + attribution machinery as the
    2026 preset. Bounds are enforced (422) before the run; the response carries
    the PSD-guard outcome so the UI can disclose any correlation repair.
    """
    _validate_regime_bounds(req.regime)
    internal = _build_internal_regime(req.regime)
    description = req.regime.name or "Custom scenario"
    try:
        return _run_regime_stress(
            req.tickers, req.weights, req.window, req.include_attribution,
            internal, description,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"Sandbox stress failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Sandbox stress failed: {exc}")


@app.post("/api/optimize/all", response_model=MultiOptimizeResponse)
async def optimize_all(req: MultiOptimizeRequest):
    """
    Run all four objectives (Max Sharpe, Max Return, Min Vol, Risk Parity)
    and return results for each. Powers the Compare view.
    """
    _ensure_data(req.tickers)
    returns = _store["returns"]
    normal_returns = _store["normal_returns"]
    stress_returns = _store["stress_returns"]

    constraints = _build_profile_constraints(req.constraints, req.risk_score)
    lam = lambda_from_risk_score(req.risk_score) if req.risk_score is not None else DEFAULT_LAMBDA

    try:
        cov_result = build_covariance_matrix(
            normal_returns, stress_returns, req.tickers,
            lam=lam, window=req.window,
        )
        exp_ret, betas = _get_expected_returns_and_betas(returns, cov_result.tickers)

        results = optimize_multi_objective(
            tickers=cov_result.tickers,
            expected_returns=exp_ret,
            cov_matrix=cov_result.matrix,
            betas=betas,
            constraints=constraints,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    response_results = {}
    for obj_name, result in results.items():
        response_results[obj_name] = OptimizeResponse(
            weights=result.weights,
            objective=result.objective,
            portfolio_return=result.portfolio_return,
            portfolio_volatility=result.portfolio_volatility,
            sharpe_ratio=result.sharpe_ratio,
            beta=result.beta,
            var_95=result.var_95,
            cvar_95=result.cvar_95,
            risk_contributions=result.risk_contributions,
            solver_success=result.solver_success,
            solver_message=result.solver_message,
            iterations=result.iterations,
            constraints_active=result.constraints_active,
        )

    return MultiOptimizeResponse(results=response_results)


@app.post("/api/frontier", response_model=FrontierResponse)
async def efficient_frontier(req: FrontierRequest):
    """
    Trace the efficient frontier for the given tickers.
    Returns n_points along the frontier plus the Max Sharpe and Min Vol
    optimal points and (optionally) the user's current portfolio position.
    """
    _ensure_data(req.tickers)
    returns = _store["returns"]
    normal_returns = _store["normal_returns"]
    stress_returns = _store["stress_returns"]

    constraints = _build_profile_constraints(req.constraints, req.risk_score)
    lam = lambda_from_risk_score(req.risk_score) if req.risk_score is not None else DEFAULT_LAMBDA

    try:
        cov_result = build_covariance_matrix(
            normal_returns, stress_returns, req.tickers,
            lam=lam, window=req.window,
        )
        exp_ret, betas = _get_expected_returns_and_betas(returns, cov_result.tickers)

        # Trace the frontier
        frontier = generate_efficient_frontier(
            tickers=cov_result.tickers,
            expected_returns=exp_ret,
            cov_matrix=cov_result.matrix,
            betas=betas,
            constraints=constraints,
            n_points=req.n_points,
        )

        # Get the two key optimal points
        max_sh_result = optimize_portfolio(
            cov_result.tickers, exp_ret, cov_result.matrix, betas,
            Objective.MAX_SHARPE, constraints,
        )
        min_vol_result = optimize_portfolio(
            cov_result.tickers, exp_ret, cov_result.matrix, betas,
            Objective.MIN_VOLATILITY, constraints,
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Current portfolio position on the frontier
    current = None
    if req.current_weights:
        current = _compute_current_portfolio_stats(
            req.current_weights, cov_result.tickers,
            exp_ret, cov_result.matrix, betas,
        )

    return FrontierResponse(
        frontier=frontier,
        max_sharpe={
            "return": max_sh_result.portfolio_return,
            "volatility": max_sh_result.portfolio_volatility,
            "sharpe": max_sh_result.sharpe_ratio,
            "weights": max_sh_result.weights,
        },
        min_vol={
            "return": min_vol_result.portfolio_return,
            "volatility": min_vol_result.portfolio_volatility,
            "sharpe": min_vol_result.sharpe_ratio,
            "weights": min_vol_result.weights,
        },
        current_portfolio=current,
    )


# ── Screening Endpoint ───────────────────────────────────────────────────────

from app.screener import run_screening_pipeline

# Time horizon string → approximate years (matches frontend Onboard)
_HORIZON_MAP = {
    "<1 year": 0.75,
    "1-3 years": 2.0,
    "3-7 years": 5.0,
    "7-15 years": 10.0,
    "15+ years": 20.0,
}


@app.post("/api/screen", response_model=ScreenResponse)
async def screen_universe(req: ScreenRequest):
    """
    Run the full screening pipeline against the Nasdaq universe.

    Takes the investor profile (goal, risk_score, time_horizon) and returns
    a ranked shortlist of 30-40 candidates with fit scores, factor exposures,
    and fundamental data.

    Pipeline stages:
      0. Load universe (~150-3,700 tickers depending on data source)
      1. Hard gates: market cap ≥ $2B, volume ≥ 500K, price ≥ $5
      2. Factor scoring: momentum, quality, value, low-vol, yield → z-scores
      3. Profile-aware fit scoring: goal-weighted composite → 0-100 fit score

    This endpoint is expensive (~30-60s on first call) due to fundamental
    data fetching. Results are cached.
    """
    ck = cache_key("screen", [req.goal], risk=req.risk_score,
                   horizon=f"{req.time_horizon_years:.1f}", max=req.max_results)
    cached = cache_get(ck)
    if cached:
        return ScreenResponse(**cached)

    # Ensure the return data actually covers the requested tickers. Always
    # call _ensure_data — its own missing-set logic (see _ensure_data) decides
    # whether a fetch is needed, so this is a no-op when the store is already
    # warm. Previously this only ran when the store was empty, which silently
    # dropped any requested ticker that wasn't in a prior request's basket.
    loop = asyncio.get_running_loop()
    from app.screener import load_nasdaq_tickers
    tickers = req.tickers or load_nasdaq_tickers()
    await loop.run_in_executor(None, _ensure_data, tickers)

    returns = _store["returns"]
    prices = _store["prices"]

    try:
        result = await loop.run_in_executor(
            None,
            lambda: run_screening_pipeline(
                returns=returns,
                goal=req.goal,
                risk_score=req.risk_score,
                time_horizon_years=req.time_horizon_years,
                max_results=req.max_results,
                tickers=req.tickers,
                prices=prices,
            ),
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Screening failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Screening pipeline failed: {e}")

    response = ScreenResponse(
        shortlist=[
            ScreenedAssetResponse(
                ticker=a.ticker, name=a.name, sector=a.sector, industry=a.industry,
                asset_type=a.asset_type,
                market_cap=a.market_cap, price=a.price, avg_volume=a.avg_volume,
                return_1y=a.return_1y, return_6m=a.return_6m, return_3m=a.return_3m,
                volatility=a.volatility, sharpe=a.sharpe, beta=a.beta,
                max_drawdown=a.max_drawdown, dividend_yield=a.dividend_yield,
                pe_ratio=a.pe_ratio, forward_pe=a.forward_pe,
                pb_ratio=a.pb_ratio, earnings_yield=a.earnings_yield,
                earnings_date=a.earnings_date,
                dividend_growth_5y=a.dividend_growth_5y,
                debt_to_equity=a.debt_to_equity,
                revenue_growth=a.revenue_growth,
                return_3y=a.return_3y, return_5y=a.return_5y,
                max_dd_5y=a.max_dd_5y, dd_window_days=a.dd_window_days,
                pct_off_52wk_high=a.pct_off_52wk_high,
                z_momentum=a.z_momentum, z_quality=a.z_quality,
                z_value=a.z_value, z_low_vol=a.z_low_vol, z_yield=a.z_yield,
                factor_composite=a.factor_composite, fit_score=a.fit_score, rank=a.rank,
            )
            for a in result.shortlist
        ],
        universe_size=result.universe_size,
        passed_gates=result.passed_gates,
        stage2_size=result.stage2_size,
        final_size=result.final_size,
        goal_used=result.goal_used,
        factor_weights=result.factor_weights,
        sector_distribution=result.sector_distribution,
    )

    # Cache for 18 hours (refreshed daily after market close)
    cache_set(ck, response.model_dump(), CACHE_TTL_DAILY)
    return response


# ── Fundamental Analysis Endpoints ───────────────────────────────────────────

from app.fundamental import analyze_company, analyze_batch, analyze_sector, generate_fallback_analysis


@app.post("/api/analyze", response_model=CompanyAnalysisResponse)
async def analyze_single(req: AnalyzeSingleRequest):
    """
    Run full 3-step fundamental analysis on a single company.
    Requires an Anthropic API key.

    Pipeline:
      Step 0: Fetch financial data from yfinance
      Step 1: Claude analyzes financials → structured metrics
      Step 2: Claude performs SWOT + competitive assessment
      Step 3: Claude synthesizes → conviction score + recommendation

    Sector-level PESTEL and Porter's Five Forces are computed once per sector
    and cached for 90 days.
    """
    try:
        result = await analyze_company(
            req.ticker, req.api_key, req.risk_score, req.goal,
        )
    except Exception as e:
        logger.error(f"Analysis failed for {req.ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    return CompanyAnalysisResponse(
        ticker=result.ticker, name=result.name, sector=result.sector,
        financial_summary=result.financial_summary,
        financial_strengths=result.financial_strengths,
        financial_weaknesses=result.financial_weaknesses,
        swot=result.swot, competitive_position=result.competitive_position,
        moat_assessment=result.moat_assessment,
        conviction_score=result.conviction_score,
        conviction_reasoning=result.conviction_reasoning,
        key_catalysts=result.key_catalysts, key_risks=result.key_risks,
        recommendation=result.recommendation,
        price_target_rationale=result.price_target_rationale,
        analysis_source=result.analysis_source,
    )


@app.post("/api/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_multiple(req: AnalyzeRequest):
    """
    Run fundamental analysis on multiple tickers (up to 20).
    Processes sequentially to respect API rate limits.
    Returns sorted by conviction score descending.
    """
    try:
        results = await analyze_batch(
            req.tickers, req.api_key, req.risk_score, req.goal,
        )
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {e}")

    analyses = [
        CompanyAnalysisResponse(
            ticker=r.ticker, name=r.name, sector=r.sector,
            financial_summary=r.financial_summary,
            financial_strengths=r.financial_strengths,
            financial_weaknesses=r.financial_weaknesses,
            swot=r.swot, competitive_position=r.competitive_position,
            moat_assessment=r.moat_assessment,
            conviction_score=r.conviction_score,
            conviction_reasoning=r.conviction_reasoning,
            key_catalysts=r.key_catalysts, key_risks=r.key_risks,
            recommendation=r.recommendation,
            price_target_rationale=r.price_target_rationale,
            analysis_source=r.analysis_source,
        )
        for r in results
    ]

    scores = [a.conviction_score for a in analyses if a.analysis_source != "error"]
    avg = sum(scores) / max(len(scores), 1)

    return BatchAnalysisResponse(
        analyses=analyses,
        total_requested=len(req.tickers),
        total_completed=len([a for a in analyses if a.analysis_source != "error"]),
        avg_conviction=round(avg, 1),
    )


@app.post("/api/analyze/sector", response_model=SectorAnalysisResponse)
async def analyze_sector_endpoint(sector: str, api_key: str):
    """
    Get PESTEL and Porter's Five Forces for a sector.
    Cached for 90 days. Shared across all companies in the sector.
    """
    try:
        result = await analyze_sector(sector, api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sector analysis failed: {e}")

    return SectorAnalysisResponse(
        sector=result.sector, pestel=result.pestel, porters=result.porters,
        sector_outlook=result.sector_outlook, key_trends=result.key_trends,
    )


# ── Paper Trading Endpoints ───────────────────────────────────────────────────

from app.paper_trading import (
    PaperAccount, OrderSide, OrderType, TIF,
    get_cfa_tag, get_all_cfa_tags, get_batch_prices,
)
from app.models import (
    OrderSubmitRequest, OrderResponse, OrderListResponse,
    CancelOrderRequest, CheckOrdersResponse,
)

# In-memory account store (keyed by account_id)
# In production, persist to database
_accounts: dict[str, PaperAccount] = {}


def _get_or_create_account(account_id: str = "default") -> PaperAccount:
    """Get existing account or create a new one."""
    if account_id not in _accounts:
        _accounts[account_id] = PaperAccount(account_id=account_id)
    return _accounts[account_id]


def _check_pending_against_live(acct: PaperAccount) -> None:
    """Fetch current prices for any pending-order tickers and check fills.
    Called as a side effect from price-fetching endpoints — no background task."""
    pending = acct.get_pending_orders()
    if not pending:
        return
    tickers = sorted({o["ticker"] for o in pending})
    prices = get_batch_prices(tickers)
    acct.check_pending_orders(prices)


@app.post("/api/paper/trade", response_model=TradeResponse)
async def paper_trade(req: TradeRequest, account_id: str = "default"):
    """
    Execute a paper trade at real market price.
    Buy or sell shares of any ticker. Price fetched from yfinance if not provided.
    Also checks any pending limit/stop orders against current prices.
    """
    acct = _get_or_create_account(account_id)
    _check_pending_against_live(acct)
    side = OrderSide.BUY if req.side == "buy" else OrderSide.SELL
    tx = acct.execute_trade(req.ticker.upper(), side, req.quantity, req.price)

    return TradeResponse(
        id=tx.id, timestamp=tx.timestamp, ticker=tx.ticker,
        side=tx.side, quantity=tx.quantity, price=tx.price,
        total=tx.total, commission=tx.commission, status=tx.status,
        reason=tx.reason, balance_after=tx.balance_after,
        cfa_concepts=tx.cfa_concepts,
    )


@app.post("/api/paper/order", response_model=OrderResponse)
async def paper_submit_order(req: OrderSubmitRequest, account_id: str = "default"):
    """Submit a new order (market/limit/stop/stop-limit). Checks pending orders
    against current prices first so newly ripe ones fill before the new submission."""
    acct = _get_or_create_account(account_id)
    _check_pending_against_live(acct)
    order = acct.submit_order(
        ticker=req.ticker.upper(),
        side=OrderSide(req.side),
        quantity=req.quantity,
        order_type=OrderType(req.order_type),
        tif=TIF(req.tif),
        limit_price=req.limit_price,
        stop_price=req.stop_price,
    )
    return OrderResponse(**asdict(order))


@app.get("/api/paper/orders", response_model=OrderListResponse)
async def paper_orders(account_id: str = "default"):
    """Return all orders (all statuses), newest first."""
    acct = _get_or_create_account(account_id)
    _check_pending_against_live(acct)
    return OrderListResponse(orders=[OrderResponse(**o) for o in acct.get_orders()])


@app.post("/api/paper/order/cancel", response_model=OrderResponse)
async def paper_cancel_order(req: CancelOrderRequest, account_id: str = "default"):
    """Cancel a pending order by order_id."""
    acct = _get_or_create_account(account_id)
    order = acct.cancel_order(req.order_id)
    if order is None:
        raise HTTPException(status_code=404, detail=f"Order {req.order_id} not found")
    return OrderResponse(**asdict(order))


@app.post("/api/paper/check-orders", response_model=CheckOrdersResponse)
async def paper_check_orders(account_id: str = "default"):
    """Manually fetch current prices for all pending orders and check for fills.
    Used by the 'Check now' button in the UI."""
    acct = _get_or_create_account(account_id)
    pending = acct.get_pending_orders()
    tickers = sorted({o["ticker"] for o in pending}) if pending else []
    prices = get_batch_prices(tickers) if tickers else {}
    changed = acct.check_pending_orders(prices)
    return CheckOrdersResponse(changed=[OrderResponse(**asdict(o)) for o in changed])


@app.get("/api/paper/positions", response_model=list[PositionResponse])
async def paper_positions(account_id: str = "default"):
    """Get all current positions with live prices and P&L."""
    acct = _get_or_create_account(account_id)
    _check_pending_against_live(acct)
    positions = acct.get_positions()
    return [
        PositionResponse(
            ticker=p.ticker, shares=p.shares, avg_cost=p.avg_cost,
            current_price=p.current_price, market_value=p.market_value,
            unrealized_pnl=p.unrealized_pnl, unrealized_pnl_pct=p.unrealized_pnl_pct,
            cost_basis=p.cost_basis, weight=p.weight,
            first_bought=p.first_bought, last_traded=p.last_traded,
            total_realized=p.total_realized,
        )
        for p in positions
    ]


@app.get("/api/paper/performance", response_model=PerformanceResponse)
async def paper_performance(account_id: str = "default"):
    """
    Full performance analytics for the paper trading account.
    Includes TWR, MWR, Sharpe, Sortino, max drawdown, trade stats.
    All metrics mapped to CFA curriculum readings.
    """
    acct = _get_or_create_account(account_id)
    perf = acct.get_performance()
    return PerformanceResponse(**perf.__dict__)


@app.get("/api/paper/summary", response_model=AccountSummaryResponse)
async def paper_summary(account_id: str = "default"):
    """Quick account overview: NAV, cash, positions, total return."""
    acct = _get_or_create_account(account_id)
    _check_pending_against_live(acct)
    summary = acct.get_summary()
    return AccountSummaryResponse(**summary)


@app.get("/api/paper/transactions")
async def paper_transactions(account_id: str = "default", limit: int = 50):
    """Get recent transaction history, newest first."""
    acct = _get_or_create_account(account_id)
    return acct.get_transactions(limit)


@app.get("/api/paper/nav-history")
async def paper_nav_history(account_id: str = "default"):
    """Get NAV history for charting portfolio value over time."""
    acct = _get_or_create_account(account_id)
    return acct.get_nav_history()


@app.post("/api/paper/reset")
async def paper_reset(account_id: str = "default"):
    """Reset the paper trading account to starting cash."""
    _accounts[account_id] = PaperAccount(account_id=account_id)
    return {"status": "reset", "account_id": account_id, "cash": PAPER_STARTING_CASH}


@app.get("/api/paper/export")
async def paper_export(account_id: str = "default"):
    """Export full account state for backup/persistence."""
    acct = _get_or_create_account(account_id)
    return acct.to_dict()


@app.post("/api/paper/import")
async def paper_import(data: dict):
    """Import account state from backup."""
    try:
        acct = PaperAccount.from_dict(data)
        _accounts[acct.account_id] = acct
        return {"status": "imported", "account_id": acct.account_id}
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid account data: {e}")


# ── CFA Curriculum Tags ─────────────────────────────────────────────────────

@app.get("/api/cfa/tags")
async def cfa_tags():
    """Get all CFA curriculum tags used in the platform."""
    return get_all_cfa_tags()


@app.get("/api/cfa/tag/{concept}", response_model=CfaTagResponse)
async def cfa_tag(concept: str):
    """
    Get CFA curriculum tag for a specific concept.
    Returns the CFA level, reading, formula, and explanation.
    """
    tag = get_cfa_tag(concept)
    if not tag:
        raise HTTPException(status_code=404, detail=f"No CFA tag for concept: {concept}")
    return CfaTagResponse(concept=concept, **tag)


from app.config import PAPER_STARTING_CASH

# ── Derivatives Endpoints ────────────────────────────────────────────────────

from app.derivatives import (
    price_option as _price_opt,
    implied_volatility as _impl_vol,
    build_iv_surface,
    compute_strategy_payoff,
    OptionType,
    StrategyType,
)


@app.post("/api/derivatives/price", response_model=OptionPriceResponse)
async def price_option_endpoint(req: PriceOptionRequest):
    """
    Price a European option using Black-Scholes-Merton with full Greeks.

    CFA Level II: Valuation of Contingent Claims
    Returns: price, delta, gamma, theta, vega, rho, probability ITM/OTM.
    """
    opt_type = OptionType.CALL if req.option_type == "call" else OptionType.PUT
    result = _price_opt(
        req.spot, req.strike, req.tte, req.volatility,
        opt_type, req.rate, req.dividend_yield,
    )
    return OptionPriceResponse(**result.__dict__)


@app.post("/api/derivatives/iv")
async def implied_vol_endpoint(req: IVRequest):
    """
    Extract implied volatility from a market option price.

    CFA Level II: IV is the market's expectation of future realized volatility.
    """
    opt_type = OptionType.CALL if req.option_type == "call" else OptionType.PUT
    iv = _impl_vol(req.market_price, req.spot, req.strike, req.tte, opt_type)
    return {
        "implied_volatility": iv,
        "implied_volatility_pct": round(iv * 100, 2),
        "cfa_concept": "L2: Implied volatility is the σ that makes BSM price = market price",
    }


@app.get("/api/derivatives/iv-surface/{ticker}")
async def iv_surface_endpoint(ticker: str):
    """
    Extract the full implied volatility surface from live options chains.

    CFA Level II: The IV surface shows how the market prices risk across
    strikes (skew) and expirations (term structure).
    """
    try:
        surface = build_iv_surface(ticker.upper())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {
        "ticker": surface.ticker,
        "spot": surface.spot,
        "atm_iv": surface.atm_iv,
        "atm_iv_pct": round(surface.atm_iv * 100, 1),
        "iv_rank": surface.iv_rank,
        "iv_percentile": surface.iv_percentile,
        "skew_25d": surface.skew_25d,
        "points": [p.__dict__ for p in surface.points],
        "point_count": len(surface.points),
        "cfa_concepts": [
            "L2: Implied Volatility Surface",
            "L2: Volatility Smile/Skew",
            "L3: Volatility Trading Strategies",
        ],
    }


@app.post("/api/derivatives/strategy")
async def strategy_payoff_endpoint(req: StrategyRequest):
    """
    Compute payoff diagram and analytics for an options strategy.

    CFA Level III: Options Strategies
    Supports: covered call, protective put, collar, bull call spread,
    bear put spread, straddle, strangle.
    """
    spot = req.spot
    vol = req.volatility

    # Fetch from yfinance if ticker provided
    if req.ticker and not spot:
        price = None
        try:
            t = yf.Ticker(req.ticker.upper())
            info = t.info or {}
            price = info.get("regularMarketPrice") or info.get("currentPrice")
        except Exception:
            pass
        if not price:
            raise HTTPException(status_code=422, detail=f"Cannot get price for {req.ticker}")
        spot = price

    if not spot or spot <= 0:
        raise HTTPException(status_code=422, detail="Spot price required")

    try:
        strategy = StrategyType(req.strategy)
    except ValueError:
        valid = [s.value for s in StrategyType]
        raise HTTPException(status_code=422, detail=f"Invalid strategy. Valid: {valid}")

    result = compute_strategy_payoff(
        spot=spot, strategy_type=strategy, volatility=vol, tte=req.tte,
        call_strike=req.call_strike, put_strike=req.put_strike,
        upper_strike=req.upper_strike, lower_strike=req.lower_strike,
    )

    return {
        "strategy": result.strategy_type,
        "spot": spot,
        "legs": result.legs,
        "max_profit": result.max_profit,
        "max_loss": result.max_loss,
        "breakeven": result.breakeven,
        "prob_profit": result.prob_profit,
        "cost_to_enter": result.cost_to_enter,
        "greeks": {
            "delta": result.net_delta,
            "gamma": result.net_gamma,
            "theta": result.net_theta,
            "vega": result.net_vega,
        },
        "payoff_chart": {
            "prices": result.price_range,
            "payoff": result.payoff_curve,
        },
        "cfa_concepts": result.cfa_concepts,
    }


import yfinance as yf

# ── AI Proxy Endpoint ────────────────────────────────────────────────────────
# Proxies Claude calls via Replicate so users don't need their own API key.
# Set REPLICATE_API_TOKEN env var on server. If not set, returns a helpful error.

import asyncio as _asyncio
import os
import httpx as _httpx

REPLICATE_MODEL = "anthropic/claude-4.5-sonnet"

@app.post("/api/ai/chat")
async def ai_chat(body: dict):
    """
    Proxy chat to Claude via Replicate using server-side token.
    Frontend sends {system, message} — backend adds the Replicate token.
    """
    token = os.environ.get("REPLICATE_API_TOKEN", "")
    if not token:
        raise HTTPException(status_code=503, detail="AI features require a REPLICATE_API_TOKEN environment variable on the server. Set it in your Render dashboard under Environment.")

    system = body.get("system", "You are an educational explainer for Quantex's analytics. You explain what displayed metrics mean and how they are computed. You never recommend buying, selling, holding, or allocating to any security or strategy, never answer 'what should I do' questions with a course of action, and redirect such questions to how the relevant concept works. You are not a financial adviser and must say so if asked.")
    message = body.get("message", "")
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    url = f"https://api.replicate.com/v1/models/{REPLICATE_MODEL}/predictions"
    auth_headers = {"Authorization": f"Bearer {token}"}
    payload = {"input": {"prompt": message, "system_prompt": system, "max_tokens": 1024}}

    def _extract_output(pred: dict) -> str:
        out = pred.get("output")
        if isinstance(out, list):
            return "".join(out)
        return "" if out is None else str(out)

    try:
        async with _httpx.AsyncClient(timeout=75.0) as client:
            resp = await client.post(
                url,
                headers={**auth_headers, "Content-Type": "application/json", "Prefer": "wait"},
                json=payload,
            )
            data = resp.json()
            if resp.status_code >= 400:
                logger.error("Replicate API error (status=%s): %s", resp.status_code, data)
                raise HTTPException(status_code=502, detail=data.get("detail") or "Replicate API error")

            status = data.get("status")
            if status == "succeeded":
                return {"response": _extract_output(data)}
            if status == "failed":
                logger.error("Replicate prediction failed: %s", data)
                raise HTTPException(status_code=502, detail=data.get("error") or "Replicate prediction failed")

            # Prefer: wait window exceeded — poll until terminal state.
            poll_url = (data.get("urls") or {}).get("get")
            if not poll_url:
                logger.error("Replicate returned status=%s without poll URL: %s", status, data)
                raise HTTPException(status_code=502, detail="Replicate response missing poll URL")
            for _ in range(30):
                await _asyncio.sleep(2)
                pr = await client.get(poll_url, headers=auth_headers)
                pd = pr.json()
                if pd.get("status") == "succeeded":
                    return {"response": _extract_output(pd)}
                if pd.get("status") == "failed":
                    logger.error("Replicate prediction failed on poll: %s", pd)
                    raise HTTPException(status_code=502, detail=pd.get("error") or "Replicate prediction failed")
            raise HTTPException(status_code=504, detail="Replicate prediction timed out")
    except _httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Replicate API timed out")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"AI proxy error: {exc}")


# ── Serve Frontend ───────────────────────────────────────────────────────────
# Serves quantex.html at root — same domain, no CORS, one deploy.
# Place quantex.html in the backend/static/ directory.

STATIC_DIR = Path(__file__).parent.parent / "static"


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the Quantex frontend."""
    html_path = STATIC_DIR / "quantex.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(), status_code=200)
    return HTMLResponse(
        content="<h1>Quantex</h1><p>Place quantex.html in backend/static/ to serve the frontend.</p>",
        status_code=200,
    )
