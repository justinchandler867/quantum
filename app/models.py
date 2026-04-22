"""
API Models — Pydantic request/response schemas.
"""
from pydantic import BaseModel, Field


class TickerList(BaseModel):
    tickers: list[str] = Field(..., min_length=2, max_length=50, description="List of ticker symbols")


class CorrelationRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=2, max_length=50)
    regime: str = Field("blended", pattern="^(normal|stress|blended)$")
    window: int = Field(252, ge=60, le=1260)


class CovarianceRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=2, max_length=50)
    lam: float = Field(0.30, ge=0.0, le=1.0, description="Stress blending weight λ")
    risk_score: int | None = Field(None, ge=0, le=100, description="If provided, λ is computed from risk score")
    window: int = Field(252, ge=60, le=1260)


class DiagnosticsRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=2, max_length=50)
    weights: list[float] | None = Field(None, description="Portfolio weights (same order as tickers, sum to 1.0)")


class CorrelationEntry(BaseModel):
    ticker_a: str
    ticker_b: str
    correlation: float


class CorrelationResponse(BaseModel):
    matrix: list[list[float]]
    tickers: list[str]
    regime: str
    observation_days: int
    shrinkage_applied: bool
    shrinkage_coefficient: float


class CovarianceResponse(BaseModel):
    matrix: list[list[float]]
    tickers: list[str]
    lambda_used: float
    vol_normal: list[float]
    vol_stress: list[float]
    vol_blended: list[float]


class PairInfo(BaseModel):
    ticker_a: str
    ticker_b: str
    correlation: float


class ReferenceHedgeInfo(BaseModel):
    ticker: str
    avg_corr_normal: float
    avg_corr_stress: float | None = None


class DiagnosticsResponse(BaseModel):
    avg_corr_normal: float
    avg_corr_stress: float
    corr_spike_pct: float
    diversification_message: str
    high_corr_pairs_normal: list[PairInfo]
    high_corr_pairs_stress: list[PairInfo]
    hedging_pairs: list[PairInfo]
    reference_hedges: list[ReferenceHedgeInfo]
    warnings: list[str]


# ── Optimizer Models ─────────────────────────────────────────────────────────

class ProfileConstraintsInput(BaseModel):
    max_position_pct: float = Field(0.30, ge=0.01, le=1.0, description="Max weight per asset (0-1)")
    min_position_pct: float = Field(0.02, ge=0.0, le=0.5, description="Min weight if included (0-1)")
    max_beta: float | None = Field(None, ge=0.0, le=5.0, description="Portfolio-level beta cap")
    max_volatility: float | None = Field(None, ge=0.0, le=2.0, description="Portfolio-level annualized vol cap")
    excluded_tickers: list[str] = Field(default_factory=list, description="Tickers to exclude (force weight=0)")


class OptimizeRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=2, max_length=50)
    objective: str = Field("max_sharpe", pattern="^(max_sharpe|max_return|min_vol|risk_parity|max_diversification)$")
    risk_score: int | None = Field(None, ge=0, le=100, description="Investor risk score → auto-computes λ and constraints")
    constraints: ProfileConstraintsInput | None = None
    window: int = Field(252, ge=60, le=1260)
    include_current_weights: dict[str, float] | None = Field(
        None, description="Current weights for comparison (ticker → pct 0-100)"
    )


class OptimizeResponse(BaseModel):
    weights: dict[str, float]          # ticker → weight (0-1)
    objective: str
    portfolio_return: float
    portfolio_volatility: float
    sharpe_ratio: float
    beta: float
    var_95: float
    cvar_95: float
    risk_contributions: dict[str, float]
    solver_success: bool
    solver_message: str
    iterations: int
    constraints_active: list[str]
    comparison: dict | None = None     # before/after if current weights provided


class MultiOptimizeRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=2, max_length=50)
    risk_score: int | None = Field(None, ge=0, le=100)
    constraints: ProfileConstraintsInput | None = None
    window: int = Field(252, ge=60, le=1260)


class MultiOptimizeResponse(BaseModel):
    results: dict[str, OptimizeResponse]  # objective → result


class FrontierRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=2, max_length=50)
    risk_score: int | None = Field(None, ge=0, le=100)
    constraints: ProfileConstraintsInput | None = None
    n_points: int = Field(50, ge=10, le=200)
    window: int = Field(252, ge=60, le=1260)
    current_weights: dict[str, float] | None = Field(
        None, description="Current portfolio weights for plotting (ticker → pct 0-100)"
    )


class FrontierPoint(BaseModel):
    ret: float = Field(..., alias="return")
    volatility: float
    sharpe: float

    class Config:
        populate_by_name = True


class FrontierResponse(BaseModel):
    frontier: list[dict]               # [{return, volatility, sharpe}, ...]
    max_sharpe: dict                   # the optimal point
    min_vol: dict                      # the min-vol point
    current_portfolio: dict | None     # user's current position on the map


# ── Screener Models ──────────────────────────────────────────────────────────

class ScreenRequest(BaseModel):
    goal: str = Field("Balanced", pattern="^(Growth|Income|Preservation|Balanced)$")
    risk_score: int = Field(50, ge=0, le=100)
    time_horizon_years: float = Field(7.0, ge=0.5, le=30.0)
    max_results: int = Field(40, ge=5, le=100)
    tickers: list[str] | None = Field(
        None, description="Custom ticker list — skips universe loading if provided"
    )


class ScreenedAssetResponse(BaseModel):
    ticker: str
    name: str
    sector: str
    industry: str
    market_cap: float
    price: float
    avg_volume: float
    return_1y: float
    return_6m: float
    return_3m: float
    volatility: float
    sharpe: float
    beta: float
    max_drawdown: float
    dividend_yield: float
    pe_ratio: float | None
    forward_pe: float | None
    pb_ratio: float | None
    earnings_yield: float | None
    earnings_date: str | None = None
    dividend_growth_5y: float | None = None
    debt_to_equity: float | None = None
    revenue_growth: float | None = None
    z_momentum: float
    z_quality: float
    z_value: float
    z_low_vol: float
    z_yield: float
    factor_composite: float
    fit_score: int
    rank: int


class ScreenResponse(BaseModel):
    shortlist: list[ScreenedAssetResponse]
    universe_size: int
    passed_gates: int
    stage2_size: int
    final_size: int
    goal_used: str
    factor_weights: dict[str, float]
    sector_distribution: dict[str, int]


# ── Fundamental Analysis Models ──────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=1, max_length=20)
    api_key: str = Field(..., min_length=10, description="Anthropic API key")
    risk_score: int = Field(50, ge=0, le=100)
    goal: str = Field("Balanced", pattern="^(Growth|Income|Preservation|Balanced)$")


class AnalyzeSingleRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    api_key: str = Field(..., min_length=10)
    risk_score: int = Field(50, ge=0, le=100)
    goal: str = Field("Balanced", pattern="^(Growth|Income|Preservation|Balanced)$")


class SWOTResponse(BaseModel):
    strengths: list[str] = []
    weaknesses: list[str] = []
    opportunities: list[str] = []
    threats: list[str] = []


class CompanyAnalysisResponse(BaseModel):
    ticker: str
    name: str
    sector: str
    financial_summary: str
    financial_strengths: list[str]
    financial_weaknesses: list[str]
    swot: SWOTResponse | dict
    competitive_position: str
    moat_assessment: str
    conviction_score: int
    conviction_reasoning: str
    key_catalysts: list[str]
    key_risks: list[str]
    recommendation: str
    price_target_rationale: str
    analysis_source: str


class BatchAnalysisResponse(BaseModel):
    analyses: list[CompanyAnalysisResponse]
    total_requested: int
    total_completed: int
    avg_conviction: float


class SectorAnalysisResponse(BaseModel):
    sector: str
    pestel: dict
    porters: dict
    sector_outlook: str
    key_trends: list[str]


# ── Paper Trading Models ─────────────────────────────────────────────────────

class TradeRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    side: str = Field(..., pattern="^(buy|sell)$")
    quantity: int = Field(..., ge=1, le=100000)
    price: float | None = Field(None, description="If null, uses live market price")


class TradeResponse(BaseModel):
    id: str
    timestamp: str
    ticker: str
    side: str
    quantity: int
    price: float
    total: float
    commission: float
    status: str
    reason: str = ""
    balance_after: float
    cfa_concepts: list[str] = []


class PositionResponse(BaseModel):
    ticker: str
    shares: int
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    cost_basis: float
    weight: float
    first_bought: str
    last_traded: str
    total_realized: float


class PerformanceResponse(BaseModel):
    total_return: float
    total_return_pct: float
    annualized_return: float
    daily_returns_std: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: int
    current_drawdown: float
    total_realized_pnl: float
    total_unrealized_pnl: float
    total_commissions: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    twr: float
    mwr: float
    num_positions: int
    cash_pct: float
    invested_pct: float
    days_active: int


class AccountSummaryResponse(BaseModel):
    account_id: str
    nav: float
    cash: float
    positions_count: int
    total_return_pct: float
    total_trades: int
    created_at: str


class CfaTagResponse(BaseModel):
    concept: str
    level: str
    reading: str
    topic: str
    formula: str
    explanation: str


# ── Derivatives Models ───────────────────────────────────────────────────────

class PriceOptionRequest(BaseModel):
    spot: float = Field(..., gt=0)
    strike: float = Field(..., gt=0)
    tte: float = Field(..., gt=0, le=5, description="Time to expiration in years")
    volatility: float = Field(..., gt=0, le=5, description="Annualized vol, e.g. 0.30")
    option_type: str = Field("call", pattern="^(call|put)$")
    rate: float = Field(0.043, ge=0)
    dividend_yield: float = Field(0.0, ge=0)


class OptionPriceResponse(BaseModel):
    price: float
    intrinsic: float
    time_value: float
    option_type: str
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    prob_itm: float
    prob_otm: float
    spot: float
    strike: float
    tte: float
    volatility: float
    rate: float
    cfa_concepts: list[str]


class IVRequest(BaseModel):
    market_price: float = Field(..., gt=0)
    spot: float = Field(..., gt=0)
    strike: float = Field(..., gt=0)
    tte: float = Field(..., gt=0, le=5)
    option_type: str = Field("call", pattern="^(call|put)$")


class StrategyRequest(BaseModel):
    ticker: str | None = Field(None, description="If provided, fetches spot + vol from yfinance")
    spot: float | None = Field(None, gt=0)
    strategy: str = Field(..., description="Strategy type")
    volatility: float = Field(0.30, gt=0)
    tte: float = Field(0.25, gt=0)
    call_strike: float | None = None
    put_strike: float | None = None
    upper_strike: float | None = None
    lower_strike: float | None = None


# ── Prices Models ────────────────────────────────────────────────────────────

class PricesRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=1, max_length=50)
    range: str = Field("1Y", pattern="^(1W|1M|1Q|1Y|3Y|5Y|common)$")
    include_volume: bool = False


class PricesResponse(BaseModel):
    dates: list[str]
    prices: dict[str, list[float]]
    start_date: str
    end_date: str
    skipped: list[str] = Field(default_factory=list, description="Requested tickers with no price data")
    volume: dict[str, list[float]] | None = None


# ── Search / Ticker Add Models ───────────────────────────────────────────────

class SearchRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=50)


class SearchResult(BaseModel):
    symbol: str
    name: str
    exchange: str
    in_universe: bool


class TickerAddRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10)


class OrderSubmitRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    side: str = Field(..., pattern="^(buy|sell)$")
    quantity: int = Field(..., ge=1)
    order_type: str = Field("market", pattern="^(market|limit|stop|stop_limit)$")
    tif: str = Field("day", pattern="^(day|gtc)$")
    limit_price: float | None = Field(None, gt=0)
    stop_price: float | None = Field(None, gt=0)


class OrderResponse(BaseModel):
    order_id: str
    ticker: str
    side: str
    quantity: int
    order_type: str
    tif: str
    status: str
    limit_price: float | None = None
    stop_price: float | None = None
    triggered_at: str | None = None
    submitted_at: str
    filled_at: str | None = None
    cancelled_at: str | None = None
    expired_at: str | None = None
    fill_price: float | None = None
    reject_reason: str | None = None


class OrderListResponse(BaseModel):
    orders: list[OrderResponse] = Field(default_factory=list)


class CancelOrderRequest(BaseModel):
    order_id: str = Field(..., min_length=1)


class CheckOrdersResponse(BaseModel):
    changed: list[OrderResponse] = Field(default_factory=list)


class NewsRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10)


class NewsItem(BaseModel):
    title: str
    link: str
    publisher: str
    published: str | None = None
    summary: str | None = None


class NewsResponse(BaseModel):
    items: list[NewsItem] = Field(default_factory=list)


class TickerAddResponse(BaseModel):
    id: str
    n: str
    t: str = "Stock"
    sec: str = "Unknown"
    ret: float = 0.0
    vol: float = 20.0
    beta: float = 1.0
    sh: float = 0.0
    yld: float = 0.0
    sig: str = "Hold"
    rsi: float | None = None
    macd: str | None = None
    tgt: float | None = None
    stop: float | None = None
    goals: list[str] = Field(default_factory=list)
    rb: list[int] = Field(default_factory=lambda: [1, 5])
    hm: int = 3
    s08: float = 0.0
    s20: float = 0.0
    s22: float = 0.0
    fit: int = 50
    pe: float | None = None
    fpe: float | None = None
    mktCap: float | None = None
    maxDD: float = 0.0
