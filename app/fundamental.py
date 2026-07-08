"""
Fundamental Analysis Chain
Claude-powered pipeline that takes screened candidates and produces:
  1. Financial statement analysis (revenue, margins, FCF, ROIC, debt)
  2. Qualitative assessment (SWOT, competitive position)
  3. Sector-level context (PESTEL, Porter's Five Forces)
  4. Factual synthesis — source-tied description, no rating or recommendation

Each step produces structured JSON that feeds the next.
Sector analyses are cached quarterly. Company analyses cached 7 days.
"""
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import httpx
import yfinance as yf

from app.config import (
    CLAUDE_MODEL,
    CLAUDE_MAX_TOKENS,
)
from app.cache import cache_key, cache_get, cache_set

logger = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class FinancialProfile:
    """Structured financial data extracted from yfinance."""
    ticker: str
    name: str
    sector: str
    industry: str
    market_cap: float
    price: float

    # Income statement
    revenue: float | None = None
    revenue_growth: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None
    ebitda: float | None = None

    # Balance sheet
    total_debt: float | None = None
    total_cash: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None

    # Cash flow
    free_cash_flow: float | None = None
    operating_cash_flow: float | None = None
    capex: float | None = None

    # Valuation
    pe_ratio: float | None = None
    forward_pe: float | None = None
    peg_ratio: float | None = None
    price_to_book: float | None = None
    price_to_sales: float | None = None
    ev_to_ebitda: float | None = None

    # Returns
    roe: float | None = None
    roa: float | None = None
    roic: float | None = None

    # Forward estimates
    earnings_growth_est: float | None = None
    revenue_growth_est: float | None = None
    target_price: float | None = None
    analyst_recommendation: str | None = None
    num_analysts: int | None = None

    # Description
    business_summary: str = ""


@dataclass
class CompanyAnalysis:
    """Full analysis output for a single company."""
    ticker: str
    name: str
    sector: str

    # Financial analysis (Step 1)
    financial_summary: str = ""
    financial_strengths: list[str] = field(default_factory=list)
    financial_weaknesses: list[str] = field(default_factory=list)

    # Qualitative analysis (Step 2)
    swot: dict = field(default_factory=dict)  # {strengths, weaknesses, opportunities, threats}
    competitive_position: str = ""

    # Synthesis (Step 3)
    synthesis: str = ""
    key_catalysts: list[str] = field(default_factory=list)
    key_risks: list[str] = field(default_factory=list)
    price_target_rationale: str = ""

    # Metadata
    analysis_source: str = "claude"  # 'claude' or 'fallback'


@dataclass
class SectorAnalysis:
    """Sector-level PESTEL and Porter's Five Forces."""
    sector: str
    pestel: dict = field(default_factory=dict)
    porters: dict = field(default_factory=dict)
    sector_outlook: str = ""
    key_trends: list[str] = field(default_factory=list)


# ── Step 0: Fetch Financial Data ─────────────────────────────────────────────

def fetch_financial_profile(ticker: str) -> FinancialProfile:
    """
    Fetch comprehensive financial data from yfinance for a single ticker.
    Returns a structured FinancialProfile.
    """
    t = yf.Ticker(ticker)
    info = t.info or {}

    profile = FinancialProfile(
        ticker=ticker,
        name=info.get("shortName", info.get("longName", ticker)),
        sector=info.get("sector", "Unknown"),
        industry=info.get("industry", "Unknown"),
        market_cap=info.get("marketCap", 0) or 0,
        price=info.get("regularMarketPrice") or info.get("currentPrice") or 0,

        revenue=info.get("totalRevenue"),
        revenue_growth=info.get("revenueGrowth"),
        gross_margin=info.get("grossMargins"),
        operating_margin=info.get("operatingMargins"),
        net_margin=info.get("profitMargins"),
        ebitda=info.get("ebitda"),

        total_debt=info.get("totalDebt"),
        total_cash=info.get("totalCash"),
        debt_to_equity=info.get("debtToEquity"),
        current_ratio=info.get("currentRatio"),

        free_cash_flow=info.get("freeCashflow"),
        operating_cash_flow=info.get("operatingCashflow"),

        pe_ratio=info.get("trailingPE"),
        forward_pe=info.get("forwardPE"),
        peg_ratio=info.get("pegRatio"),
        price_to_book=info.get("priceToBook"),
        price_to_sales=info.get("priceToSalesTrailing12Months"),
        ev_to_ebitda=info.get("enterpriseToEbitda"),

        roe=info.get("returnOnEquity"),
        roa=info.get("returnOnAssets"),

        earnings_growth_est=info.get("earningsGrowth"),
        revenue_growth_est=info.get("revenueGrowth"),
        target_price=info.get("targetMeanPrice"),
        analyst_recommendation=info.get("recommendationKey"),
        num_analysts=info.get("numberOfAnalystOpinions"),

        business_summary=(info.get("longBusinessSummary") or "")[:500],
    )

    # Compute ROIC if we have the data
    if info.get("operatingCashflow") and info.get("totalDebt") and info.get("totalStockholderEquity"):
        invested = (info.get("totalDebt", 0) or 0) + (info.get("totalStockholderEquity", 0) or 0)
        if invested > 0:
            profile.roic = (info.get("operatingCashflow", 0) or 0) / invested

    # Compute capex from cash flow
    if info.get("operatingCashflow") and info.get("freeCashflow"):
        profile.capex = (info.get("operatingCashflow", 0) or 0) - (info.get("freeCashflow", 0) or 0)

    logger.info(f"Fetched financial profile for {ticker}: mcap={profile.market_cap:,.0f}")
    return profile


def _format_profile_for_prompt(p: FinancialProfile) -> str:
    """Format a FinancialProfile into a readable string for Claude."""
    def _fmt(v, pct=False, dollar=False):
        if v is None: return "N/A"
        if dollar: return f"${v:,.0f}" if v >= 1 else f"${v:,.2f}"
        if pct: return f"{v * 100:.1f}%" if abs(v) < 10 else f"{v:.1f}%"
        return f"{v:,.2f}" if isinstance(v, float) else str(v)

    return f"""COMPANY: {p.name} ({p.ticker})
Sector: {p.sector} | Industry: {p.industry}
Market Cap: ${p.market_cap:,.0f} | Price: ${p.price:.2f}

INCOME STATEMENT:
  Revenue: {_fmt(p.revenue, dollar=True)} | Revenue Growth: {_fmt(p.revenue_growth, pct=True)}
  Gross Margin: {_fmt(p.gross_margin, pct=True)} | Operating Margin: {_fmt(p.operating_margin, pct=True)}
  Net Margin: {_fmt(p.net_margin, pct=True)} | EBITDA: {_fmt(p.ebitda, dollar=True)}

BALANCE SHEET:
  Total Debt: {_fmt(p.total_debt, dollar=True)} | Total Cash: {_fmt(p.total_cash, dollar=True)}
  Debt/Equity: {_fmt(p.debt_to_equity)} | Current Ratio: {_fmt(p.current_ratio)}

CASH FLOW:
  Operating CF: {_fmt(p.operating_cash_flow, dollar=True)} | Free CF: {_fmt(p.free_cash_flow, dollar=True)}
  CapEx: {_fmt(p.capex, dollar=True)}

VALUATION:
  P/E: {_fmt(p.pe_ratio)} | Forward P/E: {_fmt(p.forward_pe)} | PEG: {_fmt(p.peg_ratio)}
  P/B: {_fmt(p.price_to_book)} | P/S: {_fmt(p.price_to_sales)} | EV/EBITDA: {_fmt(p.ev_to_ebitda)}

RETURNS:
  ROE: {_fmt(p.roe, pct=True)} | ROA: {_fmt(p.roa, pct=True)} | ROIC: {_fmt(p.roic, pct=True)}

FORWARD ESTIMATES:
  Earnings Growth Est: {_fmt(p.earnings_growth_est, pct=True)}
  Revenue Growth Est: {_fmt(p.revenue_growth_est, pct=True)}
  Analyst Target Price: {_fmt(p.target_price, dollar=True)} | Recommendation: {p.analyst_recommendation or "N/A"}
  Number of Analysts: {p.num_analysts or "N/A"}

BUSINESS: {p.business_summary}"""


# ── Claude API Helper ────────────────────────────────────────────────────────

async def _call_claude(
    system: str,
    user_message: str,
    api_key: str,
    max_tokens: int = CLAUDE_MAX_TOKENS,
) -> str:
    """Call Claude API and return the text response."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            ANTHROPIC_API_URL,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": CLAUDE_MODEL,
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": user_message}],
            },
        )
        data = resp.json()
        if "error" in data:
            raise ValueError(f"Claude API error: {data['error'].get('message', data['error'])}")
        return "".join(c.get("text", "") for c in data.get("content", []))


async def _call_claude_json(
    system: str,
    user_message: str,
    api_key: str,
    max_tokens: int = CLAUDE_MAX_TOKENS,
) -> dict:
    """Call Claude and parse the response as JSON."""
    text = await _call_claude(
        system + "\n\nRespond ONLY with valid JSON. No markdown, no backticks, no preamble.",
        user_message, api_key, max_tokens,
    )
    # Strip any markdown fencing
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    return json.loads(text)


# ── Step 1: Financial Analysis ───────────────────────────────────────────────

async def analyze_financials(profile: FinancialProfile, api_key: str) -> dict:
    """
    Step 1: Claude analyzes the financial statements.
    Returns structured assessment of financial health.
    """
    system = """You are a CFA-level equity analyst. Analyze the financial data provided.
Assess: revenue quality, margin trends, balance sheet health, cash flow generation, 
capital efficiency (ROIC), and valuation relative to growth.
Be specific — reference actual numbers."""

    prompt = f"""Analyze these financials and return JSON:
{{
  "summary": "2-3 sentence financial health summary",
  "strengths": ["strength 1", "strength 2", ...],
  "weaknesses": ["weakness 1", "weakness 2", ...],
  "revenue_quality": "high/medium/low with reasoning",
  "balance_sheet_health": "strong/adequate/weak with reasoning",
  "cash_flow_quality": "strong/adequate/weak with reasoning",
  "valuation_context": "factual description of the multiples relative to growth and, where given, to sector norms — state the numbers, do NOT judge whether the stock is cheap, undervalued, or overvalued"
}}

{_format_profile_for_prompt(profile)}"""

    return await _call_claude_json(system, prompt, api_key)


# ── Step 2: Qualitative Analysis (SWOT + Competitive Position) ───────────────

async def analyze_qualitative(
    profile: FinancialProfile,
    financial_analysis: dict,
    sector_context: dict | None,
    api_key: str,
) -> dict:
    """
    Step 2: Claude performs SWOT analysis and competitive assessment.
    Uses financial results + sector context for informed analysis.
    """
    sector_info = ""
    if sector_context:
        sector_info = f"\nSECTOR CONTEXT:\n{json.dumps(sector_context, indent=2)[:800]}"

    system = """You are a strategy analyst describing a company's competitive characteristics from the data.
Use the financial data, business description, and sector context to describe — factually — the competitive
position (segments, scale, customer/supplier dynamics, named competitors) and a SWOT of what the inputs
disclose. Do NOT rate, grade, or characterize the strength or width of any moat, and do NOT judge whether
the position is strong or weak — describe the characteristics, not a verdict on them."""

    prompt = f"""Analyze this company and return JSON:
{{
  "swot": {{
    "strengths": ["strength 1", ...],
    "weaknesses": ["weakness 1", ...],
    "opportunities": ["opportunity 1", ...],
    "threats": ["threat 1", ...]
  }},
  "competitive_position": "factual prose describing the competitive characteristics the data actually shows (segments, scale, customer/supplier dynamics, named competitors); describe, do NOT rate or grade the strength or width of any moat"
}}

COMPANY DATA:
{_format_profile_for_prompt(profile)}

FINANCIAL ANALYSIS:
{json.dumps(financial_analysis, indent=2)[:600]}
{sector_info}"""

    return await _call_claude_json(system, prompt, api_key)


# ── Step 3: Synthesis — Factual Description ──────────────────────────────────

async def synthesize_summary(
    profile: FinancialProfile,
    financial_analysis: dict,
    qualitative_analysis: dict,
    sector_context: dict | None,
    risk_score: int,
    goal: str,
    api_key: str,
) -> dict:
    """
    Step 3: Final factual synthesis combining all inputs into a source-tied
    description. Emits no rating, score, or recommendation.
    """
    system = f"""You are a CFA-level analyst producing a FACTUAL synthesis of a company from the data provided.
Describe the business, its segments, and its financial trends, and tie every statement to the specific
input it rests on — a named metric, the business description, or the sector context.

Hard constraints (non-negotiable):
- Do NOT recommend, rate, or characterize the investment's attractiveness. No buy / sell / hold framing,
  no "undervalued" / "overvalued", no advice on whether or how much to own.
- Do NOT forecast a price or assert a target of your own. The analyst target and analyst rating below are
  third-party facts you may cite AS third-party facts, and nothing more.
- If a piece of requested information is not present in the inputs, state that it is unavailable rather than
  inferring it or filling the gap from general knowledge.
Report what the data shows, not what to do about it."""

    sector_info = ""
    if sector_context:
        sector_info = f"\nSECTOR CONTEXT:\n{json.dumps(sector_context, indent=2)[:600]}"

    prompt = f"""Synthesize the inputs into FACTUAL JSON. Every string must describe what the data shows,
tied to its source, with no recommendation and no attractiveness judgment:
{{
  "synthesis": "3-4 sentence factual synthesis of the business, its segments, and its financial trends; tie each claim to its data source",
  "key_catalysts": ["factual, already-disclosed developments in the inputs — not predictions"],
  "key_risks": ["factual risk factors disclosed in the inputs"],
  "price_target_rationale": "state the third-party analyst target as a labeled fact plus factual valuation context; assert NO target of your own; write 'Not available' if no analyst target is present",
  "data_gaps": ["any requested information not present in the inputs"]
}}

COMPANY: {profile.name} ({profile.ticker})
Price: ${profile.price:.2f} | Market Cap: ${profile.market_cap:,.0f}
Third-party analyst target (fact): ${profile.target_price or 0:.2f} | Third-party analyst rating (fact): {profile.analyst_recommendation or "N/A"}

FINANCIAL ANALYSIS:
{json.dumps(financial_analysis, indent=2)[:500]}

COMPETITIVE ANALYSIS:
{json.dumps(qualitative_analysis, indent=2)[:500]}
{sector_info}"""

    return await _call_claude_json(system, prompt, api_key)


# ── Sector Analysis (PESTEL + Porter's) ─────────────────────────────────────

async def analyze_sector(sector: str, api_key: str) -> SectorAnalysis:
    """
    Sector-level analysis: PESTEL + Porter's Five Forces.
    Cached quarterly — run once per sector, shared across all companies in it.
    """
    ck = cache_key("sector", [sector])
    cached = cache_get(ck)
    if cached:
        return SectorAnalysis(**cached)

    system = """You are a macro strategist analyzing industry sectors.
Provide PESTEL analysis and Porter's Five Forces for the given sector."""

    prompt = f"""Analyze the {sector} sector and return JSON:
{{
  "pestel": {{
    "political": "key political factors affecting this sector",
    "economic": "key economic factors",
    "social": "key social/demographic factors",
    "technological": "key technology trends",
    "environmental": "key environmental/ESG factors",
    "legal": "key regulatory/legal factors"
  }},
  "porters": {{
    "rivalry": "competitive rivalry intensity (high/medium/low) with reasoning",
    "new_entrants": "threat of new entrants (high/medium/low) with reasoning",
    "substitutes": "threat of substitutes (high/medium/low) with reasoning",
    "buyer_power": "buyer bargaining power (high/medium/low) with reasoning",
    "supplier_power": "supplier bargaining power (high/medium/low) with reasoning"
  }},
  "sector_outlook": "2-3 sentence sector outlook for next 12-18 months",
  "key_trends": ["trend 1", "trend 2", "trend 3"]
}}"""

    try:
        data = await _call_claude_json(system, prompt, api_key)
        result = SectorAnalysis(
            sector=sector,
            pestel=data.get("pestel", {}),
            porters=data.get("porters", {}),
            sector_outlook=data.get("sector_outlook", ""),
            key_trends=data.get("key_trends", []),
        )
        # Cache for 90 days
        from app.config import SECTOR_ANALYSIS_REFRESH_DAYS
        cache_set(ck, {
            "sector": result.sector, "pestel": result.pestel,
            "porters": result.porters, "sector_outlook": result.sector_outlook,
            "key_trends": result.key_trends,
        }, SECTOR_ANALYSIS_REFRESH_DAYS * 24 * 3600)
        return result
    except Exception as e:
        logger.error(f"Sector analysis failed for {sector}: {e}")
        return SectorAnalysis(sector=sector)


# ── Full Pipeline ────────────────────────────────────────────────────────────

async def analyze_company(
    ticker: str,
    api_key: str,
    risk_score: int = 50,
    goal: str = "Balanced",
) -> CompanyAnalysis:
    """
    Run the full 3-step fundamental analysis for a single company.

    Step 0: Fetch financial data from yfinance
    Step 1: Claude analyzes financials → structured metrics
    Step 2: Claude performs SWOT + competitive assessment
    Step 3: Claude synthesizes → factual, source-tied description (no recommendation)
    """
    # Check cache
    ck = cache_key("fund", [ticker], risk=risk_score, goal=goal)
    cached = cache_get(ck)
    if cached:
        logger.info(f"Cache hit for {ticker} fundamental analysis")
        return CompanyAnalysis(**cached)

    logger.info(f"Running fundamental analysis for {ticker}...")

    # Step 0: Fetch data
    try:
        profile = fetch_financial_profile(ticker)
    except Exception as e:
        logger.error(f"Failed to fetch data for {ticker}: {e}")
        return CompanyAnalysis(
            ticker=ticker, name=ticker, sector="Unknown",
            synthesis=f"Unable to fetch financial data: {e}",
            analysis_source="error",
        )

    # Sector analysis (cached, shared)
    sector_context = None
    try:
        sector_result = await analyze_sector(profile.sector, api_key)
        sector_context = {
            "pestel": sector_result.pestel,
            "porters": sector_result.porters,
            "outlook": sector_result.sector_outlook,
            "trends": sector_result.key_trends,
        }
    except Exception as e:
        logger.warning(f"Sector analysis failed for {profile.sector}: {e}")

    # Step 1: Financial analysis
    try:
        fin = await analyze_financials(profile, api_key)
    except Exception as e:
        logger.error(f"Financial analysis failed for {ticker}: {e}")
        fin = {"summary": f"Analysis unavailable: {e}",
               "strengths": [], "weaknesses": []}

    # Step 2: Qualitative analysis
    try:
        qual = await analyze_qualitative(profile, fin, sector_context, api_key)
    except Exception as e:
        logger.error(f"Qualitative analysis failed for {ticker}: {e}")
        qual = {"swot": {}, "competitive_position": f"Analysis unavailable: {e}"}

    # Step 3: Synthesis
    try:
        synth = await synthesize_summary(
            profile, fin, qual, sector_context, risk_score, goal, api_key,
        )
    except Exception as e:
        logger.error(f"Synthesis failed for {ticker}: {e}")
        synth = {"synthesis": f"Synthesis unavailable: {e}",
                 "key_catalysts": [], "key_risks": []}

    result = CompanyAnalysis(
        ticker=ticker,
        name=profile.name,
        sector=profile.sector,
        financial_summary=fin.get("summary", ""),
        financial_strengths=fin.get("strengths", []),
        financial_weaknesses=fin.get("weaknesses", []),
        swot=qual.get("swot", {}),
        competitive_position=qual.get("competitive_position", ""),
        synthesis=synth.get("synthesis", ""),
        key_catalysts=synth.get("key_catalysts", []),
        key_risks=synth.get("key_risks", []),
        price_target_rationale=synth.get("price_target_rationale", ""),
        analysis_source="claude",
    )

    # Cache for 7 days
    from app.config import FUNDAMENTAL_CACHE_TTL
    cache_set(ck, result.__dict__, FUNDAMENTAL_CACHE_TTL)

    logger.info(f"Analysis complete for {ticker}")
    return result


async def analyze_batch(
    tickers: list[str],
    api_key: str,
    risk_score: int = 50,
    goal: str = "Balanced",
) -> list[CompanyAnalysis]:
    """
    Run fundamental analysis on a batch of tickers.
    Processes sequentially to respect API rate limits.
    Returns list of CompanyAnalysis sorted by ticker.
    """
    results = []
    for i, ticker in enumerate(tickers):
        logger.info(f"Analyzing {ticker} ({i + 1}/{len(tickers)})...")
        try:
            analysis = await analyze_company(ticker, api_key, risk_score, goal)
            results.append(analysis)
        except Exception as e:
            logger.error(f"Failed to analyze {ticker}: {e}")
            results.append(CompanyAnalysis(
                ticker=ticker, name=ticker, sector="Unknown",
                synthesis=f"Analysis failed: {e}",
                analysis_source="error",
            ))

    # Sort by ticker (no rating to rank by)
    results.sort(key=lambda x: x.ticker)
    return results
