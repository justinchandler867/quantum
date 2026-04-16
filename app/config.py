"""
Quantex Backend Configuration
"""
import os

# ── Data Settings ────────────────────────────────────────────────────────────
PRICE_HISTORY_YEARS = 5
NORMAL_WINDOW_DAYS = 252          # 1 trading year
STRESS_DRAWDOWN_THRESHOLD = -0.15  # -15% from 52-week high triggers stress regime
MIN_STRESS_DAYS = 60               # minimum trading days in stress windows for stable estimate
SHRINKAGE_ALWAYS_STRESS = True     # always apply Ledoit-Wolf to stress matrix
SHRINKAGE_NORMAL_THRESHOLD = 50    # apply shrinkage to normal matrix if tickers > this

# ── Blended Matrix ───────────────────────────────────────────────────────────
# λ = LAMBDA_BASE - (risk_score / 100) * LAMBDA_RANGE
# risk_score 0   → λ = 0.45  (very defensive, heavy stress weight)
# risk_score 100 → λ = 0.20  (growth-tolerant, lighter stress weight)
LAMBDA_BASE = 0.45
LAMBDA_RANGE = 0.25
DEFAULT_LAMBDA = 0.30

# ── Refresh Cadence ──────────────────────────────────────────────────────────
PRICE_REFRESH_HOUR = 16            # 4:30 PM ET (runs at :30)
PRICE_REFRESH_MINUTE = 30
STRESS_REFRESH_DAY = "sun"         # Sunday night
STRESS_REFRESH_HOUR = 22

# ── Diversification Thresholds ───────────────────────────────────────────────
HIGH_CORR_THRESHOLD = 0.85         # stress correlation above this = no diversification
HIGH_CORR_NORMAL_THRESHOLD = 0.75  # normal correlation flag threshold
NEGATIVE_CORR_THRESHOLD = -0.05    # below this = hedging benefit
PORTFOLIO_SIZE_SOFT_CAP = 18
PORTFOLIO_SIZE_HARD_FLOOR = 8

# ── Reference Hedging Instruments (non-Nasdaq) ──────────────────────────────
# Included in correlation matrix as reference rows, not investable
REFERENCE_HEDGES = ["TLT", "GLD", "SHY", "UUP", "VXX"]

# ── Redis ────────────────────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL_DAILY = 60 * 60 * 18     # 18 hours (covers market close → next refresh)
CACHE_TTL_WEEKLY = 60 * 60 * 24 * 7

# ── Database ─────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./quantex.db")

# ── Risk-Free Rate ───────────────────────────────────────────────────────────
RISK_FREE_RATE = 0.043             # 4.3% — update from US 10Y yield

# ── Screening Pipeline ──────────────────────────────────────────────────────
# Stage 1: Hard gates
SCREEN_MIN_MARKET_CAP = 2_000_000_000    # $2B minimum
SCREEN_MIN_AVG_VOLUME = 500_000          # 500K shares/day average
SCREEN_MIN_PRICE = 5.0                   # no penny stocks
SCREEN_MIN_HISTORY_DAYS = 252            # need at least 1 year of price data

# Stage 2: Factor scoring — weights per factor in composite score
FACTOR_WEIGHTS_DEFAULT = {
    "momentum": 0.25,
    "quality": 0.25,
    "value": 0.20,
    "low_vol": 0.15,
    "yield": 0.15,
}

# Stage 3: Profile goal → factor weight overrides
FACTOR_WEIGHTS_BY_GOAL = {
    "Growth": {"momentum": 0.35, "quality": 0.25, "value": 0.15, "low_vol": 0.10, "yield": 0.15},
    "Income": {"momentum": 0.10, "quality": 0.20, "value": 0.15, "low_vol": 0.20, "yield": 0.35},
    "Preservation": {"momentum": 0.10, "quality": 0.25, "value": 0.15, "low_vol": 0.35, "yield": 0.15},
    "Balanced": {"momentum": 0.20, "quality": 0.25, "value": 0.20, "low_vol": 0.20, "yield": 0.15},
}

# Shortlist sizes
SCREEN_STAGE2_SIZE = 120     # after factor scoring
SCREEN_FINAL_SIZE = 40       # final ranked output

# ── Paper Trading Engine ─────────────────────────────────────────────────────
PAPER_STARTING_CASH = 100_000.00
PAPER_COMMISSION = 0.00              # zero commission (realistic for modern brokers)
PAPER_MIN_TRADE = 1                  # minimum 1 share
PAPER_MAX_POSITIONS = 50
PAPER_CACHE_TTL = 60 * 5             # 5 min for live price lookups
CLAUDE_MODEL = "claude-sonnet-4-20250514"
CLAUDE_MAX_TOKENS = 1500
FUNDAMENTAL_CACHE_TTL = 60 * 60 * 24 * 7  # 7 days — fundamentals change slowly

# PESTEL & Porter's are sector-level, cached per GICS sector
SECTOR_ANALYSIS_REFRESH_DAYS = 90  # quarterly refresh

# Conviction score weights
CONVICTION_WEIGHTS = {
    "financials": 0.30,        # revenue growth, margins, FCF, ROIC, debt
    "competitive_position": 0.25,  # moat, market share, Porter's forces
    "growth_trajectory": 0.20,     # forward estimates, TAM, catalysts
    "risk_factors": 0.15,          # concentration, regulatory, macro
    "valuation": 0.10,             # DCF margin of safety
}

