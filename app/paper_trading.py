"""
Paper Trading Engine
Virtual portfolio management with real market prices.

Core capabilities:
  - Execute paper trades (buy/sell) at real-time prices via yfinance
  - Track positions, cash balance, and NAV over time
  - Compute realized and unrealized P&L per position
  - Daily mark-to-market with NAV history
  - Performance analytics: TWR, MWR, Sharpe, max DD, attribution
  - Full transaction log with timestamps

All state is stored in-memory per account with JSON serialization
for persistence via cache or localStorage.
"""
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np
import yfinance as yf

from app.config import (
    PAPER_STARTING_CASH,
    PAPER_COMMISSION,
    PAPER_MAX_POSITIONS,
    RISK_FREE_RATE,
)

logger = logging.getLogger(__name__)


# ── Enums & Data Classes ─────────────────────────────────────────────────────

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    FILLED = "filled"
    REJECTED = "rejected"
    PENDING = "pending"


@dataclass
class Transaction:
    """A single executed trade."""
    id: str
    timestamp: str
    ticker: str
    side: str               # buy / sell
    quantity: int
    price: float            # execution price
    total: float            # quantity × price
    commission: float
    status: str             # filled / rejected
    reason: str = ""        # rejection reason if applicable
    balance_after: float = 0.0

    # CFA curriculum tags
    cfa_concepts: list[str] = field(default_factory=list)


@dataclass
class Position:
    """A current portfolio holding."""
    ticker: str
    shares: int
    avg_cost: float         # volume-weighted average cost basis
    current_price: float
    market_value: float     # shares × current_price
    unrealized_pnl: float   # market_value - (shares × avg_cost)
    unrealized_pnl_pct: float
    cost_basis: float       # shares × avg_cost
    weight: float = 0.0     # % of portfolio NAV

    # For analytics
    first_bought: str = ""
    last_traded: str = ""
    total_realized: float = 0.0  # realized P&L from partial sells


@dataclass
class NAVSnapshot:
    """Daily portfolio valuation snapshot."""
    date: str
    nav: float              # total portfolio value (cash + positions)
    cash: float
    positions_value: float
    daily_return: float = 0.0
    cumulative_return: float = 0.0


@dataclass
class PerformanceMetrics:
    """Portfolio performance analytics — maps to CFA Level I & II readings."""
    # Returns
    total_return: float          # (current NAV / starting NAV) - 1
    total_return_pct: float
    annualized_return: float
    daily_returns_std: float     # daily volatility
    annualized_volatility: float

    # Risk-adjusted (CFA L1 R52, L2 R43)
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Drawdown (CFA L1 R52)
    max_drawdown: float
    max_drawdown_duration_days: int
    current_drawdown: float

    # P&L
    total_realized_pnl: float
    total_unrealized_pnl: float
    total_commissions: float

    # Trade stats
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float     # gross wins / gross losses

    # Time-weighted vs money-weighted (CFA L1 R52)
    twr: float               # time-weighted return
    mwr: float               # money-weighted return (approx)

    # Portfolio characteristics
    num_positions: int
    cash_pct: float
    invested_pct: float
    days_active: int


# ── Price Fetching ───────────────────────────────────────────────────────────

def get_live_price(ticker: str) -> float | None:
    """Fetch the current market price for a ticker."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        price = info.get("regularMarketPrice") or info.get("currentPrice")
        if price and price > 0:
            return float(price)
        # Fallback to last close
        hist = t.history(period="1d")
        if hist is not None and not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as e:
        logger.warning(f"Price fetch failed for {ticker}: {e}")
    return None


def get_batch_prices(tickers: list[str]) -> dict[str, float]:
    """Fetch current prices for multiple tickers."""
    prices = {}
    for ticker in tickers:
        price = get_live_price(ticker)
        if price is not None:
            prices[ticker] = price
    return prices


# ── Paper Trading Account ────────────────────────────────────────────────────

class PaperAccount:
    """
    A virtual trading account with full position and P&L tracking.

    State is fully serializable to JSON for persistence.
    """

    def __init__(self, account_id: str = None, starting_cash: float = PAPER_STARTING_CASH):
        self.account_id = account_id or str(uuid.uuid4())[:8]
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.created_at = datetime.now(timezone.utc).isoformat()

        # Positions: ticker → {shares, avg_cost, first_bought, last_traded, total_realized}
        self._positions: dict[str, dict] = {}

        # Transaction log
        self._transactions: list[dict] = []

        # NAV history for performance computation
        self._nav_history: list[dict] = []
        self._record_nav()

    # ── Trade Execution ──────────────────────────────────────────────────────

    def execute_trade(
        self,
        ticker: str,
        side: OrderSide,
        quantity: int,
        price: float | None = None,
        order_type: OrderType = OrderType.MARKET,
    ) -> Transaction:
        """
        Execute a paper trade at real market price.

        If price is None, fetches live price from yfinance.
        Returns a Transaction with fill details or rejection reason.
        """
        now = datetime.now(timezone.utc).isoformat()
        tx_id = str(uuid.uuid4())[:12]

        # Fetch live price if not provided
        if price is None:
            price = get_live_price(ticker)
            if price is None:
                return self._reject(tx_id, now, ticker, side.value, quantity,
                                    "Unable to fetch market price")

        # Validate
        if quantity < 1:
            return self._reject(tx_id, now, ticker, side.value, quantity,
                                "Quantity must be at least 1")

        if side == OrderSide.BUY:
            return self._execute_buy(tx_id, now, ticker, quantity, price)
        else:
            return self._execute_sell(tx_id, now, ticker, quantity, price)

    def _execute_buy(self, tx_id, now, ticker, quantity, price) -> Transaction:
        """Execute a buy order."""
        total_cost = quantity * price + PAPER_COMMISSION

        # Check cash
        if total_cost > self.cash:
            max_affordable = int((self.cash - PAPER_COMMISSION) / price)
            return self._reject(tx_id, now, ticker, "buy", quantity,
                                f"Insufficient cash. Need ${total_cost:,.2f}, "
                                f"have ${self.cash:,.2f}. Max affordable: {max_affordable} shares")

        # Check position limit
        if ticker not in self._positions and len(self._positions) >= PAPER_MAX_POSITIONS:
            return self._reject(tx_id, now, ticker, "buy", quantity,
                                f"Maximum {PAPER_MAX_POSITIONS} positions reached")

        # Execute
        self.cash -= total_cost

        if ticker in self._positions:
            pos = self._positions[ticker]
            old_shares = pos["shares"]
            old_cost = pos["avg_cost"]
            new_shares = old_shares + quantity
            # Volume-weighted average cost
            pos["avg_cost"] = (old_shares * old_cost + quantity * price) / new_shares
            pos["shares"] = new_shares
            pos["last_traded"] = now
        else:
            self._positions[ticker] = {
                "shares": quantity,
                "avg_cost": price,
                "first_bought": now,
                "last_traded": now,
                "total_realized": 0.0,
            }

        tx = Transaction(
            id=tx_id, timestamp=now, ticker=ticker, side="buy",
            quantity=quantity, price=price, total=quantity * price,
            commission=PAPER_COMMISSION, status="filled",
            balance_after=self.cash,
            cfa_concepts=["L1: Equity Valuation", "L1: Portfolio Management"],
        )
        self._transactions.append(asdict(tx))
        self._record_nav()

        logger.info(f"BUY {quantity} {ticker} @ ${price:.2f} = ${quantity * price:,.2f}")
        return tx

    def _execute_sell(self, tx_id, now, ticker, quantity, price) -> Transaction:
        """Execute a sell order."""
        if ticker not in self._positions:
            return self._reject(tx_id, now, ticker, "sell", quantity,
                                f"No position in {ticker}")

        pos = self._positions[ticker]
        if quantity > pos["shares"]:
            return self._reject(tx_id, now, ticker, "sell", quantity,
                                f"Insufficient shares. Have {pos['shares']}, trying to sell {quantity}")

        # Compute realized P&L
        realized_pnl = (price - pos["avg_cost"]) * quantity
        proceeds = quantity * price - PAPER_COMMISSION
        self.cash += proceeds

        pos["shares"] -= quantity
        pos["total_realized"] += realized_pnl
        pos["last_traded"] = now

        # Remove position if fully closed
        if pos["shares"] == 0:
            del self._positions[ticker]

        tx = Transaction(
            id=tx_id, timestamp=now, ticker=ticker, side="sell",
            quantity=quantity, price=price, total=quantity * price,
            commission=PAPER_COMMISSION, status="filled",
            balance_after=self.cash,
            cfa_concepts=["L1: Equity Valuation", "L1: Portfolio Management",
                          "L2: Performance Attribution"],
        )
        self._transactions.append(asdict(tx))
        self._record_nav()

        logger.info(f"SELL {quantity} {ticker} @ ${price:.2f} = ${proceeds:,.2f} "
                     f"(realized P&L: ${realized_pnl:,.2f})")
        return tx

    def _reject(self, tx_id, now, ticker, side, quantity, reason) -> Transaction:
        """Create a rejected transaction."""
        tx = Transaction(
            id=tx_id, timestamp=now, ticker=ticker, side=side,
            quantity=quantity, price=0, total=0,
            commission=0, status="rejected", reason=reason,
        )
        self._transactions.append(asdict(tx))
        logger.warning(f"REJECTED: {side} {quantity} {ticker} — {reason}")
        return tx

    # ── Portfolio Valuation ──────────────────────────────────────────────────

    def get_positions(self) -> list[Position]:
        """Get all current positions with live prices and P&L."""
        if not self._positions:
            return []

        tickers = list(self._positions.keys())
        prices = get_batch_prices(tickers)
        nav = self._compute_nav(prices)

        positions = []
        for ticker, pos in self._positions.items():
            current_price = prices.get(ticker, pos["avg_cost"])
            market_value = pos["shares"] * current_price
            cost_basis = pos["shares"] * pos["avg_cost"]
            unrealized = market_value - cost_basis
            unrealized_pct = (unrealized / cost_basis * 100) if cost_basis > 0 else 0

            positions.append(Position(
                ticker=ticker,
                shares=pos["shares"],
                avg_cost=round(pos["avg_cost"], 2),
                current_price=round(current_price, 2),
                market_value=round(market_value, 2),
                unrealized_pnl=round(unrealized, 2),
                unrealized_pnl_pct=round(unrealized_pct, 2),
                cost_basis=round(cost_basis, 2),
                weight=round(market_value / nav * 100, 2) if nav > 0 else 0,
                first_bought=pos["first_bought"],
                last_traded=pos["last_traded"],
                total_realized=round(pos["total_realized"], 2),
            ))

        positions.sort(key=lambda p: p.market_value, reverse=True)
        return positions

    def get_nav(self) -> float:
        """Current net asset value (cash + positions at market)."""
        prices = get_batch_prices(list(self._positions.keys()))
        return self._compute_nav(prices)

    def _compute_nav(self, prices: dict[str, float] = None) -> float:
        """Compute NAV from cash + position values."""
        positions_value = 0
        for ticker, pos in self._positions.items():
            price = prices.get(ticker, pos["avg_cost"]) if prices else pos["avg_cost"]
            positions_value += pos["shares"] * price
        return self.cash + positions_value

    def _record_nav(self):
        """Record a NAV snapshot."""
        nav = self._compute_nav()
        prev_nav = self._nav_history[-1]["nav"] if self._nav_history else self.starting_cash
        daily_ret = (nav / prev_nav - 1) if prev_nav > 0 else 0
        cum_ret = (nav / self.starting_cash - 1) if self.starting_cash > 0 else 0

        positions_value = sum(
            pos["shares"] * pos["avg_cost"] for pos in self._positions.values()
        )

        self._nav_history.append({
            "date": datetime.now(timezone.utc).isoformat(),
            "nav": round(nav, 2),
            "cash": round(self.cash, 2),
            "positions_value": round(positions_value, 2),
            "daily_return": round(daily_ret, 6),
            "cumulative_return": round(cum_ret, 6),
        })

    # ── Performance Analytics ────────────────────────────────────────────────

    def get_performance(self) -> PerformanceMetrics:
        """
        Compute comprehensive performance metrics.
        Maps to CFA Level I Reading 52 and Level II Reading 43.
        """
        nav = self.get_nav()
        positions = self.get_positions()

        # Returns
        total_return = nav - self.starting_cash
        total_return_pct = (nav / self.starting_cash - 1) * 100

        # Daily returns from NAV history
        navs = [snap["nav"] for snap in self._nav_history]
        daily_rets = []
        for i in range(1, len(navs)):
            if navs[i - 1] > 0:
                daily_rets.append(navs[i] / navs[i - 1] - 1)

        daily_std = float(np.std(daily_rets)) if len(daily_rets) > 1 else 0
        ann_vol = daily_std * np.sqrt(252)

        # Days active
        days_active = max(len(self._nav_history), 1)
        ann_factor = 252 / max(days_active, 1)
        ann_return = ((nav / self.starting_cash) ** ann_factor - 1) if days_active > 1 else 0

        # Sharpe ratio (CFA L1 R52)
        rf_daily = RISK_FREE_RATE / 252
        excess_rets = [r - rf_daily for r in daily_rets]
        sharpe = (np.mean(excess_rets) / np.std(excess_rets) * np.sqrt(252)) if len(excess_rets) > 1 and np.std(excess_rets) > 0 else 0

        # Sortino ratio (CFA L2 R43) — uses downside deviation only
        downside_rets = [r for r in excess_rets if r < 0]
        downside_std = float(np.std(downside_rets)) if len(downside_rets) > 1 else daily_std
        sortino = (np.mean(excess_rets) / downside_std * np.sqrt(252)) if downside_std > 0 else 0

        # Max drawdown (CFA L1 R52)
        peak = navs[0] if navs else self.starting_cash
        max_dd = 0
        max_dd_duration = 0
        current_dd_start = 0
        current_dd = 0
        for i, n in enumerate(navs):
            if n > peak:
                peak = n
                current_dd_start = i
            dd = (n - peak) / peak if peak > 0 else 0
            if dd < max_dd:
                max_dd = dd
                max_dd_duration = i - current_dd_start
            current_dd = dd

        # Calmar ratio
        calmar = (ann_return / abs(max_dd)) if abs(max_dd) > 0.001 else 0

        # Trade statistics
        filled_trades = [t for t in self._transactions if t["status"] == "filled"]
        sell_trades = [t for t in filled_trades if t["side"] == "sell"]

        winning = []
        losing = []
        for t in sell_trades:
            ticker = t["ticker"]
            # Find the avg cost at time of sell (approximate from current positions or realized)
            pnl = 0
            for pos in positions:
                if pos.ticker == ticker:
                    pnl = (t["price"] - pos.avg_cost) * t["quantity"]
            if pnl >= 0:
                winning.append(pnl)
            else:
                losing.append(pnl)

        total_realized = sum(p.total_realized for p in positions) + sum(
            self._positions.get(t["ticker"], {}).get("total_realized", 0)
            for t in sell_trades
        )
        total_unrealized = sum(p.unrealized_pnl for p in positions)
        total_commissions = len(filled_trades) * PAPER_COMMISSION

        win_rate = len(winning) / max(len(sell_trades), 1)
        avg_win = np.mean(winning) if winning else 0
        avg_loss = np.mean(losing) if losing else 0
        gross_wins = sum(winning)
        gross_losses = abs(sum(losing))
        profit_factor = gross_wins / max(gross_losses, 0.01)

        # Time-weighted return (CFA L1 R52)
        # TWR = product of (1 + r_i) - 1
        twr = 1.0
        for r in daily_rets:
            twr *= (1 + r)
        twr -= 1

        # Money-weighted return approximation (CFA L1 R52)
        # MWR ≈ total return / average capital deployed
        mwr = total_return_pct / 100

        # Portfolio composition
        positions_value = sum(p.market_value for p in positions)
        cash_pct = (self.cash / nav * 100) if nav > 0 else 100
        invested_pct = 100 - cash_pct

        return PerformanceMetrics(
            total_return=round(total_return, 2),
            total_return_pct=round(total_return_pct, 2),
            annualized_return=round(ann_return * 100, 2),
            daily_returns_std=round(daily_std * 100, 4),
            annualized_volatility=round(ann_vol * 100, 2),
            sharpe_ratio=round(float(sharpe), 3),
            sortino_ratio=round(float(sortino), 3),
            calmar_ratio=round(float(calmar), 3),
            max_drawdown=round(max_dd * 100, 2),
            max_drawdown_duration_days=max_dd_duration,
            current_drawdown=round(current_dd * 100, 2),
            total_realized_pnl=round(total_realized, 2),
            total_unrealized_pnl=round(total_unrealized, 2),
            total_commissions=round(total_commissions, 2),
            total_trades=len(filled_trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=round(win_rate * 100, 1),
            avg_win=round(float(avg_win), 2),
            avg_loss=round(float(avg_loss), 2),
            profit_factor=round(float(profit_factor), 2),
            twr=round(twr * 100, 2),
            mwr=round(mwr, 2),
            num_positions=len(positions),
            cash_pct=round(cash_pct, 1),
            invested_pct=round(invested_pct, 1),
            days_active=days_active,
        )

    # ── Transaction History ──────────────────────────────────────────────────

    def get_transactions(self, limit: int = 50) -> list[dict]:
        """Get recent transactions, newest first."""
        return list(reversed(self._transactions[-limit:]))

    def get_nav_history(self) -> list[dict]:
        """Get NAV history for charting."""
        return self._nav_history

    # ── Serialization ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize the full account state for persistence."""
        return {
            "account_id": self.account_id,
            "starting_cash": self.starting_cash,
            "cash": self.cash,
            "created_at": self.created_at,
            "positions": self._positions,
            "transactions": self._transactions,
            "nav_history": self._nav_history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PaperAccount":
        """Restore account from serialized state."""
        acct = cls.__new__(cls)
        acct.account_id = data["account_id"]
        acct.starting_cash = data["starting_cash"]
        acct.cash = data["cash"]
        acct.created_at = data["created_at"]
        acct._positions = data["positions"]
        acct._transactions = data["transactions"]
        acct._nav_history = data["nav_history"]
        return acct

    # ── Account Summary ──────────────────────────────────────────────────────

    def get_summary(self) -> dict:
        """Quick account summary."""
        nav = self.get_nav()
        return {
            "account_id": self.account_id,
            "nav": round(nav, 2),
            "cash": round(self.cash, 2),
            "positions_count": len(self._positions),
            "total_return_pct": round((nav / self.starting_cash - 1) * 100, 2),
            "total_trades": len([t for t in self._transactions if t["status"] == "filled"]),
            "created_at": self.created_at,
        }


# ── CFA Curriculum Tags ─────────────────────────────────────────────────────
# Maps concepts used in the platform to CFA readings

CFA_TAGS = {
    # Level I
    "sharpe_ratio": {
        "level": "I", "reading": "Portfolio Risk and Return: Part I",
        "topic": "Portfolio Management",
        "formula": "S = (Rp - Rf) / σp",
        "explanation": "Measures excess return per unit of total risk. Higher is better. "
                       "A Sharpe above 1.0 means you earn more than one unit of return "
                       "for each unit of risk taken."
    },
    "sortino_ratio": {
        "level": "II", "reading": "Portfolio Risk and Return: Part II",
        "topic": "Portfolio Management",
        "formula": "Sortino = (Rp - Rf) / σ_downside",
        "explanation": "Like Sharpe but only penalizes downside volatility. "
                       "Upside volatility is good — Sortino doesn't punish it."
    },
    "beta": {
        "level": "I", "reading": "Portfolio Risk and Return: Part II",
        "topic": "Portfolio Management",
        "formula": "β = Cov(Ri, Rm) / Var(Rm)",
        "explanation": "Measures sensitivity to market movements. β=1.0 moves with "
                       "the market. β>1 amplifies moves. β<1 dampens them."
    },
    "capm": {
        "level": "I", "reading": "Portfolio Risk and Return: Part II",
        "topic": "Portfolio Management",
        "formula": "E(Ri) = Rf + βi × (E(Rm) - Rf)",
        "explanation": "The Capital Asset Pricing Model. Expected return equals the "
                       "risk-free rate plus beta times the market risk premium."
    },
    "efficient_frontier": {
        "level": "I", "reading": "Portfolio Risk and Return: Part I",
        "topic": "Portfolio Management",
        "formula": "min σp² = wᵀΣw subject to wᵀμ = target, wᵀ1 = 1",
        "explanation": "The set of portfolios offering the highest return for each "
                       "level of risk. Portfolios below the frontier are suboptimal."
    },
    "var_95": {
        "level": "II", "reading": "Measuring and Managing Market Risk",
        "topic": "Risk Management",
        "formula": "VaR₉₅ = -(μ - 1.645σ)",
        "explanation": "Value at Risk: the maximum expected daily loss with 95% confidence. "
                       "There's a 5% chance of losing more than this amount on any given day."
    },
    "cvar": {
        "level": "II", "reading": "Measuring and Managing Market Risk",
        "topic": "Risk Management",
        "formula": "CVaR₉₅ = E[Loss | Loss > VaR₉₅]",
        "explanation": "Conditional VaR (Expected Shortfall): the average loss in the "
                       "worst 5% of scenarios. Captures tail risk better than VaR."
    },
    "max_drawdown": {
        "level": "I", "reading": "Portfolio Risk and Return: Part I",
        "topic": "Portfolio Management",
        "formula": "MDD = (Trough - Peak) / Peak",
        "explanation": "The largest peak-to-trough decline. Shows the worst loss an "
                       "investor would have experienced. Critical for risk tolerance."
    },
    "twr": {
        "level": "I", "reading": "Portfolio Risk and Return: Part I",
        "topic": "Portfolio Management",
        "formula": "TWR = ∏(1 + ri) - 1",
        "explanation": "Time-weighted return removes the effect of cash flows. "
                       "Used to evaluate the portfolio manager's skill independent "
                       "of when money was added or withdrawn."
    },
    "mwr": {
        "level": "I", "reading": "Portfolio Risk and Return: Part I",
        "topic": "Portfolio Management",
        "formula": "Solve: Σ CFt / (1+r)^t = 0",
        "explanation": "Money-weighted return (IRR) reflects the investor's actual "
                       "experience including timing of cash flows. Penalizes bad "
                       "timing even if the underlying portfolio performed well."
    },
    "correlation": {
        "level": "I", "reading": "Portfolio Risk and Return: Part I",
        "topic": "Portfolio Management",
        "formula": "ρij = Cov(Ri, Rj) / (σi × σj)",
        "explanation": "Measures how two assets move together. ρ=1 means perfect "
                       "co-movement (no diversification). ρ<0 means they offset each "
                       "other (hedging benefit)."
    },
    "black_scholes": {
        "level": "II", "reading": "Valuation of Contingent Claims",
        "topic": "Derivatives",
        "formula": "C = S₀N(d₁) - Ke^(-rT)N(d₂)",
        "explanation": "Prices a European call option from five inputs: stock price, "
                       "strike, time, rate, and volatility. The foundation of all "
                       "options pricing."
    },
    "greeks_delta": {
        "level": "II", "reading": "Valuation of Contingent Claims",
        "topic": "Derivatives",
        "formula": "Δ = ∂C/∂S = N(d₁)",
        "explanation": "How much the option price changes per $1 move in the stock. "
                       "Delta of 0.5 means the option moves ~$0.50 for each $1 stock move."
    },
    "covered_call": {
        "level": "III", "reading": "Options Strategies",
        "topic": "Derivatives",
        "formula": "Payoff = min(S-K, 0) + premium collected",
        "explanation": "Sell a call against stock you own. Collects premium income "
                       "but caps your upside at the strike price. Most common "
                       "options income strategy."
    },
    "risk_parity": {
        "level": "III", "reading": "Asset Allocation",
        "topic": "Portfolio Management",
        "formula": "RCi = wi × (Σw)i / σp, target: RC₁ = RC₂ = ... = RCn",
        "explanation": "Each position contributes equal risk to the portfolio. "
                       "Unlike equal-weight, this accounts for volatility and correlation."
    },
    "information_ratio": {
        "level": "II", "reading": "Active Portfolio Management",
        "topic": "Portfolio Management",
        "formula": "IR = (Rp - Rb) / σ(Rp - Rb)",
        "explanation": "Measures active return per unit of tracking error. "
                       "An IR above 0.5 is considered skilled active management."
    },
}


def get_cfa_tag(concept: str) -> dict | None:
    """Look up CFA curriculum tag for a concept."""
    return CFA_TAGS.get(concept)


def get_all_cfa_tags() -> dict:
    """Return all CFA curriculum tags."""
    return CFA_TAGS
