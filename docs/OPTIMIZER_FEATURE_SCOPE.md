# Portfolio Optimizer Feature — Scope

## Origin
Session 2026-04-20: user requested "more optimization features and
AI advisor recommendation for how to weight the portfolio." This
expands Quantex beyond its current CFA-prep positioning into
portfolio construction territory. Framing decision pending (see below).

## Decided scope
- 4 optimizers: Max Sharpe, Min Variance, Risk Parity, Black-Litterman
- Backend first (next session), frontend second, AI explanation third
- Realistic: ship Max Sharpe + Min Var + Risk Parity next session;
  Black-Litterman likely slips to a follow-up session due to UI
  complexity around user "views"

## Undecided — must resolve before shipping
**Framing**: Educational tool vs. tool-with-user-ownership
- Educational framing: "Here's what a CFA-style optimizer produces"
  — reinforces CFA prep positioning, stays clear of advisor line
- Tool-ownership framing: "Optimizer output — apply?" with
  explicit user-click-to-apply flow
- Decision deferred. Affects UI copy and AI system prompts only,
  not backend math. Can build backend without resolving this.

## Backend work (next session target)
New endpoint: POST /api/portfolio/optimize
Input: { holdings: [...], optimizer: "max_sharpe" | "min_var" |
         "risk_parity" | "black_litterman", constraints: {...} }
Output: { weights: {...}, expected_return, expected_vol,
          sharpe, risk_contribution: {...}, notes: [...] }

Leverage existing infrastructure:
- correlation_engine.py (Ledoit-Wolf shrinkage already there)
- Historical returns pipeline via data_ingest.py
- Stress-regime-aware covariance already available

Math libraries:
- Max Sharpe & Min Var: closed-form via numpy.linalg.solve on the
  Ledoit-Wolf-shrunk covariance matrix. ~30 lines each.
- Risk Parity: iterative (Newton or cyclical coordinate descent).
  Use scipy.optimize. ~50 lines.
- Black-Litterman: reverse-optimization from market-cap weights +
  user views + tau/omega parameters. ~100 lines plus requires
  market-cap data source (yfinance has it but inconsistent).

Constraints to support:
- Long-only (no short positions)
- Max-per-asset weight cap
- Sum-to-1

## Frontend work (session after backend)
New section in ◆ A tab below HOLDINGS DETAIL: "Optimize"
4 cards in a grid (or 3 if Black-Litterman deferred).
Each card:
- Optimizer name + 2-sentence description
- "Run on Portfolio A" button
- On click: side-by-side weights comparison (current vs. suggested)
- Sharpe/vol/return metrics delta
- AI explanation panel (Replicate-backed) — prompt varies by optimizer

## AI explanation layer (session 3)
System prompt per optimizer, each teaching the concept using the
user's own portfolio as the example. Keep each explanation under
200 tokens output (max_tokens: 1024 per Replicate minimum).
Example Max Sharpe system prompt:
"You are a CFA Level 2 instructor. The student's portfolio has been
run through a Max Sharpe optimizer. Explain: (1) what the optimizer
is trying to do in one sentence, (2) which three positions changed
most and why, (3) one legitimate CFA L3 critique of pure Max Sharpe
portfolios. Be specific to the numbers provided. Under 200 words."

## Regulatory reminder
Per Investment Advisers Act of 1940: specific, tailored buy/sell/hold
recommendations for compensation may constitute regulated investment
advice. Educational framing (Framing 1) keeps Quantex clearly on the
teaching side of the line. Direct advice framing (Framing 3) requires
RIA registration. Framing 2 is grey zone. Resolve framing before
shipping to production.

## Time estimate
Backend: 2-3 hours
Frontend: 2-3 hours
AI prompts + tuning: 1-2 hours
Total: 5-8 hours across ~3 sessions

## Do NOT start this work until:
1. Today's uncommitted work is all committed (PortTab fix, AI Advisor
   Replicate switch from f550302)
2. Fix #3 (AI Adviser) is pushed to production and verified working
3. User has resolved framing decision (or at least acknowledged
   building backend without resolving it)
