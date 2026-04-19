# CLAUDE.md — Quantex Project Context

> This file is loaded by Claude Code at the start of every session. Keep it concise; every line should earn its place.

## Project

Quantex is a portfolio intelligence platform and CFA exam prep tool. Full-stack Python/FastAPI backend + single-file React frontend, deployed via Docker on Render.

**Live:** https://quantex-jjrk.onrender.com
**Repo:** https://github.com/justinchandler867/quantum (main branch, auto-deploys to Render)
**Target market:** CFA candidates (400K+ globally). Positioned between Kaplan Schweser and Bloomberg Terminal.

## Commands

\`\`\`bash
# Local dev (from backend/)
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Deploy
git add -A && git commit -m "..." && git push origin main
# Render auto-deploys from main on push
\`\`\`

Python: \`python3.12\` from python.org (NOT system Python on macOS).

## Architecture

**Frontend:** Single file \`backend/static/quantex.html\` (~2,400 lines).

- React 18 via CDN, no build step, no node_modules
- Pure \`React.createElement\` (no JSX)
- 19 components, 15 tabs, all state in localStorage
- Fonts: Outfit (display), DM Sans (body), JetBrains Mono (data)
- Client-side Black-Scholes pricer with hand-rolled normal CDF
- Falls back to 38 sample assets when backend is unreachable
- Mobile responsive layer added via CSS media queries at <=768px and <=480px + vanilla-JS drawer toggle. Zero React logic changes.

**Backend:** FastAPI, ~5,400 lines across \`backend/app/\`:

| File | Purpose |
|---|---|
| \`main.py\` | FastAPI app, 27+ endpoints, AI proxy, serves frontend at \`/\` |
| \`paper_trading.py\` | Virtual \$100K account, TWR/MWR, CFA concept tags |
| \`derivatives.py\` | BSM pricer, Greeks, IV surface, 8 strategy payoffs |
| \`fundamental.py\` | Claude-powered 3-step company analysis |
| \`screener.py\` | 4-stage Nasdaq screening pipeline |
| \`optimizer.py\` | scipy SLSQP, 4 objectives, efficient frontier |
| \`correlation_engine.py\` | Regime-aware correlations, Ledoit-Wolf shrinkage |
| \`data_ingest.py\` | yfinance (PATCHED to use \`Ticker.history()\` — \`yf.download\` is broken in recent versions) |
| \`cache.py\` | Redis with in-memory fallback |
| \`models.py\` | Pydantic schemas |
| \`config.py\` | Constants, thresholds |

Same-origin deployment: FastAPI serves HTML at \`/\`, no CORS needed.

## Conventions

- **No JSX in the frontend.** Intentional (no build step). Use \`React.createElement\` or aliased \`e(...)\`. Don't suggest JSX rewrites unless the whole build-step conversation happens first.
- **Class names crowded.** Before adding new CSS classes, check if a short form already exists (\`.btn\`, \`.card\`, \`.stat-box\`).
- **Mobile CSS lives at the bottom of the \`<style>\` block.** Add new mobile rules to the existing \`@media (max-width: 768px)\` block.
- **API calls** use \`apiFetch()\` with 180s timeout. New endpoints reuse this.
- **localStorage keys** prefixed with \`qx_\`. Maintain this for eventual Supabase migration.

## Current state and open workstreams

### Recently shipped (April 2026)

- Mobile responsive layer (breakpoints 768px and 480px, slide-out AI Advisor drawer)
- CFA question bank: live CourseHub has **6 modules / ~22 inline questions** (not 12/43 as previously noted here)
  - Expanded 173-question bank exists only as a chat artifact — NOT in the repo yet
  - Needs to be saved to \`backend/static/cfa-questions-combined.json\` before wiring
  - Wiring will be gated behind a staging flag (default off) so the 22 inline remain the live source until spot-checked

### Next up (priority order)

1. **Wire CFA question bank into CourseHub** (~30 lines in CourseHub component)
2. **Supabase auth** — replace localStorage \`LoginGate\` (~line 1808) with real accounts
3. **Batch 3 of CFA questions** (target 300+, adds L2 vignette item sets)
4. **Render starter plan migration** — currently free tier, sleeps after 15min causing 30s cold start
5. **Replace yfinance** with Financial Modeling Prep or Polygon.io

### Known issues

- yfinance \`yf.download()\` broken in newer versions -> patched to per-ticker \`Ticker.history()\` in \`data_ingest.py\`
- No test suite — CFA question explanations and calculation outputs need manual review
- 15 tabs is a lot for mobile; nav relies on horizontal scroll with fade mask
- No spot check done yet on mobile responsive on real iPhone/Android

## What NOT to do

- Don't introduce a build step for the frontend (bundler, transpiler, npm install). Single-file deployment is a feature.
- Don't rewrite working components "for clarity" — they're optimized for terseness. Refactor only when functionality changes.
- Don't add features to \`DerivativesSandbox\` or \`StressSimulator\` without confirming mobile acceptable.
- Don't modify \`data_ingest.py\`'s yfinance call pattern without testing — already worked around compatibility issues.
- Don't use localStorage directly in new components — wrap it in a helper so the Supabase migration is single-surface.

## Business context

- **Monetization:** Free tier (paper trading, sample data, basic metrics) -> paid \$15-25/month (derivatives sandbox, AI adviser, advanced analytics, all challenges)
- **Regulatory positioning:** Educational tool, NOT investment advice. Avoid language construable as recommending specific investments.
- **AI adviser:** Tries server proxy (\`/api/ai/chat\` with \`ANTHROPIC_API_KEY\` env var) first, falls back to user's own API key via gear icon. Claude Max subscription does NOT include API access.
- **Differentiator:** CFA prep with a real portfolio lab attached.

## Future workstreams (parked, not yet started)

These are committed-to but deferred until current question expansion
+ wiring is complete. When the user says "let's revisit the 8
suggestions" or "let's start essay grading" or "add the 13th module,"
load the relevant section.

### AI enhancement suite (8 items)
1. Upgrade AI system prompt — CFA-trained quant educator persona with
   structured behavioral rules (tie analysis to CFA concepts, show
   formulas before numbers, no specific buy/sell advice)
2. Inject portfolio context into every AI request — current portfolio
   state, profile, recent activity prepended silently
3. "Study with my portfolio" mode — new tab/button where user picks
   a CFA concept and AI generates exercise using their actual holdings
4. Quiz-aware AI tutoring — when user gets quiz wrong, AI walks
   through it using their actual portfolio data, not the canned explain
5. Two AI personas — Tutor (Socratic, patient) vs Analyst (fast,
   interpretive). Same model, different system prompts, mode toggle
6. Curriculum coverage tracker — AI tracks which CFA LOSes the user
   has practically engaged with through trades and analysis
7. Cross-session memory — implement when Supabase auth lands; AI
   remembers prior conversations
8. Confidence-calibrated answers — AI explicitly marks uncertainty
   rather than confidently bullshitting

Strategic frame: position is "CFA prep with AI tutor that knows your
real portfolio" — NOT "AI hedge fund manager" (regulatory + liability
+ capability honesty concerns).

### 13th module: Corporate Issuers (L1)
Closes the curriculum gap — current 12 modules miss this CFA L1 topic
area. Target ~12-15 questions covering: capital structure, working
capital management, corporate governance, ESG considerations.
Build after the 4 remaining beef-ups complete.

### Level III essay grading with AI rubric assessment
Substantial feature (2-3 week build). Architecture:
- Question + rubric data structure (each Q has structured rubric:
  total points, key points with point thresholds, partial-credit
  guidance)
- New UI for essay composition and feedback display
- AI grading via structured prompt: evaluates each rubric criterion,
  shows work, awards points with justification
- Always show the rubric used (not just a score) so candidate can
  challenge
- Position as "AI-assisted practice grading" not "equivalent to CFA
  Institute scoring"
- Initial bank: 20-30 essay questions, mix of original + adapted
  from public CFA past papers + Claude-generated reviewed by user
Build after AI enhancement suite — needs the better AI scaffolding
to grade well.

## User preferences

- Prefers concrete, shippable deliverables over extensive planning docs
- Wants honest assessment of tradeoffs, not validation
- Comfortable with iterative shipping
- Testing infrastructure is weak; expect manual review of LLM-generated content
