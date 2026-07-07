# Landing Page + Sign-In Integration — Scope

## Origin
Session 2026-04-20. User instinct: current onboarding is fine,
but the sign-in page (the first thing a new user sees) tells them
nothing about what Quantex is. New users from r/CFA or similar
traffic sources will bounce before learning the product.

## Decision
Don't build a separate landing page. Instead, the sign-in page
becomes the landing page. Same URL, same component, enriched with
hero + value props above the existing sign-in form.

## Not building
- Separate marketing domain
- Video or animated content
- Testimonials (no real users yet)
- Mobile responsive (deferred with rest of mobile work)
- A/B testing infrastructure

## Content decisions (made this session)

### Hero headline — candidate drafts (pick next session):
1. "CFA exam prep with an AI tutor that knows your portfolio"
2. "Study CFA concepts by applying them to real portfolios —
   with an AI tutor that teaches"
3. "The AI-powered CFA prep platform that uses real portfolio
   analysis as your study material"

Prefer shorter unless the longer one genuinely says more.

### Three value props (equal weight between CFA + portfolio tools):
1. "172+ CFA exam questions across all 12 topic areas, Levels 1
   and 2 coverage" (accurate to current content, don't overstate)
2. "AI tutor that explains your portfolio using CFA frameworks —
   Sharpe, beta, VaR, optimization, risk parity" (differentiator)
3. "Professional portfolio tools: correlation engine with
   Ledoit-Wolf shrinkage, stress testing, efficient frontier"
   (credibility for the finance-serious audience)

### Primary CTA: Sign Up
- Big button: "Sign Up Free"
- Secondary link: "Try as guest first" (small text below or next to)
- Sign-in for returning users: "Already have an account? Sign in"
  (tertiary, smallest)

### Layout structure:
- Hero: big headline, subheading (1 sentence elaboration)
- 3 value props in row (cards or icons + text)
- Sign-up form inline below, or CTA button that expands form
- Footer: tiny placeholder for legal stuff later

## Technical scope

### Files to modify
- backend/static/quantex.html: the login/signup component
  (currently around the top of the file — grep for "Continue
  as Guest" or similar to locate)

### Effort budget
- Copy refinement (pick headline, finalize value props): 30 min
- Component restructure (hero + value props + existing form): 45 min
- Styling (typography, spacing, colors consistent with app): 45 min
Total: ~2 hours

## Explicit scope constraints
- No new React dependencies
- No separate routing — same single-page structure
- Keep desktop-only (mobile deferred)
- Must not break the existing sign-in flow for returning users
  who already have localStorage accounts
- The onboarding wizard (Investment Goals) stays as-is — it runs
  AFTER signup. Don't touch that.

## Next session first actions
1. Review current login component code before changing anything
2. Pick the hero headline from the 3 candidates
3. Build the new landing layout as a single commit
4. Test: (a) logged-out visitor sees landing, (b) guest flow still
   works, (c) returning user sees sign-in normally
5. Commit with message like "Add landing content to sign-in page"

## Do not start until previous work pushed and verified
All Phase 1.1 work (AI Adviser Replicate switch) is shipped and
verified. Phase 1.2 (CFA question validation) complete.
