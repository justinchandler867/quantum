# Quantex 3.0 Roadmap

Captured from planning session April 22, 2026.

## 1. First-page routing
New users (signup, login, guest) should land on Profile first,
then Discovery after Profile is complete.

## 2. Clarify the Discovery process
Current: users see ~170 pre-filtered tickers with no explanation
of why these, not others. Add a visible "How are these chosen?"
affordance explaining profile match, factor scores, fit calculation.

## 3. TA indicators need user-friendly interpretation
Keep all 8 indicators. Add plain-English help. Consider Buy/Sell/
Hold stamps on the ticker chart based on indicator consensus.

## 4. Optimizer navigator (formerly "optimizer optimizer")
A guided tool for beginner-to-moderate investors to pick between
Max Sharpe / Min Vol / Risk Parity / Max Diversification. Already
have partial version — needs to be better. Must explain what each
optimizer does.

## 5. Broader Discovery universe
Current ~170 tickers is too narrow. The full tickers.json (10,932)
is now available via search. Discovery universe could be expanded
or dynamically curated.

## 6. MBA + CFA thinking
Current app is heavy on TA, light on fundamentals. Add:
- Earnings dates
- Ex-dividend dates
- P/E, forward P/E
- Dividend growth, revenue growth
- Debt/equity
- Volume TA indicator
CFA thinking alone can miss qualitative factors. MBA thinking adds
fundamentals, business quality, strategic positioning.

## 7. MBA/CFA value prop on landing page
Current landing mentions CFA only. MBA angle needs to come through
— quantitative + qualitative, technical + fundamental analysis.

## 8. News aggregation per ticker
Each ticker in Discovery or the ticker modal should show recent
relevant news. Requires picking a news API (Finnhub, Polygon,
NewsAPI, etc.)

---

## Thursday-ready sprint priority

1. First-page routing (#1)
2. MBA angle on landing page (#7)
3. "How these are chosen" on Discovery (#2)
4. Stretch: Buy/Sell/Hold signal on ticker chart (#3 scoped down)
