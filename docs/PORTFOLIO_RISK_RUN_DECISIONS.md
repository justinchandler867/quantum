# PORTFOLIO_RISK_RUN_DECISIONS.md: Build 3 judgment calls (unattended run)

**Date:** 2026-09-28. **Branch:** build3-portfolio-health, cut from demo-prep at 06739ea.

## How to read this log

- The spec's STOP conditions were converted into logged judgment calls, per the run instructions.
- Each call used the most defensible choice, biased against the product looking better than it is.
- Format for each entry: ambiguity, options, choice, reasoning.
- **26 judgment calls.**

---

**J1. What "Build docs/PORTFOLIO_RISK_SPEC.md including Amendment A" means.**
- **Ambiguity:** the spec, including Amendment A §8, already exists in the repo (commits 25c3423 and 2f28838).
- **Options:** (a) rewrite the spec; (b) implement it.
- **Choice:** (b). The acceptance items, commit message and ranker measurement all describe an implementation.
- **Reasoning:** the spec file is not modified. Sealed specs are not edited after the fact.

**J2. §1 Ledoit-Wolf reuse versus sample covariance (converted STOP).**
- **Situation:** `app/return_models.py::ledoit_wolf_constant_correlation` is pure numpy, takes a (T, N) array, and is callable without modifying shipped code.
- **Choice:** reuse it; no fork.
- **Hand check:** evidence item 9 re-derives it from the 2004 paper and matches to about 1e-17.
- **Consequence, disclosed:** the Vol line (shrunk Σ) and the Sharpe denominator (the portfolio series' own realized volatility, which is the shipped per-ticker method) differ slightly. On live data: 12.90% versus 12.95%. The Sharpe tooltip says so explicitly.

**J3. Return transform.**
- **Options:** (a) the store's daily log returns, which is the exact series the β chip and correlation column use; (b) simple returns via `expm1`.
- **Choice:** (a), so there is one series.
- **Details:** the drawdown path is `exp(cumsum r_P)`. The panel header states "daily log returns".
- **Cost:** a weighted sum of log returns approximates the log of the weighted simple return. The error is second order in daily returns, about r²/2, or 0.02 pp on a 2% day.

**J4. The §1 "fewer than 60 overlapping days" rule on a store with no gaps.**
- **Finding:** `fetch_prices` forward-fills gaps and then drops any row that still has one, so the store has no missing values. A ticker that is too short is simply absent.
- **Choice:** three explicit reasons, never imputed:
  - `not_loaded` when the column is absent;
  - `insufficient_history` when there are fewer than 60 non-missing days in the window;
  - `insufficient_overlap` when the included holdings' common rows are fewer than 60.

**J5. §2 preview weight (converted STOP).**
- **Finding:** there is no default-lot concept. An actual add (`App.addToPort`) re-splits all holdings equally at `floor(100/n)`, with the remainder going to the first key.
- **Choice:**
  - Extract that exact rule into `addWeights`/`removeWeights` and make the real add/remove call them. Semantics are byte-identical; a test pins them.
  - The preview uses the same functions and states the weight visibly: "(adding X at 25% weight · equal re-split of 4 holdings)".
  - The spec's 10% default was not used, because no real add produces 10%.

**J6. Remove preview semantics changed.**
- **Finding:** the shipped β remove-preview computed β on the remaining weights renormalized, but `rmFromPort` actually re-splits them equally, so the old preview described a portfolio the click never produces.
- **Choice:** the new card computes both lines (σ and β) on the weights the click actually produces. The β-preview numbers therefore change versus the shipped chip preview.
- **Reasoning:** logged because it alters a shipped readout.

**J7. The preview reads only the loaded store.**
- **Options:** (a) call `_ensure_data` as `/api/portfolio/beta` does; (b) use only what is already loaded.
- **Choice:** (b) for `/api/portfolio/preview` and `/api/discovery/redundancy`. `/api/portfolio/health` does call `_ensure_data`.
- **Reasoning:** `_ensure_data` re-fetches the whole universe when any ticker is missing, which takes minutes, and a mouse hover must never do that. Unloaded tickers come back as `excluded` or `not_loaded` and render as "—".

**J8. Reweight preview.**
- **Choice:** the base is the weights at slider pointer-down. The card shows base → now until the next interaction, and is dropped if the holding set changes.

**J9. What triggers the Discovery preview.**
- **Choice:** hovering the per-portfolio add/remove buttons (`+ A` / `✓ A`), since the click on the row itself opens the ticker modal. The card is fixed at the bottom and names the portfolio.

**J10. §4 effective bets with a negative RC (converted STOP).**
- **Choice:** |RC| normalization, `s_i = |RC_i| / Σ|RC_j|`, disclosed in the tooltip. With no negative RC it equals the spec formula exactly (tested).
- **Literature alternatives considered:** Meucci (2009, "Managing Diversification") entropy of principal-component bets, and Meucci, Santangelo & Deguest (2015) minimum-torsion bets. Both are cleaner for correlated assets but measure a different object: bets on rotated, uncorrelated factors. That would need a factor rotation and different copy. Not adopted; noted for a future spec decision.

**J11. Where effective bets sit.**
- **Conflict:** §4 says "next to the beta chip"; §8.2 says the Structure block is "their home".
- **Choice:** Structure block, since the amendment supersedes. The β chip sits in Sensitivity (§8.3).

**J12. §5 endpoint shape (converted STOP).**
- **Contract:**
  - Request: `POST /api/discovery/redundancy`, body `{candidates: [≤100 tickers], window: 252}`.
  - Response: `{threshold, window, flags: [{ticker, peer, r, days}], not_loaded: [...], n_pairs}`.
- **Why a new endpoint:** `/api/discovery/context` returns only candidate-versus-portfolio values, not candidate-versus-candidate, so the smallest addition is one batched call.
- **Rules:** the threshold is `HIGH_CORR_THRESHOLD` from config (0.85). The candidate set is the whole shortlist, which is the same list the Corr column sends, not the type-filtered subset.
- **Consequence:** a marker can name a peer hidden by the current type filter.

**J13. §8.2 pairwise correlation.**
- **Choice:** sample Pearson correlations over the window.
- **Proof of consistency:** under constant-correlation shrinkage each ρ moves toward ρ̄, so the average is identical under the sample and the shrunk matrix (proved in a test). The highest pair is reported from the sample.

**J14. §8.3 capture-ratio observation guard (converted STOP).**
- **Conflict:** the spec says 30. The codebase already requires `MIN_OVERLAP_DAYS = 60` before showing any co-movement statistic (correlation column, β).
- **Choice:** 60, reusing the constant.
- **Reasoning:** the stricter guard shows "—" more often rather than a noisy ratio. With a 252-day window both sides normally exceed 60 (live: 135 up / 117 down).

**J15. β chip relocation.**
- **Choice:** the shipped chip was moved verbatim into the Sensitivity block (§8.3: "renders here").
- **Flag:** it keeps its shipped green/amber band-state colors ("In band ✓" is green, from BETA_TRACKER_SPEC). That conflicts with §8's "no green/red good/bad semantics". It was not changed, because it is governed by a sealed earlier spec; this is a decision for the owner.

**J16. §8.4 Sharpe method.**
- **Finding:** two per-ticker methods ship.
  - `/api/ticker/add`: (1Y return − rf) ÷ vol.
  - The screener, which produces the whole Discovery shortlist: mean(log r)×252 ÷ vol, **with no rf**.
- **Choice:** the rf-subtracting method, which matches §8.4 ("same rf") and the Discovery tooltip's stated definition.
- **Details:** SPY's reference Sharpe is computed by the same function over the same window. It is not read from the Discovery SPY row.
- **Consequence, flagged:** the panel's Sharpe is not comparable to the numbers in the Discovery SHARPE (1Y) column.

**J17. §8.4 Sortino.**
- **Choice:** deferred; ship Sharpe only.
- **Reasoning:** a second ratio carrying the same market-comparison labels could contradict Sharpe's label, which is exactly the ambiguity §8.4 names.

**J18. §8.5 bucketing code path.**
- **Finding:** §B's path is `ASSETS[].sec → sleeveOf`.
- **Choice:** extracted `sectorOf(tk)`, now used by both the §B chart and §8.5, giving one lookup.
- **Caveat:** `sec` comes from sectors.json only for screened names. Tickers added by symbol carry yfinance's sector or quoteType (for example "MUTUALFUND").

**J19. §8.5 yield (J2 convention).**
- **Method:** per holding, trailing-12-month cash dividends ÷ latest unadjusted close, from `yfinance history(auto_adjust=False)`. Fetched at panel load and cached for 18 h.
- **Failures:** listed as "(excludes X)", never counted as 0.
- **Cost:** adds one network call per holding on the first panel load.

**J20. Panel placement and the old stat grid.**
- **Choice:** the panel is placed at the top of the builder's right column. The shipped `calcStats()` stat grid below it (RETURN, VOL σ, SHARPE, BETA, SORTINO, TREYNOR, VAR, MAX DD) was left in place.
- **Flag:** that grid uses random-noise correlations and parametric formulas, so the builder now shows two different Vol / VaR / Max DD / Sharpe / β values for the same portfolio. Removing shipped UI is outside this spec; this is the first item flagged in the report.

**J21. Weight basis when holdings are excluded.**
- **Choice:** risk rows re-base weights to the holdings with history, and the panel says so ("Risk rows re-base weights to the N holdings with history."). Composition facts use all holdings.
- **Reasoning:** without the note, the same ticker could show two weights on one panel.

**J22. Holdings count.**
- **Choice:** "N holdings" counts every holding, including excluded ones. The exclusion qualifier appears once at panel level (§8.6).

**J23. Empty state.**
- **Choice:** PortTab's existing empty card is kept, and the panel renders beneath it with "Add holdings to see portfolio analytics", with no placeholder numbers.

**J24. The ranker "standard offline fixture".**
- **Finding:** no fixture by that name exists.
- **Definition used** (`scripts/ranker_order_dump.py`):
  1. Frontend: the shipped `FALLBACK_ASSETS` through the Discovery row pipeline, `fitScore` and `cmpRows`, plus 40 seeded z-scored rows so the factor re-rank path runs. 4 goals × 3 horizons × 2 derivative-experience answers, both sort directions.
  2. Backend: `compute_factor_scores → rank_by_composite → compute_fit_scores` on the cached backtest prices (`backtest/data/raw`, local and gitignored; 585 tickers × 751 days), fundamentals derived deterministically. 4 goals × 3 risk scores × 3 horizons.
- **Result:** the dump is taken before and after; the diff must be empty.

**J25. Commit handling.**
- **Choice:** the run instructions' commit message and "commit and push the branch" supersede the spec's own commit message and "no commit until audit".

**J26. How UI rendering is verified.**
- **Finding:** headless Chrome became unusable mid-session (it auto-updated).
- **Choice:** rendered strings come from the shipped `healthPanelBody` and `volPreviewLines` executed under node with stub React, fed payloads the real engine computed.
- **Limitation:** layout and visual styling are not verified in a browser.
