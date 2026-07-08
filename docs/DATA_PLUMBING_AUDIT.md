# DATA_PLUMBING_AUDIT.md — How Quantex Actually Handles Market Data

Read-only audit. No code changed, nothing committed. Every claim carries a
`file:line` citation verified 2026-07-08. Cross-references the overnight backtest
where relevant (BACKTEST_RUN_DECISIONS.md).

---

## a. Price adjustment — adjusted or raw? splits/dividends? consistency?

**Production is on adjusted (total-return) closes — implicitly.**
`fetch_prices` calls `yf.Ticker(t).history(start, end)` and takes the `"Close"`
column (`app/data_ingest.py:53-55`). It does **not** pass `auto_adjust`. In the
installed yfinance (1.3.0) `history()` defaults to `auto_adjust=True`, so `"Close"`
is **split- and dividend-adjusted** — a total-return series. The docstring calls
it "adjusted close" (`app/data_ingest.py:32`).

- **Splits & dividends:** both are folded into the adjusted price by yfinance's
  auto-adjustment. Dividends are not separately modeled in production returns —
  they live inside the adjusted price path.
- **Consistency across the app:** returns (`compute_log_returns`,
  `app/data_ingest.py:86-92`), Sharpe (`app/main.py:655-671`), and correlation
  (`app/correlation_engine.py` consumes the same `_store["returns"]`) all derive
  from this one adjusted-price frame → **internally consistent**.
- **Backtest consistency:** the backtest fetched with `auto_adjust=False` and
  explicitly uses `"Adj Close"` for returns (== auto-adjusted close) and raw
  `"Close"` only for the point-in-time yield denominator
  (`backtest/backtest_fit.py`; BACKTEST_RUN_DECISIONS.md J2). So today the backtest
  and production agree — but via **different mechanisms**: the backtest *pins*
  adjustment; production *relies on a library default*.

**Weak point:** production never pins `auto_adjust`. yfinance has changed this
default historically. If it flips, `"Close"` silently becomes raw and every
Sharpe/return/correlation number shifts with no error and no visible signal.

## b. Survivorship — what universe, can a delisted ticker appear?

Screens draw from `app/data/universe.json` (S&P 500 + ETFs + extended, 596 names)
loaded as **today's membership** (`app/screener.py` universe load; gates at
`app/config.py:58-61`). A delisted ticker can appear **only** if it still sits in
`universe.json` or a user explicitly searches it and yfinance still serves it.
Because `universe.json` is a current snapshot, **companies that were delisted
before today are absent** — the classic survivorship hole.

**Interaction with the backtest 2016–2025 freezes:** the backtest inherits this
exactly (BACKTEST_RUN_DECISIONS.md **J1**). Every freeze's eligible set is drawn
from survivors, so the momentum factor is applied to names we already know
survived. The 1000-portfolio **null draws from the same survivor set**, which
cancels *plain* survivorship — but not the **momentum×survivorship interaction**:
high-momentum names whose eventual crash led to delisting are missing, truncating
momentum's left tail. This is the leading suspect for the draft **Growth-95 PASS**
(median Sharpe percentile 78.8), which the spec pre-registered as a trap ("if
Growth-95 PASSes cleanly, suspect the backtest first" — VALIDATION_SPEC.md §7 P5).
A pass here is weak evidence; a failure would have been strong.

## c. Missing data / gaps — handled vs silently corrupting

**Handled (explicit):**
- Tickers with < 60% of expected trading days are dropped (`app/data_ingest.py:66-72`).
- Price gaps forward-filled up to 5 days (`app/data_ingest.py:75`).
- Factor scoring requires ≥ `SCREEN_MIN_HISTORY_DAYS` (252) of returns
  (`app/screener.py:369`).
- Correlation column: overlap < `MIN_OVERLAP_DAYS` (60) → `insufficient_data`,
  renders `—`, "never default to 0" (`app/discovery_context.py:34,164`; §A4).
- Multi-horizon 3Y/5Y: insufficient history → `None`, never a partial-window
  number (`app/screener.py:492-502`).
- Ticker Sharpe endpoint: < 20 closes → HTTP 422 (`app/main.py:652`).

**NOT handled / silently corrupting:**
1. **`result.fillna(0)`** at the end of factor scoring (`app/screener.py:454`):
   a missing factor (e.g. no dividend history) becomes **z = 0 = universe
   average**, silently treated as neutral rather than flagged or excluded. A
   name with no yield data looks like an average-yield name.
2. **`ffill(limit=5).dropna()`** (`app/data_ingest.py:75`): after the 5-day fill,
   `dropna()` drops any date with a remaining NaN in **any** column. One name's
   long halt, or one short-history name, silently **truncates the date matrix for
   every ticker** — the common-date intersection is never surfaced.
3. **Volume gaps → 0** (`app/data_ingest.py:81`, `fillna(0)`): a gap-filled zero
   drags down trailing average volume and can wrongly trip/skew the liquidity gate.
4. **Beta fallback to 1.0** when the benchmark column is absent
   (`app/screener.py:400-404`): a silent, plausible-looking wrong beta.
5. **"1Y" Sharpe on as few as 20 days** (`app/main.py:652-671`): the column
   labeled *Sharpe (1Y)* is computed over whatever history exists (≥ 20 days), and
   mixes a **simple total-period return** (`ret_1y`, line 655) with an
   **annualized** volatility — mislabeled for short-history names.

## d. yfinance reliability — caching, retries, rate-limit, cold start

- **Retries / rate-limit:** none. Each ticker is fetched in a bare try/except; a
  failure is logged at debug and the ticker is **silently skipped**
  (`app/data_ingest.py:51-59`). No backoff, no retry, no sleep. Only a *total*
  failure (zero frames) raises (`app/data_ingest.py:61-62`).
- **Cold start:** data loads **lazily on first request** — `lifespan` does not
  prefetch (`app/main.py:117-120`); `_store` starts empty (`app/main.py:103`) and
  `_ensure_data` fills it on first use (`app/main.py:143-176`). First request after
  a cold boot pays the full-universe fetch (the ~30s Render cold-start in CLAUDE.md).
- **Full-universe refetch on any new ticker:** if a request needs a ticker not in
  `_store`, `_ensure_data` refetches **everything** (`app/main.py:151-161`), not
  just the new name.
- **Caching:** `cache.py` is Redis with an in-memory dict fallback; on Redis
  failure it sets a `_redis = False` sentinel and **never retries** the connection
  (`app/cache.py:32-34`), running the rest of the process on a per-process dict
  (`app/cache.py:63`) that dies on restart. Daily TTL 18h (`app/config.py:47`).
  On Render free-tier sleep, the process restarts → `_store` and the in-memory
  cache are both lost → the next request is a full refetch.
- **What the user sees on partial failure:** the ranking is computed on whatever
  subset succeeded, with **no user-visible indicator** of how many names were
  dropped. A quietly-smaller universe looks identical to a complete one.

## e. The five weakest data-integrity points, ranked (findings, not fixes)

1. **Survivorship in `universe.json` (today's membership).** Biases every
   backward-looking statistic upward and contaminates the backtest; it is the
   single largest and least visible distortion, and it is the reason the Growth-95
   backtest PASS cannot be taken at face value. (a/b above; backtest J1.)
2. **Unpinned `auto_adjust` in the one price fetch** (`app/data_ingest.py:53`). A
   single library-default flip would silently switch the entire app between
   total-return and raw prices — corrupting every downstream number with no error.
3. **`fillna(0)` on factor z-scores** (`app/screener.py:454`) + **`ffill(5).dropna()`
   on the price matrix** (`app/data_ingest.py:75`). Missing data silently becomes
   "average" or silently truncates history for everyone; both change numbers with
   no flag.
4. **No retry / rate-limit handling; partial fetch failures are invisible**
   (`app/data_ingest.py:58-59`) — compounded by full-universe refetch on any new
   ticker (`app/main.py:151`). At any load, the displayed universe can silently
   shrink.
5. **"Sharpe (1Y)" mislabeled for short-history names** (`app/main.py:652-671`):
   as few as 20 days, and a simple-return numerator over an annualized-vol
   denominator, under a column that claims a 1-year figure.

---

## Draft 30-second spoken answers (grounded only in the audit; weaknesses conceded)

**"How do you handle survivorship bias?"**
Honestly, I don't eliminate it — the universe is today's index membership, so
companies that were delisted before now are simply absent, and that inflates every
backward-looking number I show. In the walk-forward backtest I at least made the
random benchmark draw from that *same* survivor set, so relative comparisons cancel
the plain version of the bias. But I treat a pass as weak evidence and say so out
loud; the real fix is a point-in-time universe from a paid source like FMP, which
is already on the roadmap.

**"Splits and dividends?"**
Both are handled through yfinance's adjusted closes — the adjustment folds splits
and dividends into one total-return price series, so returns, Sharpe, and
correlation are all consistent on that basis. The honest caveat is that production
relies on yfinance's *default* adjustment rather than pinning it explicitly, so a
library default change could silently switch me to raw prices. The backtest pins it
deliberately; the app should too, and doesn't yet.

**"What if yfinance returns bad data?"**
Today, not enough. A per-ticker fetch failure is caught and the name is silently
dropped — there's no retry and no rate-limit backoff — and gaps are forward-filled
five days before the whole date row is dropped. The worst part is that a partial
failure just produces a quietly smaller universe with no indicator to the user, and
missing factor values get filled to the average rather than flagged. Those are
known gaps, not surprises.

**"Why trust a backtest built on free data?"**
You shouldn't trust it as proof — trust it as disciplined disconfirmation. It's
walk-forward and point-in-time, with sealed pass/fail thresholds written before it
ran and a thousand-portfolio null from the same universe, so it can genuinely fail.
What free data costs me is survivorship bias and real point-in-time fundamentals —
I dropped the value factor entirely rather than look ahead. So when Growth-95 came
back a pass, I flagged it as a likely survivorship artifact rather than a win.

**"What would break first at scale?"**
The data layer, without question. Everything lazy-loads on first request with no
retry, and any new ticker triggers a full-universe refetch, so concurrent cold
starts would hammer yfinance straight into rate limits. On the free Render tier the
process sleeps and loses both the in-memory store and the cache, so every wake is a
full refetch. Redis is wired as a fallback, but the app will happily run on a
per-process dict that vanishes on restart — fine for a demo, the first thing I'd
harden for real traffic.
