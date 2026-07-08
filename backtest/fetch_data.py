"""
One-time (resumable) historical data fetch for the fit-score walk-forward
validation (VALIDATION_SPEC.md §9). NOT imported by app/ — backtest-only.

Fetches full price + dividend + split history for the universe via yfinance
Ticker.history() (yf.download is broken per CLAUDE.md). Caches one pickle per
ticker under backtest/data/raw/ so the fetch is resumable and every freeze-date
computation reads the cache — deterministic, re-runnable, Render-independent.

auto_adjust=False so we keep BOTH raw "Close" (for point-in-time yield
denominator, §L2) and "Adj Close" (split+dividend adjusted, for total returns,
§5.5), plus the "Dividends" action series.
"""
import json
import os
import sys
import time
import pickle

import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
os.makedirs(RAW, exist_ok=True)


def load_tickers():
    uni = json.load(open(os.path.join(HERE, "..", "app", "data", "universe.json")))
    t = set()
    for v in uni.values():
        t.update(v)
    t.add("SPY")  # benchmark
    return sorted(t)


def fetch_one(ticker):
    # yfinance uses '-' not '.' for share classes (BRK.B -> BRK-B)
    yt = ticker.replace(".", "-")
    h = yf.Ticker(yt).history(period="max", auto_adjust=False, actions=True)
    return h


def main():
    tickers = load_tickers()
    ok, fail, skip = [], [], []
    log = open(os.path.join(HERE, "data", "fetch.log"), "w")

    def say(m):
        log.write(m + "\n"); log.flush()
        print(m, flush=True)

    say(f"START fetch: {len(tickers)} tickers -> {RAW}")
    for i, tk in enumerate(tickers):
        path = os.path.join(RAW, tk.replace("/", "_") + ".pkl")
        if os.path.exists(path):
            skip.append(tk)
            continue
        try:
            h = fetch_one(tk)
            if h is None or len(h) == 0:
                fail.append((tk, "empty"))
                say(f"[{i+1}/{len(tickers)}] {tk}: EMPTY")
            else:
                with open(path, "wb") as f:
                    pickle.dump(h, f)
                ok.append(tk)
                if (i + 1) % 25 == 0:
                    say(f"[{i+1}/{len(tickers)}] ok so far={len(ok)} fail={len(fail)} (last {tk}: {len(h)} rows)")
            time.sleep(0.20)  # be polite to Yahoo
        except Exception as e:
            fail.append((tk, str(e)[:120]))
            say(f"[{i+1}/{len(tickers)}] {tk}: FAIL {type(e).__name__} {str(e)[:100]}")
            time.sleep(0.5)

    say(f"DONE: fetched={len(ok)} skipped(cached)={len(skip)} failed={len(fail)}")
    if fail:
        say("FAILURES: " + json.dumps(fail))
    json.dump({"ok": ok, "skip": skip, "fail": fail},
              open(os.path.join(HERE, "data", "fetch_summary.json"), "w"), indent=2)
    log.close()


if __name__ == "__main__":
    main()
