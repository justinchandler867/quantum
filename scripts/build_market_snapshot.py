"""
Build app/data/market_snapshot.json.gz — the committed cold-start snapshot
(same pattern as caps.json: a dated data file loaded at startup).

Contents, all from the SAME code paths the live app uses:
  * prices / volumes: data_ingest.fetch_prices over the screening universe
    (+ reference hedges, SPY, QQQ), 5 years, adjusted closes;
  * fundamentals: screener.fetch_fundamentals over the universe (the slow,
    ~6 minute stage of a cold screen).

The app loads this at startup so the first screen needs no network; a live
refresh runs in the background and replaces it when complete. The UI labels
the data "Prices as of {asof}".

Usage (from backend/, venv active):  python scripts/build_market_snapshot.py
"""
import gzip
import json
import math
import os
import sys
import time
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.config import BETA_BENCHMARK, REFERENCE_HEDGES  # noqa: E402
from app.data_ingest import fetch_prices  # noqa: E402
from app.screener import fetch_fundamentals, load_nasdaq_tickers  # noqa: E402

OUT = os.path.join(ROOT, "app", "data", "market_snapshot.json.gz")


def _clean(v):
    if v is None:
        return None
    if isinstance(v, float):
        return None if math.isnan(v) or math.isinf(v) else v
    return v


def main():
    t0 = time.time()
    universe = load_nasdaq_tickers()
    tickers = sorted(set(universe + REFERENCE_HEDGES + [BETA_BENCHMARK, "QQQ"]))
    prices, volumes = fetch_prices(tickers, include_hedges=True, return_volumes=True)
    t1 = time.time()
    fund = fetch_fundamentals(universe)
    t2 = time.time()
    doc = {
        "asof": str(prices.index[-1].date()),
        "built": str(date.today()),
        "prices": {
            "dates": [d.strftime("%Y-%m-%d") for d in prices.index],
            "columns": {t: [None if v != v else round(float(v), 4) for v in prices[t].values]
                        for t in prices.columns},
        },
        "volumes": {t: [int(v) for v in volumes[t].values] for t in volumes.columns},
        "fundamentals": [{k: _clean(v) for k, v in rec.items()} for rec in fund.to_dict("records")],
    }
    raw = json.dumps(doc, separators=(",", ":")).encode()
    with gzip.open(OUT, "wb", compresslevel=9) as fh:
        fh.write(raw)
    print(f"asof={doc['asof']} prices={prices.shape} fundamentals={len(fund)} "
          f"raw={len(raw)/1e6:.1f}MB gz={os.path.getsize(OUT)/1e6:.1f}MB "
          f"fetch_prices={t1-t0:.0f}s fundamentals={t2-t1:.0f}s")


if __name__ == "__main__":
    main()
