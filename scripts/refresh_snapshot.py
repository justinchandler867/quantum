"""
Refresh the committed market snapshot (app/data/market_snapshot.json.gz).

Rebuilds it with scripts/build_market_snapshot.py into a temporary file,
validates the result, and only then replaces the committed file. The running
app is not touched; deploy by committing the new file.

Usage (from backend/, venv active; needs network, ~8-10 minutes):
    python scripts/refresh_snapshot.py            # build, validate, replace
    python scripts/refresh_snapshot.py --check    # validate the committed file only

Validation (the new file is rejected, and the committed one kept, if any fails):
  * at least 500 tickers with prices and 500 fundamentals rows;
  * stress days (QQQ, -15% from rolling 252d high) >= MIN_STRESS_DAYS (60),
    so Correlations / Frontier / Optimizer / Stress Test can compute;
  * as-of date not older than the committed snapshot's.
After a successful refresh: run the test suite, then
    git add app/data/market_snapshot.json.gz && git commit -m "snapshot: prices as of <asof>"
"""
import argparse
import gzip
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import pandas as pd  # noqa: E402

from app.config import MIN_STRESS_DAYS  # noqa: E402
from app.data_ingest import compute_log_returns, identify_stress_windows  # noqa: E402

TARGET = os.path.join(ROOT, "app", "data", "market_snapshot.json.gz")


def validate(path: str, previous_asof: str | None = None) -> dict:
    with gzip.open(path, "rb") as fh:
        doc = json.loads(fh.read())
    idx = pd.to_datetime(doc["prices"]["dates"])
    prices = pd.DataFrame({t: pd.Series(v, index=idx, dtype="float64")
                           for t, v in doc["prices"]["columns"].items()})
    stress = int(identify_stress_windows(compute_log_returns(prices)).sum())
    report = {"asof": doc["asof"], "tickers": prices.shape[1], "days": prices.shape[0],
              "fundamentals": len(doc.get("fundamentals") or []), "stress_days": stress}
    problems = []
    if report["tickers"] < 500:
        problems.append(f"only {report['tickers']} tickers")
    if report["fundamentals"] < 500:
        problems.append(f"only {report['fundamentals']} fundamentals rows")
    if stress < MIN_STRESS_DAYS:
        problems.append(f"{stress} stress days < {MIN_STRESS_DAYS}")
    if previous_asof and doc["asof"] < previous_asof:
        problems.append(f"as-of {doc['asof']} older than committed {previous_asof}")
    report["problems"] = problems
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="validate the committed snapshot only")
    args = ap.parse_args()
    current = validate(TARGET) if os.path.exists(TARGET) else None
    if args.check:
        print(json.dumps(current, indent=1))
        sys.exit(1 if not current or current["problems"] else 0)
    from build_market_snapshot import main as build
    fd, tmp = tempfile.mkstemp(suffix=".json.gz", dir=os.path.dirname(TARGET))
    os.close(fd)
    try:
        build(tmp)
        report = validate(tmp, current["asof"] if current else None)
        print(json.dumps(report, indent=1))
        if report["problems"]:
            print("REJECTED — committed snapshot unchanged.")
            sys.exit(1)
        os.replace(tmp, TARGET)
        print(f"Replaced {os.path.relpath(TARGET, ROOT)} (prices as of {report['asof']}). "
              "Run the test suite, then commit the file.")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    main()
