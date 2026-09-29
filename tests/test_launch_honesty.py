"""
Launch honesty remediation (docs/LAUNCH_HONESTY_DECISIONS.md).

1. No random-number analytics anywhere in the frontend (grep, explicit allowlist).
2. Honest failure states replace every fabricated fallback.
3. Price history keeps each ticker's own history (no cross-ticker truncation),
   so the 2022 stress window survives.
4. Committed market snapshot: loads at startup, feeds the first screen with no
   network, labels its as-of date; missing tickers are merged, not refetched wholesale.
5. Displayed Sharpe subtracts rf; the ranker's `sharpe` input is untouched.
6. Ask Quantex without a token: one clean line, no raw JSON, no key prompt.
7. β chip band states are color-neutral.
"""
import gzip
import json
import os
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.main as main
import app.screener as screener
from app.config import RISK_FREE_RATE
from app.correlation_engine import MIN_STRESS_OBS, compute_stress_correlation

HERE = os.path.dirname(__file__)
FRONTEND = os.path.abspath(os.path.join(HERE, "..", "static", "quantex.html"))
SRC = open(FRONTEND, encoding="utf-8").read()
NODE = shutil.which("node")

# LEGITIMATE (non-analytic) uses of Math.random, by enclosing function name.
# Empty: the only legitimate use (a transaction id) lived in dead code that was
# removed with the random-price paper-trade path.
MATH_RANDOM_ALLOWLIST: dict[str, str] = {}


def _enclosing_function(src, pos):
    m = None
    for m in re.finditer(r"function\s+([A-Za-z_$][\w$]*)\s*\(", src[:pos]):
        pass
    return m.group(1) if m else None


# ── 1/2: no random analytics, no fabricated fallbacks ─────────────────────────
def test_no_math_random_outside_allowlist():
    hits = [(m.start(), _enclosing_function(SRC, m.start())) for m in re.finditer(r"Math\.random", SRC)]
    offenders = [(SRC.count("\n", 0, p) + 1, fn) for p, fn in hits if fn not in MATH_RANDOM_ALLOWLIST]
    assert offenders == [], f"Math.random in analytics code: {offenders}"


def test_fabricating_helpers_removed():
    for name in ("function getCorr(", "function calcStats(", "function optimizeLocal(",
                 "calcStats(", "getCorr(", "optimizeLocal(", "const executeTrade=", "const getPrice=",
                 "const MKT={"):
        assert name not in SRC, name


def test_old_builder_stat_grid_removed():
    assert 'className:"stat-grid"' not in SRC


def test_honest_failure_states_present():
    assert '"Correlations unavailable for the "+regime+" regime: "' in SRC
    assert '"Efficient frontier unavailable: "' in SRC
    assert '"Optimization did not run: "+lastMeta.error+" Your weights were not changed."' in SRC
    assert "SAMPLE DATA: illustrative rows, not live analysis" in SRC
    assert "uvicorn" not in SRC


def test_stress_failure_message_names_cause_and_numbers():
    idx = pd.bdate_range("2024-01-01", periods=13)
    r = pd.DataFrame(np.random.default_rng(0).normal(0, 0.01, (13, 2)), index=idx, columns=["A", "B"])
    with pytest.raises(ValueError) as ex:
        compute_stress_correlation(r, ["A", "B"])
    assert str(ex.value) == ("Not enough market-stress history to compute this: 13 stress days "
                             f"available, {MIN_STRESS_OBS} required.")
    assert "lowering" not in str(ex.value)


@pytest.mark.skipif(not NODE, reason="node not available")
def test_health_facts_never_invent_values():
    def fn(name):
        i = SRC.index("function " + name + "(")
        s = SRC.index("{", SRC.index(")", i))
        d = 0
        for j in range(s, len(SRC)):
            d += SRC[j] == "{"
            d -= SRC[j] == "}"
            if d == 0:
                return SRC[i:j + 1]
    script = fn("healthFacts") + """
      const f=healthFacts(null);const g=healthFacts({status:"insufficient_data",excluded:[{ticker:"X"}]});
      const vals=[...Object.entries(f),...Object.entries(g)].filter(([k])=>!["ok","excluded"].includes(k)).map(([,v])=>v);
      if(vals.some(v=>v!==null)){console.error("non-null",JSON.stringify([f,g]));process.exit(1);}
      console.log("OK");"""
    r = subprocess.run([NODE, "-e", script], capture_output=True, text=True)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr


# ── 3: history length ─────────────────────────────────────────────────────────
def test_fetch_prices_keeps_each_tickers_own_history(monkeypatch):
    import app.data_ingest as di
    full = pd.bdate_range("2021-09-01", periods=1250)
    young = full[400:]                 # 850 rows: passes the 60%-of-5y admission rule

    class T:
        def __init__(self, sym):
            self.sym = sym

        def history(self, start, end):
            ix = young if self.sym == "YOUNG" else full
            px = pd.Series(np.linspace(50, 80, len(ix)), index=ix)
            return pd.DataFrame({"Close": px, "Volume": 1_000_000})

    monkeypatch.setattr(di.yf, "Ticker", T)
    prices = di.fetch_prices(["OLD", "YOUNG"], years=5, include_hedges=False)
    assert prices.index[0] == full[0]                     # not truncated to YOUNG's listing
    assert prices["YOUNG"].first_valid_index() == young[0]
    assert prices["OLD"].notna().all()
    r = di.compute_log_returns(prices)
    assert len(r) == len(full) - 1 and r["OLD"].notna().all()


# ── 4: market snapshot ────────────────────────────────────────────────────────
@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2021-07-01", periods=1250)
    tick = ["AAA", "BBB", "CCC", "DDD", "EEE", "SPY", "QQQ", "TLT", "GLD", "SHY", "UUP", "VXX"]
    steps = rng.normal(0.0003, 0.012, (len(idx), len(tick)))
    steps[250:450, tick.index("QQQ")] -= 0.004             # a real stress stretch
    px = pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=idx, columns=tick)
    fund = [{"ticker": t, "name": t, "sector": "Technology", "industry": "Software",
             "market_cap": 5e10, "price": 100.0, "avg_volume": 5e6, "dividend_yield": 1.0,
             "pe_ratio": 20.0, "forward_pe": 18.0, "pb_ratio": 3.0, "earnings_yield": 0.05,
             "earnings_date": None, "dividend_growth_5y": None, "debt_to_equity": 1.0,
             "revenue_growth": 5.0, "total_assets": 0, "asset_type": "Stock"} for t in tick[:5]]
    doc = {"asof": str(idx[-1].date()), "built": "test",
           "prices": {"dates": [d.strftime("%Y-%m-%d") for d in idx],
                      "columns": {t: [round(float(v), 4) for v in px[t]] for t in tick}},
           "volumes": {t: [1_000_000] * len(idx) for t in tick}, "fundamentals": fund}
    p = tmp_path / "snap.json.gz"
    with gzip.open(p, "wb") as fh:
        fh.write(json.dumps(doc).encode())
    for k in list(main._store):
        monkeypatch.setitem(main._store, k, None)
    monkeypatch.setitem(main._store, "fundamentals", {})
    monkeypatch.setitem(main._store, "unavailable", set())
    monkeypatch.setattr(main, "_SNAPSHOT_PATH", p)
    monkeypatch.setattr(main, "cache_get", lambda *a, **k: None)
    monkeypatch.setattr(main, "cache_set", lambda *a, **k: None)
    monkeypatch.setattr(screener, "load_nasdaq_tickers", lambda: tick[:5])
    monkeypatch.setattr("app.main.load_nasdaq_tickers", lambda: tick[:5], raising=False)
    return doc


def test_snapshot_loads_and_first_screen_needs_no_network(snapshot, monkeypatch):
    assert main._load_snapshot() is True
    assert main._store["asof"] == snapshot["asof"] and main._store["source"] == "snapshot"
    assert int(main._store["stress_mask"].sum()) >= 60

    def no_network(*a, **k):
        raise AssertionError("first screen must not fetch")
    monkeypatch.setattr(screener, "fetch_fundamentals", no_network)
    monkeypatch.setattr(main, "fetch_prices", no_network)
    r = TestClient(main.app).post("/api/screen", json={"goal": "Growth", "risk_score": 60,
                                                       "time_horizon_years": 10, "max_results": 10})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prices_asof"] == snapshot["asof"] and body["data_source"] == "snapshot"
    assert {a["ticker"] for a in body["shortlist"]} <= {"AAA", "BBB", "CCC", "DDD", "EEE"}
    h = TestClient(main.app).get("/health").json()
    assert h["market"]["asof"] == snapshot["asof"] and "SPY" in h["market"]["closes"]


def test_ensure_data_fetches_only_missing_and_keeps_universe(snapshot, monkeypatch):
    main._load_snapshot()
    asked = []

    def fake_fetch(tickers, include_hedges=True, return_volumes=False, **k):
        asked.append(list(tickers))
        # yfinance returns exchange-tz-aware dates; the snapshot stores plain dates
        idx = main._store["prices"].index[-300:].tz_localize("America/New_York")
        p = pd.DataFrame({t: np.linspace(10, 12, len(idx)) for t in tickers if t != "NOPE"}, index=idx)
        v = pd.DataFrame(1, index=idx, columns=p.columns)
        return (p, v) if return_volumes else p
    monkeypatch.setattr(main, "fetch_prices", fake_fetch)
    before = set(main._store["prices"].columns)
    main._ensure_data(["AAA", "NEWT", "NOPE"])
    assert asked == [["NEWT", "NOPE"]]                     # only the missing ones
    assert before | {"NEWT"} <= set(main._store["prices"].columns)
    assert "NOPE" in main._store["unavailable"]
    assert main._store["prices"].index.tz is None
    assert main._store["prices"]["NEWT"].notna().sum() == 300    # merged on the same calendar dates
    main._ensure_data(["NOPE"])                            # not re-fetched every request
    assert asked == [["NEWT", "NOPE"]]


# ── 5: rf-consistent displayed Sharpe; ranker input untouched ─────────────────
def test_display_sharpe_subtracts_rf_and_ranker_sharpe_unchanged():
    assert screener._display_sharpe_rf(np.log(1.20), 0.25) == pytest.approx((0.20 - RISK_FREE_RATE) / 0.25, abs=1e-3)
    assert screener._display_sharpe_rf(None, 0.2) is None and screener._display_sharpe_rf(0.1, 0) is None
    # the ranker's own Sharpe (compute_factor_scores) is still mean*252/vol, no rf
    assert "sharpe = ann_ret / ann_vol.replace(0, np.nan)" in open(screener.__file__).read()


def test_frontend_sharpe_cells_use_display_value():
    assert "const label=sharpeMarketLabel(sh,spyRef,isRef);" in SRC
    assert "const spyRef=spyInRows?displaySharpe(spyInRows):spyRefState;" in SRC
    assert "shRf: item.sharpe_rf ?? null," in SRC
    assert "a.sh.toFixed(2)" not in SRC


# ── 6: Ask Quantex ────────────────────────────────────────────────────────────
def test_ai_status_endpoint(monkeypatch):
    monkeypatch.delenv("DEMO_PASSWORD", raising=False)
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
    assert TestClient(main.app).get("/api/ai/status").json() == {"enabled": False}
    monkeypatch.setenv("REPLICATE_API_TOKEN", "r8_x")
    assert TestClient(main.app).get("/api/ai/status").json() == {"enabled": True}


def test_ask_quantex_frontend_copy():
    assert "The AI explainer isn't enabled on this demo." in SRC
    assert "qx_api_key" not in SRC and "sk-ant-" not in SRC      # dead key prompt removed
    assert "proxyResp.text()" not in SRC                          # raw server payload never shown
    assert "REPLICATE_API_TOKEN" not in SRC


# ── 7: β chip ─────────────────────────────────────────────────────────────────
def test_beta_chip_band_states_neutral():
    assert 'bandLbl.indexOf("In band")===0?"#1D9E75"' not in SRC
    assert 'bandLbl&&e("span",{style:{fontSize:10,fontWeight:"bold",color:"#94a3b8"}},"· "+bandLbl)' in SRC


def test_live_refresh_is_opt_in(monkeypatch):
    started = []

    class FakeThread:
        def __init__(self, target=None, name=None, daemon=None):
            self.name = name

        def start(self):
            started.append(self.name)

    monkeypatch.setattr(main.threading, "Thread", FakeThread)
    monkeypatch.setenv("QX_SNAPSHOT_LOAD", "0")
    monkeypatch.delenv("QX_LIVE_REFRESH", raising=False)
    with TestClient(main.app):
        pass
    assert started == []                                   # default: snapshot only
    monkeypatch.setenv("QX_LIVE_REFRESH", "1")
    with TestClient(main.app):
        pass
    assert started == ["live-refresh"]


# ── Amendment 3 copy pass (final) ─────────────────────────────────────────────
REMOVED_COPY = [
    "consider more conservative profile", "which assets we recommend", "no diversification in crisis",
    "good diversification", "limited diversification", "hedging benefit", "DIVERSIFICATION WARNINGS",
    "Near optimal", "Consider optimizing", "Good if you want", "Good if preserving", "Good for diversification",
    "Good for capturing", "most stable combination", "no single position dominates", "= concentrated",
    "like MSFT, COST, LLY", "Avoid high-beta tech", "are your tools", "Holy Grail", "free lunch",
    "should fall less than the market", "hedge fund managers get paid", "genuine alpha",
    "high-conviction picks", "conflicts with aggressive allocation",
]


def _app_copy():
    """User-facing app copy: the frontend minus CFA curriculum (CourseHub MODULES),
    the hidden Trade Desk (TRADE_DESK_SPEC), and the AI system prompt."""
    s = SRC
    i = s.index("  const MODULES=[")
    s = s[:i] + s[s.index("\n  ];\n", i):]
    i = s.index("function TradeDeck(")
    s = s[:i] + s[s.index("// ── Compare", i):]
    s = re.sub(r'const sysPrompt="[^"]*";', "", s)
    return s


def test_removed_advice_copy_absent():
    main_src = open(main.__file__, encoding="utf-8").read()
    for phrase in REMOVED_COPY:
        assert phrase not in _app_copy(), phrase
        assert phrase.lower() not in main_src.lower(), phrase


def test_no_imperative_or_verdict_words_in_app_copy():
    copy = _app_copy()
    lits = re.findall(r'"([^"\\n]{3,})"', copy)
    pat = re.compile(r"\b(consider(ing)? (adding|trimming|optimizing|a more)|should|well[- ]diversified|poorly"
                     r"|near optimal|good diversification|healthy|unhealthy|risky)\b", re.I)
    hits = [l for l in lits if pat.search(l)
            and "not a recommendation" not in l and "not a forecast" not in l]
    assert hits == [], hits


def test_sharpe_basis_labels():
    assert "Sharpe (realized, past 252 trading days) " in SRC
    assert '"Sharpe (model expectation) "+cs.sh' in SRC
    assert "The Frontier tab's Sharpe is a model expectation instead." in SRC
    assert "Portfolio Health's Sharpe is the realized figure for the past 252 trading days instead." in SRC


# ── Challenge progress reset (one-time) ──────────────────────────────────────
@pytest.mark.skipif(not NODE, reason="node not available")
def test_challenge_progress_reset_once():
    def grab(name):
        i = SRC.index("function " + name + "(")
        s = SRC.index("{", SRC.index(")", i))
        d = 0
        for j in range(s, len(SRC)):
            d += SRC[j] == "{"
            d -= SRC[j] == "}"
            if d == 0:
                return SRC[i:j + 1]
    consts = re.search(r'const CHALLENGES_DATA_VERSION="\d+";', SRC).group(0).replace("const ", "var ")
    code = consts + "\n" + "\n".join(grab(n) for n in ("_lsGet", "_lsSet", "_lsDel", "migrateChallengeProgress")) + r"""
    const mk=init=>{const m=new Map(Object.entries(init));return{getItem:k=>m.has(k)?m.get(k):null,setItem:(k,v)=>m.set(k,String(v)),removeItem:k=>m.delete(k),_m:m};};
    const A=(c,msg)=>{if(!c){console.error("FAIL "+msg);process.exit(1);}};
    // old progress present -> cleared once, notice pending
    global.localStorage=mk({qx_challenges:JSON.stringify(["sharpe_1","low_vol"])});
    A(migrateChallengeProgress()===true,"reset reported");
    A(localStorage.getItem("qx_challenges")===null,"progress cleared");
    A(localStorage.getItem("qx_challenges_reset_notice")==="pending","notice pending");
    // second load: no second reset, new progress kept
    localStorage.setItem("qx_challenges",JSON.stringify(["beta_target"]));
    localStorage.setItem("qx_challenges_reset_notice","seen");
    A(migrateChallengeProgress()===false,"no second reset");
    A(localStorage.getItem("qx_challenges")==='["beta_target"]',"new progress kept");
    // fresh user: nothing to reset, no notice
    global.localStorage=mk({});
    A(migrateChallengeProgress()===false,"fresh: no reset");
    A(localStorage.getItem("qx_challenges_reset_notice")===null,"fresh: no notice");
    console.log("OK");"""
    r = subprocess.run([NODE, "-e", code], capture_output=True, text=True)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr + r.stdout
    assert 'CHALLENGE_RESET_NOTICE="Challenge progress was reset: earlier results were computed on placeholder data."' in SRC


def test_refresh_snapshot_validator_accepts_committed_snapshot():
    """scripts/refresh_snapshot.py is not run here (network); its validator is."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "refresh_snapshot", os.path.join(HERE, "..", "scripts", "refresh_snapshot.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rep = mod.validate(mod.TARGET)
    assert rep["problems"] == [], rep
    assert rep["stress_days"] >= 60 and rep["tickers"] >= 500
    later = mod.validate(mod.TARGET, previous_asof="2999-01-01")
    assert later["problems"] and "older than committed" in later["problems"][0]
