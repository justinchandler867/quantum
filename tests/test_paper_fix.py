"""
Paper Trade regression: after a fill, the trade count and the transaction log
reflect it; the tab labels the shared demo account.

Root cause fixed: the tab's txLog / navHistory were only written by the old
local-simulation path (executeTrade, never called; removed in launch-honesty).
The real order path (submitOrder -> /api/paper/order) refreshed cash/positions
but never read the server's transaction log, so Trades stayed 0.
"""
import json
import os
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

import app.main as main
import app.paper_trading as pt

HERE = os.path.dirname(__file__)
FRONTEND = os.path.abspath(os.path.join(HERE, "..", "static", "quantex.html"))
SRC = open(FRONTEND, encoding="utf-8").read()
NODE = shutil.which("node")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("DEMO_PASSWORD", raising=False)
    monkeypatch.setattr(pt, "get_live_price", lambda t: 100.0)
    monkeypatch.setattr(main, "get_batch_prices", lambda ts: {t: 100.0 for t in ts})
    main._accounts.clear()
    yield TestClient(main.app)
    main._accounts.clear()


def _fn(name):
    i = SRC.index("function " + name + "(")
    s = SRC.index("{", SRC.index(")", i))
    d = 0
    for j in range(s, len(SRC)):
        d += SRC[j] == "{"
        d -= SRC[j] == "}"
        if d == 0:
            return SRC[i:j + 1]


def test_fill_is_in_server_log_and_frontend_count_reflects_it(client):
    r = client.post("/api/paper/order", json={"ticker": "MSFT", "side": "buy", "quantity": 10,
                                              "order_type": "market", "tif": "day"})
    assert r.status_code == 200 and r.json()["status"] == "filled"
    tx = client.get("/api/paper/transactions?limit=100").json()
    assert len(tx) == 1 and tx[0]["ticker"] == "MSFT" and tx[0]["status"] == "filled"
    assert client.get("/api/paper/summary").json()["total_trades"] == 1
    nav = client.get("/api/paper/nav-history").json()
    if not NODE:
        pytest.skip("node not available")
    script = "\n".join(_fn(n) for n in ("paperTxFromServer", "paperNavFromServer", "paperTradeCounts")) + """
      const tx=JSON.parse(process.argv[1]), nav=JSON.parse(process.argv[2]);
      const log=paperTxFromServer(tx), c=paperTradeCounts(log), nh=paperNavFromServer(nav,100000);
      console.log(JSON.stringify({log,c,nhLen:nh.length,nh0:nh[0].nav}));"""
    out = subprocess.run([NODE, "-e", script, json.dumps(tx), json.dumps(nav)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    res = json.loads(out.stdout)
    assert res["c"] == {"total": 1, "sells": 0}                     # Trades count reflects the fill
    assert len(res["log"]) == 1                                       # transaction log shows it
    assert res["log"][0]["tk"] == "MSFT" and res["log"][0]["qty"] == 10 and res["log"][0]["price"] == 100.0
    assert res["nhLen"] == len(nav) + 1 and res["nh0"] == 100000


def test_rejected_orders_are_not_counted():
    if not NODE:
        pytest.skip("node not available")
    script = _fn("paperTradeCounts") + """
      const c=paperTradeCounts([{status:"filled",side:"buy"},{status:"rejected",side:"sell"},{status:"filled",side:"sell"}]);
      console.log(JSON.stringify(c));"""
    out = subprocess.run([NODE, "-e", script], capture_output=True, text=True)
    assert json.loads(out.stdout) == {"total": 2, "sells": 1}


def test_order_paths_sync_from_server():
    assert 'if(r.status==="filled")await syncFromServer();' in SRC          # after a fill
    assert "if(n>0)await syncFromServer();" in SRC                            # after Check now fills
    assert "useEffect(()=>{syncFromServer();},[]);" in SRC                    # on load
    assert '"/api/paper/transactions?limit=100"' in SRC and '"/api/paper/nav-history"' in SRC
    assert 'apiFetch("/api/paper/reset",{})' in SRC                           # reset is the shared account's


def test_shared_account_label_and_honest_metrics():
    assert ('const PAPER_SHARED_NOTE="Shared demo account: every visitor sees the same trades. '
            'Resets when the server restarts.";') in SRC
    assert "},PAPER_SHARED_NOTE)," in SRC                                     # rendered on the tab
    assert '["Win Rate"' not in SRC                                           # was sells/fills, not wins
    assert '["Sharpe Ratio","—",""]' in SRC
