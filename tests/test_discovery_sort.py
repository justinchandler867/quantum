"""
Regression: the Discovery table's default view must rank SCREEN MATCH descending
(best match on top). A prior inline comparator inverted the numeric direction, so
a fresh non-Balanced load rendered the WORST matches first under a "↓" arrow.

Executes the shipped FALLBACK_ASSETS + fitScore + cmpRows under node and asserts
the real rendered top-of-table, plus the comparator direction semantics.
"""
import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(__file__)
FRONTEND = os.path.abspath(os.path.join(HERE, "..", "static", "quantex.html"))
NODE = shutil.which("node")

NODE_SCRIPT = r"""
const fs = require("fs");
const src = fs.readFileSync(process.argv[1], "utf8");
function balanced(open, close, startTok){
  const i=src.indexOf(startTok); const s=src.indexOf(open,i); let d=0,j=s;
  for(;j<src.length;j++){ if(src[j]===open)d++; else if(src[j]===close){d--; if(d===0)break;} }
  return src.slice(s,j+1);
}
function fn(name){
  const i=src.indexOf("function "+name); const s=src.indexOf("{",i); let d=0,j=s;
  for(;j<src.length;j++){ if(src[j]==="{")d++; else if(src[j]==="}"){d--; if(d===0)break;} }
  return src.slice(i,j+1);
}
eval("var FALLBACK_ASSETS = " + balanced("[","]","const FALLBACK_ASSETS = [") + ";");
eval(fn("fitScore"));
eval(fn("cmpRows"));

function pipeline(profile, sk, sd){
  return FALLBACK_ASSETS.map(a => {
    let fit = a.fit || fitScore(a, profile);
    return Object.assign({}, a, {fit});
  }).sort((a,b)=>cmpRows(a,b,sk,sd));
}
function assert(c,m){ if(!c){ console.error("FAIL: "+m); process.exit(1);} }

const growth = {goal:"Growth", rb:4, hy:5, optElig:false, futElig:false};
const fits = FALLBACK_ASSETS.map(a=>a.fit||fitScore(a,growth));
const max = Math.max(...fits), min = Math.min(...fits);
assert(max !== min, "test data has no fit spread");

// DEFAULT non-Balanced load: sk='fit', sd=-1 (header '↓') must be DESCENDING.
const def = pipeline(growth, "fit", -1);
assert(def[0].fit === max, "default top row fit " + def[0].fit + " != max " + max + " (ascending defect)");
assert(def[def.length-1].fit === min, "default bottom row != min");

// Toggle to '↑' (sd=+1) must be ASCENDING.
const asc = pipeline(growth, "fit", 1);
assert(asc[0].fit === min, "ascending toggle top != min");

// cmpRows direction unit checks (numeric + string).
assert(cmpRows({x:1},{x:9},"x",-1) > 0, "sd=-1 numeric not descending");   // 1 after 9
assert(cmpRows({x:1},{x:9},"x", 1) < 0, "sd=+1 numeric not ascending");    // 1 before 9
assert(cmpRows({id:"AAPL"},{id:"MSFT"},"id", 1) < 0, "sd=+1 string not A->Z"); // AAPL before MSFT
assert(cmpRows({id:"AAPL"},{id:"MSFT"},"id",-1) > 0, "sd=-1 string not Z->A");

console.log("OK max="+max+" min="+min+" default_top="+def[0].id+":"+def[0].fit);
"""


@pytest.mark.skipif(not NODE, reason="node not available")
def test_default_discovery_ranks_screen_match_descending():
    r = subprocess.run([NODE, "-e", NODE_SCRIPT, FRONTEND], capture_output=True, text=True)
    assert r.returncode == 0, "node failed:\n" + r.stdout + r.stderr
    assert "OK" in r.stdout, r.stdout


def test_frontend_uses_named_cmprows():
    """Guard the refactor: the row sort must go through cmpRows, not an inline
    comparator that could silently re-invert."""
    src = open(FRONTEND, encoding="utf-8").read()
    assert "function cmpRows(" in src
    assert ".sort((a,b)=>cmpRows(a,b,sk,sd))" in src
