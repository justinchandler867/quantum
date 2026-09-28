"""
Demo prep: visible "SEC Filings" control in the ticker detail modal.

Renders the shipped TickerChart under node with stub React hooks, then asserts
(1) the header carries a labeled "SEC Filings" button, (2) the modal still
renders the one FilingsPanel for this ticker (existing path intact), and
(3) clicking the button scrolls that same FilingsPanel's container into view.
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
function fn(name){
  const i=src.indexOf("function "+name+"("); const s=src.indexOf("{",src.indexOf(")",i)); let d=0,j=s;
  for(;j<src.length;j++){ if(src[j]==="{")d++; else if(src[j]==="}"){d--; if(d===0)break;} }
  return src.slice(i,j+1);
}
function assert(c,m){ if(!c){ console.error("FAIL: "+m); process.exit(1);} }

// Minimal React stand-ins: enough to evaluate one render of the component.
const refs=[];
var useState=init=>[typeof init==="function"?init():init,()=>{}];
var useRef=init=>{const r={current:init};refs.push(r);return r;};
var useEffect=()=>{};
var useMemo=f=>f();
var e=(type,props,...children)=>({type,props:props||{},children:children.flat(Infinity)});
var window={innerWidth:1200,innerHeight:800};
var localStorage={getItem:()=>null,setItem:()=>{}};
var apiFetch=()=>new Promise(()=>{});

eval(fn("FilingsPanel"));
eval(fn("TickerChart"));

const tree=TickerChart({ticker:"AAPL",name:"Apple Inc.",onClose:()=>{}});
function walk(n,visit,parent){ if(!n||typeof n!=="object")return; visit(n,parent); (n.children||[]).forEach(c=>walk(c,visit,n)); }
function text(n){ if(n==null||n===false)return ""; if(typeof n!=="object")return String(n); return (n.children||[]).map(text).join(""); }

const buttons=[], panels=[];
walk(tree,(n,parent)=>{
  if(n.type==="button"&&text(n).includes("SEC Filings"))buttons.push(n);
  if(n.type===FilingsPanel)panels.push({n,parent});
});
assert(buttons.length===1,"expected exactly one 'SEC Filings' button, got "+buttons.length);
assert(panels.length===1,"expected exactly one FilingsPanel, got "+panels.length);
assert(panels[0].n.props.ticker==="AAPL","FilingsPanel ticker prop");
const holder=panels[0].parent;
assert(holder.props.ref&&refs.includes(holder.props.ref),"FilingsPanel container carries a component ref");

// The button sits in the modal header, i.e. before the chart/news in document order.
const order=[]; walk(tree,n=>order.push(n));
const btnIdx=order.indexOf(buttons[0]);
const newsIdx=order.findIndex(n=>n.type==="div"&&text(n)==="RECENT NEWS");
assert(newsIdx>0&&btnIdx<newsIdx,"button precedes the news block (header placement)");

// Click: scrolls that same container into view.
let scrolled=null;
holder.props.ref.current={scrollIntoView:o=>{scrolled=o;}};
buttons[0].props.onClick();
assert(scrolled&&scrolled.block==="start","click scrolls FilingsPanel container into view");
console.log("OK label="+JSON.stringify(text(buttons[0])));
"""


@pytest.mark.skipif(not NODE, reason="node not available")
def test_sec_filings_control_present_and_opens_panel():
    r = subprocess.run([NODE, "-e", NODE_SCRIPT, FRONTEND], capture_output=True, text=True)
    assert r.returncode == 0, "node failed:\n" + r.stdout + r.stderr
    assert "OK" in r.stdout


def test_filings_contract_unchanged():
    """FilingsPanel still posts {ticker} to /api/filings and keeps its explicit load button."""
    html = open(FRONTEND, encoding="utf-8").read()
    assert 'apiFetch("/api/filings",{ticker},180000)' in html
    assert '"Load filings →"' in html
    assert html.count("e(FilingsPanel,") == 1
