"""Domino Map: build a causal graph from the review state and render it offline.

The renderer is a single self-contained HTML document (vanilla JS + SVG, no CDN),
so it works without an internet connection.
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

# decision -> visual status
STATUS_OF_DECISION = {
    "pending": "pending",
    "accepted": "accepted",
    "investigate": "investigate",
    "rejected": "rejected",
}
DEFAULT_STATUSES = ("existing", "pending", "accepted", "investigate")


def wrap_label(text: str, width: int = 24, max_lines: int = 3) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if len(trial) <= width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: width - 1].rstrip() + "…"
    return lines or [""]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9À-ɏ ]+", "", text.lower()).strip()


def _score(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    ta, tb = set(a.split()), set(b.split())
    jaccard = len(ta & tb) / len(ta | tb) if ta | tb else 0.0
    return max(SequenceMatcher(None, a, b).ratio(), jaccard)


def best_match(label: str, candidates: list[dict], threshold: float) -> dict | None:
    best, best_score = None, 0.0
    for node in candidates:
        score = _score(label, node["label"])
        if score > best_score:
            best, best_score = node, score
    return best if best_score >= threshold else None


# ------------------------------------------------------------------ build ----
def build_graph(a: dict, statuses=DEFAULT_STATUSES) -> dict:
    """Return ``{"nodes": [...], "edges": [...]}`` for the current review state.

    * analyst-identified mechanisms: solid nodes (status ``existing``)
    * AI suggestions awaiting a decision: dashed
    * accepted suggestions: integrated (solid, marked)
    * rejected suggestions: hidden unless ``"rejected"`` is in ``statuses``
    """
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_edges: set[tuple[str, str]] = set()

    def add_node(**kw) -> dict:
        node = {
            "description": "", "evidence": [], "evidence_class": "", "suggestion_id": "",
            "suggestion_title": "", **kw,
        }
        node["lines"] = wrap_label(node["label"])
        nodes.append(node)
        return node

    def add_edge(source: str, target: str, label: str = "", origin="analyst", status="existing", ref=None):
        if source == target or (source, target) in seen_edges:
            return
        seen_edges.add((source, target))
        edges.append({"source": source, "target": target, "label": label,
                      "origin": origin, "status": status, "ref": ref})

    amap = a.get("analyst_map") or {}
    for n in amap.get("nodes", []):
        add_node(id=f"a:{n['id']}", label=n["label"], node_type=n["node_type"],
                 description=n.get("description", ""), origin="analyst", status="existing", ref=n["id"])
    for n in a.get("manual_nodes", []):
        add_node(id=f"m:{n['id']}", label=n["label"], node_type=n["node_type"],
                 description=n.get("description", ""), origin="manual", status="existing", ref=n["id"])

    existing = [n for n in nodes]
    ids = {n["id"] for n in nodes}
    for i, e in enumerate(amap.get("edges", [])):
        s, t = f"a:{e['source']}", f"a:{e['target']}"
        if s in ids and t in ids:
            add_edge(s, t, e.get("label", ""), "analyst", "existing", ("analyst", i))

    for s in a.get("suggestions", []):
        status = STATUS_OF_DECISION.get(s.get("decision", "pending"), "pending")
        if status not in statuses:
            continue
        step_ids: list[str] = []
        for k, step in enumerate(s["causal_chain"]):
            reuse = best_match(step["label"], existing, 0.85)
            if reuse is not None:
                step_ids.append(reuse["id"])
                continue
            node = add_node(
                id=f"s:{s['id']}:{k}", label=step["label"], node_type=step["node_type"],
                description=s.get("explanation", ""), origin="ai", status=status, ref=s["id"],
                evidence=s.get("evidence", []), evidence_class=s.get("evidence_class", ""),
                suggestion_id=s["id"], suggestion_title=s.get("title", ""),
            )
            step_ids.append(node["id"])
        for left, right in zip(step_ids, step_ids[1:]):
            add_edge(left, right, "", "ai", status, s["id"])
        anchor_label = s.get("anchor_element") or s["causal_chain"][0]["label"]
        anchor = best_match(anchor_label, existing, 0.5)
        if anchor is not None and step_ids and anchor["id"] != step_ids[0]:
            add_edge(anchor["id"], step_ids[0], "", "ai", status, s["id"])

    ids = {n["id"] for n in nodes}
    for i, e in enumerate(a.get("manual_edges", [])):
        if e["source"] in ids and e["target"] in ids:
            add_edge(e["source"], e["target"], e.get("label", ""), "manual", "existing", ("manual", i))

    layout(nodes, edges)
    return {"nodes": nodes, "edges": edges}


# ----------------------------------------------------------------- layout ----
NODE_W, NODE_H = 190, 62


def layout(nodes: list[dict], edges: list[dict], col_w: int = 270, row_h: int = 92, margin: int = 28) -> None:
    """Layered left-to-right layout (cycle-safe); writes ``x``/``y`` into nodes."""
    if not nodes:
        return
    ids = [n["id"] for n in nodes]
    adj: dict[str, list[str]] = {i: [] for i in ids}
    for e in edges:
        adj[e["source"]].append(e["target"])

    # Break cycles: drop DFS back-edges for layering purposes only.
    state: dict[str, int] = {}
    back: set[tuple[str, str]] = set()
    for root in ids:
        if state.get(root):
            continue
        stack = [(root, iter(adj[root]))]
        state[root] = 1
        while stack:
            node, it = stack[-1]
            advanced = False
            for nxt in it:
                if state.get(nxt, 0) == 0:
                    state[nxt] = 1
                    stack.append((nxt, iter(adj[nxt])))
                    advanced = True
                    break
                if state[nxt] == 1:
                    back.add((node, nxt))
            if not advanced:
                state[node] = 2
                stack.pop()

    dag = {i: [t for t in adj[i] if (i, t) not in back] for i in ids}
    indeg = {i: 0 for i in ids}
    for i in ids:
        for t in dag[i]:
            indeg[t] += 1
    layer = {i: 0 for i in ids}
    queue = [i for i in ids if indeg[i] == 0]
    while queue:
        u = queue.pop(0)
        for v in dag[u]:
            layer[v] = max(layer[v], layer[u] + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)

    columns: dict[int, list[str]] = {}
    for i in ids:
        columns.setdefault(layer[i], []).append(i)
    parents: dict[str, list[str]] = {i: [] for i in ids}
    for u in ids:
        for v in dag[u]:
            parents[v].append(u)

    position = {i: k for col in columns.values() for k, i in enumerate(col)}
    for _ in range(2):  # barycentre sweeps
        for l in sorted(columns):
            if l == 0:
                continue
            columns[l].sort(
                key=lambda i: (sum(position[p] for p in parents[i]) / len(parents[i])) if parents[i] else position[i]
            )
            for k, i in enumerate(columns[l]):
                position[i] = k

    tallest = max(len(c) for c in columns.values())
    by_id = {n["id"]: n for n in nodes}
    for l, col in columns.items():
        offset = (tallest - len(col)) * row_h / 2
        for k, i in enumerate(col):
            by_id[i]["x"] = margin + l * col_w
            by_id[i]["y"] = int(margin + offset + k * row_h)


# ------------------------------------------------------------------ render ----
def render_html(graph: dict, height: int = 640, storage_key: str = "default") -> str:
    slim_nodes = [
        {k: n[k] for k in ("id", "label", "lines", "node_type", "origin", "status", "description",
                           "evidence", "evidence_class", "suggestion_title", "x", "y")}
        for n in graph["nodes"]
    ]
    slim_edges = [
        {k: e[k] for k in ("source", "target", "label", "origin", "status")} for e in graph["edges"]
    ]
    payload = json.dumps(
        {"nodes": slim_nodes, "edges": slim_edges, "key": storage_key, "w": NODE_W, "h": NODE_H},
        ensure_ascii=False,
    )
    payload = payload.replace("</", "<\\/").replace("<!--", "<\\!--")
    return _TEMPLATE.replace("__DATA__", payload).replace("__HEIGHT__", str(int(height)))


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<style>
:root{--navy:#0B1B33;--blue:#2563FF;--teal:#0E8F87;--violet:#7B5CFA;--mist:#F3F6FB;--line:#D9E1EE;--slate:#43506B}
*{box-sizing:border-box}
html,body{margin:0;height:100%;font-family:"Segoe UI","SF Pro Text","Helvetica Neue",Arial,sans-serif;color:var(--navy);background:transparent}
#app{display:flex;height:__HEIGHT__px;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#fff}
#left{flex:1;min-width:0;display:flex;flex-direction:column}
#legend{display:flex;flex-wrap:wrap;gap:6px 16px;align-items:center;padding:8px 12px;border-bottom:1px solid var(--line);font-size:12px;color:var(--slate);background:#fff}
#legend span.k{display:inline-flex;align-items:center;gap:6px}
#legend i{display:inline-block;width:22px;height:12px;border-radius:3px;border:1.6px solid var(--navy);background:#fff}
#legend button{margin-left:auto;font:inherit;font-size:12px;border:1px solid var(--line);background:var(--mist);color:var(--navy);border-radius:6px;padding:3px 9px;cursor:pointer}
#stage{flex:1;overflow:auto;background-color:#FAFBFE;background-image:radial-gradient(#d9e1ee 1px,transparent 1px);background-size:22px 22px}
#empty{padding:32px;color:var(--slate);font-size:14px;max-width:420px;line-height:1.5}
#panel{width:310px;border-left:1px solid var(--line);padding:16px;overflow:auto;background:#fff;font-size:13px;line-height:1.5}
#panel h3{margin:0 0 8px;font-family:"Iowan Old Style","Palatino Linotype",Georgia,serif;font-size:18px;line-height:1.25}
#panel .hint{color:var(--slate)}
#panel .tag{display:inline-block;font-size:11px;padding:2px 8px;border-radius:99px;margin:0 6px 8px 0;border:1px solid var(--line);color:var(--slate)}
#panel .tag.ai{border:1.5px dashed var(--violet);color:var(--violet)}
#panel .tag.ok{border:1.5px solid var(--violet);background:var(--violet);color:#fff}
#panel h4{margin:14px 0 4px;font-size:12px;color:var(--slate);font-weight:600}
#panel blockquote{margin:6px 0;padding:6px 10px;border-left:3px solid var(--teal);background:var(--mist);font-size:12.5px}
#panel blockquote small{display:block;color:var(--slate);margin-bottom:2px}
#panel ul{margin:4px 0;padding-left:18px}
.edge{fill:none;stroke:#6B7A99;stroke-width:1.7}
.edge.ai{stroke:var(--violet);stroke-dasharray:6 5}
.edge.ai.accepted{stroke-dasharray:none;stroke-width:2.2}
.edge.rejected{opacity:.35}
.elabel{font-size:11px;fill:var(--slate);paint-order:stroke;stroke:#FAFBFE;stroke-width:4px;text-anchor:middle}
.node{cursor:grab}
.node rect.box{stroke-width:1.8}
.node text{font-size:12.5px;pointer-events:none}
.node.sel rect.box{stroke:var(--blue);stroke-width:3.2}
.t-Initial{fill:var(--navy);stroke:var(--navy)}
.t-Initial text{fill:#fff}
.t-Intermediate{fill:#fff;stroke:var(--blue)}
.t-Impact{fill:#E2F5F3;stroke:var(--teal)}
.node.ai rect.box{fill:#F3EFFF;stroke:var(--violet);stroke-dasharray:7 5}
.node.ai text{fill:var(--navy)}
.node.ai.accepted rect.box{fill:#EAE4FF;stroke-dasharray:none;stroke-width:2.6}
.node.ai.investigate rect.box{fill:#F6F6FA;stroke:#8E86B8}
.node.ai.rejected{opacity:.45}
.node.ai.rejected rect.box{stroke:#8A93A8;fill:#F1F2F6}
.badge circle{fill:var(--violet)}
.badge text{fill:#fff;font-size:11px;font-weight:700;text-anchor:middle}
.node.ai.investigate .badge circle{fill:#8E86B8}
.node.ai.rejected .badge circle{fill:#8A93A8}
</style></head>
<body>
<div id="app">
  <div id="left">
    <div id="legend">
      <span class="k"><i style="background:#0B1B33"></i>Initial event</span>
      <span class="k"><i style="border-color:#2563FF"></i>Mechanism</span>
      <span class="k"><i style="border-color:#0E8F87;background:#E2F5F3"></i>Company impact</span>
      <span class="k"><i style="border:1.8px dashed #7B5CFA;background:#F3EFFF"></i>AI suggestion (pending)</span>
      <span class="k"><i style="border:2.4px solid #7B5CFA;background:#EAE4FF"></i>Accepted</span>
      <button id="reset" type="button">Reset layout</button>
    </div>
    <div id="stage"></div>
  </div>
  <aside id="panel"><h3>Domino Map</h3><p class="hint">Click a node to read its explanation and evidence. Drag nodes to rearrange them.</p></aside>
</div>
<script>
(function(){
const DATA = __DATA__;
const NS='http://www.w3.org/2000/svg', W=DATA.w, H=DATA.h;
const stage=document.getElementById('stage'), panel=document.getElementById('panel');
const nodes=DATA.nodes.map(n=>Object.assign({x0:n.x,y0:n.y},n)); const byId={}; nodes.forEach(n=>byId[n.id]=n);
const SKEY='pd-pos-'+DATA.key;
try{const saved=JSON.parse(localStorage.getItem(SKEY)||'{}'); nodes.forEach(n=>{if(saved[n.id]){n.x=saved[n.id][0];n.y=saved[n.id][1];}});}catch(e){}
function el(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e;}
function sv(tag,attrs){const e=document.createElementNS(NS,tag);for(const k in attrs)e.setAttribute(k,attrs[k]);return e;}
if(!nodes.length){const m=el('div',null,'');m.id='empty';m.textContent='The map is empty. Run "Analyze Potential Gaps" to extract your causal structure and see the suggestions, or add nodes below.';stage.appendChild(m);document.getElementById('reset').style.display='none';return;}
const svg=sv('svg',{});stage.appendChild(svg);
const defs=sv('defs',{});
[['g','#6B7A99'],['v','#7B5CFA']].forEach(([id,c])=>{const m=sv('marker',{id:'arr-'+id,viewBox:'0 0 10 10',refX:'9',refY:'5',markerWidth:'8',markerHeight:'8',orient:'auto-start-reverse'});m.appendChild(sv('path',{d:'M0,0 L10,5 L0,10 z',fill:c}));defs.appendChild(m);});
svg.appendChild(defs);
const eLayer=sv('g',{}), nLayer=sv('g',{}); svg.appendChild(eLayer); svg.appendChild(nLayer);
function size(){let mx=0,my=0;nodes.forEach(n=>{mx=Math.max(mx,n.x+W);my=Math.max(my,n.y+H);});const w=Math.max(mx+60,stage.clientWidth-4),h=Math.max(my+60,stage.clientHeight-4);svg.setAttribute('width',w);svg.setAttribute('height',h);}
const edgeEls=DATA.edges.map(e=>{const cls='edge'+(e.origin==='ai'?' ai '+e.status:'')+(e.status==='rejected'?' rejected':'');
  const p=sv('path',{class:cls,'marker-end':'url(#arr-'+(e.origin==='ai'?'v':'g')+')'});eLayer.appendChild(p);
  let t=null;if(e.label){t=sv('text',{class:'elabel'});t.textContent=e.label;eLayer.appendChild(t);} return {e,p,t};});
function drawEdges(){edgeEls.forEach(({e,p,t})=>{const s=byId[e.source],d=byId[e.target];if(!s||!d)return;
  const x1=s.x+W,y1=s.y+H/2,x2=d.x,y2=d.y+H/2,dx=Math.max(50,Math.abs(x2-x1)/2);
  p.setAttribute('d','M'+x1+','+y1+' C'+(x1+dx)+','+y1+' '+(x2-dx)+','+y2+' '+x2+','+y2);
  if(t){t.setAttribute('x',(x1+x2)/2);t.setAttribute('y',(y1+y2)/2-5);}});}
const typeClass={'Initial Event':'t-Initial','Intermediate Mechanism':'t-Intermediate','Company Impact':'t-Impact'};
let selected=null;
const nodeEls=nodes.map(n=>{
  const g=sv('g',{class:'node '+(typeClass[n.node_type]||'t-Intermediate')+(n.origin==='ai'?' ai '+n.status:'')});
  g.appendChild(sv('rect',{class:'box',width:W,height:H,rx:10,ry:10}));
  const lh=15,top=(H-n.lines.length*lh)/2+11;
  n.lines.forEach((ln,i)=>{const t=sv('text',{x:W/2,y:top+i*lh,'text-anchor':'middle'});t.textContent=ln;g.appendChild(t);});
  if(n.origin==='ai'){const b=sv('g',{class:'badge'});b.appendChild(sv('circle',{cx:W-4,cy:4,r:9}));const bt=sv('text',{x:W-4,y:8});bt.textContent=n.status==='accepted'?'✓':(n.status==='rejected'?'×':(n.status==='investigate'?'?':'AI'));if(bt.textContent==='AI')bt.setAttribute('style','font-size:8px');b.appendChild(bt);g.appendChild(b);}
  nLayer.appendChild(g);
  const title=sv('title',{});title.textContent=n.label;g.appendChild(title);
  attach(g,n);return g;});
function place(){nodes.forEach((n,i)=>nodeEls[i].setAttribute('transform','translate('+n.x+','+n.y+')'));drawEdges();size();}
function persist(){try{const o={};nodes.forEach(n=>{if(n.x!==n.x0||n.y!==n.y0)o[n.id]=[n.x,n.y];});localStorage.setItem(SKEY,JSON.stringify(o));}catch(e){}}
function attach(g,n){let drag=null;
  g.addEventListener('pointerdown',ev=>{g.setPointerCapture(ev.pointerId);const r=svg.getBoundingClientRect();drag={dx:ev.clientX-r.left-n.x,dy:ev.clientY-r.top-n.y,sx:ev.clientX,sy:ev.clientY,moved:false};g.style.cursor='grabbing';});
  g.addEventListener('pointermove',ev=>{if(!drag)return;if(Math.abs(ev.clientX-drag.sx)+Math.abs(ev.clientY-drag.sy)>4)drag.moved=true;if(!drag.moved)return;const r=svg.getBoundingClientRect();n.x=Math.max(0,ev.clientX-r.left-drag.dx);n.y=Math.max(0,ev.clientY-r.top-drag.dy);place();});
  g.addEventListener('pointerup',ev=>{if(!drag)return;g.style.cursor='grab';if(drag.moved)persist();else select(n);drag=null;});}
function related(id,dir){return DATA.edges.filter(e=>e[dir==='in'?'target':'source']===id).map(e=>byId[e[dir==='in'?'source':'target']]).filter(Boolean);}
function select(n){selected=n;nodeEls.forEach((g,i)=>g.classList.toggle('sel',nodes[i]===n));
  panel.innerHTML='';panel.appendChild(el('h3',null,n.label));
  panel.appendChild(el('span','tag',n.node_type));
  if(n.origin==='ai'){const lab={pending:'AI suggestion – awaiting your decision',accepted:'AI suggestion – accepted',investigate:'AI suggestion – needs further investigation',rejected:'AI suggestion – rejected'}[n.status];panel.appendChild(el('span','tag '+(n.status==='accepted'?'ok':'ai'),lab));}
  else panel.appendChild(el('span','tag',n.origin==='manual'?'Added by analyst':'Identified by analyst'));
  if(n.suggestion_title){panel.appendChild(el('h4',null,'Part of suggestion'));panel.appendChild(el('div',null,n.suggestion_title));}
  if(n.evidence_class){panel.appendChild(el('h4',null,'Evidence status'));panel.appendChild(el('div',null,n.evidence_class+(n.evidence_class==='Insufficient Evidence'?' – unverified hypothesis, needs further investigation':'')));}
  if(n.description){panel.appendChild(el('h4',null,n.origin==='ai'?'Causal explanation':'Description'));panel.appendChild(el('div',null,n.description));}
  const ins=related(n.id,'in'),outs=related(n.id,'out');
  if(ins.length){panel.appendChild(el('h4',null,'Caused by'));const u=el('ul');ins.forEach(x=>u.appendChild(el('li',null,x.label)));panel.appendChild(u);}
  if(outs.length){panel.appendChild(el('h4',null,'Leads to'));const u=el('ul');outs.forEach(x=>u.appendChild(el('li',null,x.label)));panel.appendChild(u);}
  if(n.evidence&&n.evidence.length){panel.appendChild(el('h4',null,'Supporting evidence'));n.evidence.forEach(ev=>{const q=el('blockquote');q.appendChild(el('small',null,ev.filename+(ev.page!=null?' · page '+ev.page:'')));q.appendChild(document.createTextNode(ev.excerpt));panel.appendChild(q);});}
}
document.getElementById('reset').addEventListener('click',()=>{try{localStorage.removeItem(SKEY);}catch(e){}nodes.forEach(n=>{n.x=n.x0;n.y=n.y0;});place();});
place();
})();
</script></body></html>"""
