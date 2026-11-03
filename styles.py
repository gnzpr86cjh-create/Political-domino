"""Visual identity: navy / electric blue / teal / violet on a cool light neutral.

System font stacks only, so the app looks the same offline.
"""

CSS = """
<style>
:root{
  --pd-navy:#0B1B33; --pd-blue:#2563FF; --pd-teal:#0E8F87; --pd-violet:#7B5CFA;
  --pd-mist:#F3F6FB; --pd-line:#D9E1EE; --pd-slate:#43506B; --pd-amber:#A03C00;
}
html, body, [class*="css"], .stApp { font-family:"Segoe UI","SF Pro Text","Helvetica Neue",Arial,sans-serif; }
h1, h2, h3, .pd-serif { font-family:"Iowan Old Style","Palatino Linotype","Book Antiqua",Georgia,serif !important; color:var(--pd-navy); letter-spacing:-0.01em; }
.block-container { padding-top:2.2rem; max-width:1180px; }
.stButton > button, .stDownloadButton > button { border-radius:8px; font-weight:600; }

/* brand block */
.pd-brand { background:var(--pd-navy); color:#fff; border-radius:12px; padding:16px 16px 14px; margin-bottom:10px; }
.pd-brand .name { font-family:"Iowan Old Style","Palatino Linotype",Georgia,serif; font-size:21px; line-height:1.1; }
.pd-brand .tag { font-size:11.5px; color:#B9C6E4; margin-top:6px; line-height:1.35; }
.pd-tiles { display:flex; align-items:flex-end; gap:5px; height:34px; margin-bottom:10px; }
.pd-tiles i { display:block; width:11px; height:26px; border-radius:2.5px; background:#fff; transform-origin:bottom left; }
.pd-tiles i:nth-child(1){ background:var(--pd-blue); }
.pd-tiles i:nth-child(2){ transform:rotate(9deg);  background:#4F86FF; }
.pd-tiles i:nth-child(3){ transform:rotate(21deg); background:#2FB7AD; }
.pd-tiles i:nth-child(4){ transform:rotate(38deg); background:var(--pd-violet); margin-left:3px; }
.pd-tiles i:nth-child(5){ transform:rotate(62deg); background:transparent; border:1.5px dashed #9B86FF; margin-left:8px; }

/* page header */
.pd-head { border-bottom:2px solid var(--pd-navy); padding-bottom:10px; margin-bottom:18px; }
.pd-head h1 { margin:0; font-size:2.05rem; }
.pd-head p { margin:4px 0 0; color:var(--pd-slate); font-size:0.98rem; max-width:70ch; }

/* workflow strip */
.pd-flow { display:flex; flex-wrap:wrap; gap:8px; margin:0 0 18px; }
.pd-flow span { font-size:12.5px; padding:5px 11px; border-radius:6px; border:1px solid var(--pd-line); background:#fff; color:var(--pd-slate); }
.pd-flow span.done { border-color:var(--pd-teal); color:var(--pd-teal); font-weight:600; }

/* badges */
.pd-badge { display:inline-block; font-size:12px; font-weight:600; padding:3px 10px; border-radius:99px; margin-right:6px; border:1.5px solid; }
.pd-b-direct { color:#fff; background:var(--pd-teal); border-color:var(--pd-teal); }
.pd-b-inferred { color:var(--pd-violet); background:#F3EFFF; border-color:var(--pd-violet); }
.pd-b-insufficient { color:var(--pd-amber); background:#FFF6EE; border-color:var(--pd-amber); border-style:dashed; }
.pd-b-neutral { color:var(--pd-slate); background:#fff; border-color:var(--pd-line); font-weight:500; }
.pd-b-accepted { color:#fff; background:var(--pd-violet); border-color:var(--pd-violet); }
.pd-b-investigate { color:var(--pd-navy); background:#fff; border-color:var(--pd-navy); border-style:dashed; }
.pd-b-rejected { color:#6B7385; background:#F1F2F6; border-color:#C4C9D6; }
.pd-b-pending { color:var(--pd-blue); background:#EEF3FF; border-color:var(--pd-blue); }

/* suggestion content */
.pd-title { font-family:"Iowan Old Style","Palatino Linotype",Georgia,serif; font-size:1.28rem; color:var(--pd-navy); margin:2px 0 8px; line-height:1.25; }
.pd-label { font-size:12.5px; font-weight:700; color:var(--pd-slate); margin:14px 0 3px; }
.pd-chain { display:flex; flex-wrap:wrap; align-items:center; gap:6px 4px; margin:4px 0 2px; }
.pd-chip { font-size:13px; padding:5px 10px; border-radius:6px; border:1.6px dashed var(--pd-violet); background:#F3EFFF; color:var(--pd-navy); }
.pd-chip.impact { border-color:var(--pd-teal); background:#E2F5F3; border-style:solid; }
.pd-chip.event { border-color:var(--pd-navy); background:var(--pd-navy); color:#fff; border-style:solid; }
.pd-arrow { color:var(--pd-violet); font-weight:700; }
.pd-quote { border-left:4px solid var(--pd-teal); background:var(--pd-mist); padding:8px 12px; margin:6px 0; border-radius:0 6px 6px 0; }
.pd-quote .src { font-size:12px; color:var(--pd-slate); font-weight:600; margin-bottom:2px; }
.pd-quote .txt { font-size:14px; color:var(--pd-navy); }
.pd-hyp { border:1.5px dashed var(--pd-amber); background:#FFF9F3; color:var(--pd-amber); padding:8px 12px; border-radius:8px; font-size:13.5px; margin:8px 0; }
.pd-note { font-size:12.5px; color:var(--pd-slate); background:var(--pd-mist); padding:6px 10px; border-radius:6px; margin:6px 0; }
.pd-ok { color:var(--pd-teal); font-weight:600; }
.pd-bad { color:#B42318; font-weight:600; }

/* disclaimer */
.pd-disclaimer { margin-top:34px; padding:10px 14px; border-top:1px solid var(--pd-line); color:var(--pd-slate); font-size:12.5px; }
</style>
"""

BRAND = """
<div class="pd-brand">
  <div class="pd-tiles"><i></i><i></i><i></i><i></i><i></i></div>
  <div class="name">Political Domino</div>
  <div class="tag">The analyst researches.<br>Political Domino challenges.<br>The analyst decides.</div>
</div>
"""
