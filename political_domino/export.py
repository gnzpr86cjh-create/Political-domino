"""Export the reviewed assessment: DOCX report, JSON archive, CSV decision log."""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from . import config

NAVY = RGBColor(0x0B, 0x1B, 0x33)
BLUE = RGBColor(0x25, 0x63, 0xFF)
VIOLET = RGBColor(0x7B, 0x5C, 0xFA)
SLATE = RGBColor(0x43, 0x50, 0x6B)

CLASS_NOTE = {
    "Directly Supported": "The uploaded sources explicitly support this causal relationship.",
    "Inferred": "The sources support parts of the mechanism; completing it requires interpretation.",
    "Insufficient Evidence": (
        "UNVERIFIED HYPOTHESIS. The uploaded sources do not establish this relationship; "
        "it requires further investigation."
    ),
}


def decisions_of(a: dict, decision: str) -> list[dict]:
    return [s for s in a.get("suggestions", []) if s.get("decision") == decision]


def _cite(ev: dict) -> str:
    page = f", page {ev['page']}" if ev.get("page") is not None else ""
    return f"{ev['filename']}{page}"


# ------------------------------------------------------------------- DOCX ----
def _shade(cell, hex_fill: str) -> None:
    props = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:color"), "auto")
    shading.set(qn("w:fill"), hex_fill)
    props.append(shading)


def _heading(doc: Document, text: str, level: int) -> None:
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        run.font.name = "Georgia"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "Georgia")
        run.font.color.rgb = NAVY if level < 3 else BLUE


def _para(doc: Document, text: str = "", bold: bool = False, italic: bool = False,
          color: RGBColor | None = None, size: float | None = None, style: str | None = None):
    p = doc.add_paragraph(style=style)
    run = p.add_run(text)
    run.bold, run.italic = bold, italic
    if color is not None:
        run.font.color.rgb = color
    if size:
        run.font.size = Pt(size)
    return p


def _labelled(doc: Document, label: str, text: str) -> None:
    p = doc.add_paragraph()
    r = p.add_run(f"{label}: ")
    r.bold = True
    r.font.color.rgb = SLATE
    p.add_run(text or "—")
    p.paragraph_format.space_after = Pt(3)


def _finding(doc: Document, s: dict, number: int, hypothesis: bool) -> None:
    _heading(doc, f"{number}. {s['title']}", 3)
    status = config.DECISIONS[s["decision"]]
    tag = f"{s['evidence_class']}  |  {config.MECHANISM_TYPES.get(s['mechanism_type'], '')}  |  {config.ORDERS.get(s['order'], '')}"
    _para(doc, tag, italic=True, color=VIOLET, size=9.5)
    if s["evidence_class"] == "Insufficient Evidence":
        _para(doc, CLASS_NOTE["Insufficient Evidence"], bold=True, color=RGBColor(0xA0, 0x3C, 0x00), size=9.5)
    elif hypothesis:
        _para(doc, "Marked by the analyst for further investigation; not part of the revised assessment.",
              bold=True, color=RGBColor(0xA0, 0x3C, 0x00), size=9.5)
    _labelled(doc, "Analyst decision", status + (f" ({s['decided_at']})" if s.get("decided_at") else ""))
    if s.get("analyst_note"):
        _labelled(doc, "Analyst note", s["analyst_note"])
    _labelled(doc, "Existing analytical context", s["existing_context"])
    if s.get("analyst_text_ref", {}).get("verified"):
        _labelled(doc, "Passage in analyst's assessment", f"“{s['analyst_text_ref']['text']}”")
    _labelled(doc, "Missing causal connection", s["missing_connection"])
    _labelled(doc, "Causal chain", "  →  ".join(step["label"] for step in s["causal_chain"]))
    _labelled(doc, "Causal explanation", s["explanation"])
    _labelled(doc, "Company-specific implication", s["company_implication"])
    if s.get("countervailing_note"):
        _labelled(doc, "Countervailing mechanisms / alternative reading", s["countervailing_note"])
    if s.get("pestle"):
        _labelled(doc, "PESTLE dimensions", ", ".join(s["pestle"]))
    _labelled(doc, "Evidence classification", f"{s['evidence_class']}. {CLASS_NOTE[s['evidence_class']]}")
    if s.get("evidence_rationale"):
        _labelled(doc, "Rationale (model-assigned)", s["evidence_rationale"])
    if s["evidence"]:
        p = doc.add_paragraph()
        r = p.add_run("Supporting evidence (verified against the uploaded sources):")
        r.bold = True
        r.font.color.rgb = SLATE
        for ev in s["evidence"]:
            q = doc.add_paragraph()
            q.paragraph_format.left_indent = Cm(0.8)
            run = q.add_run(f"“{ev['excerpt']}”")
            run.italic = True
            src = q.add_run(f"  — {_cite(ev)}" + ("  (approximate match)" if ev["match"] == "approximate" else ""))
            src.font.size, src.font.color.rgb = Pt(9), SLATE
    else:
        _para(doc, "No verifiable citation was found in the uploaded sources.", italic=True, color=SLATE)
    for note in s.get("validation_notes", []):
        _para(doc, f"Validation note: {note}", italic=True, color=SLATE, size=9)


def build_docx(a: dict) -> bytes:
    doc = Document()
    section = doc.sections[0]
    section.orientation = WD_ORIENT.PORTRAIT
    section.page_width, section.page_height = Cm(21.0), Cm(29.7)
    for side in ("left_margin", "right_margin"):
        setattr(section, side, Cm(2.2))
    section.top_margin = section.bottom_margin = Cm(2.0)

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")

    accepted = decisions_of(a, "accepted")
    investigate = decisions_of(a, "investigate")
    rejected = decisions_of(a, "rejected")
    pending = decisions_of(a, "pending")

    # -- title block
    title = doc.add_paragraph()
    run = title.add_run("Political Domino")
    run.font.name, run.font.size, run.font.color.rgb = "Georgia", Pt(11), BLUE
    t = doc.add_paragraph()
    tr = t.add_run(a["name"])
    tr.font.name, tr.font.size, tr.bold, tr.font.color.rgb = "Georgia", Pt(24), True, NAVY
    _para(doc, "Reviewed political risk assessment", color=SLATE, size=12)

    table = doc.add_table(rows=0, cols=2)
    table.style = "Table Grid"
    rows = [
        ("Company", a["company"]),
        ("Research question", a["question"]),
        ("Industry", a.get("industry", "")),
        ("Geographic market", a.get("market", "")),
        ("Time horizon", a.get("horizon", "")),
        ("PESTLE focus", ", ".join(a.get("pestle", []))),
        ("Exported", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("Analysis runs", "; ".join(f"{r['created_at']} ({r['model']})" for r in a.get("runs", [])) or "none"),
    ]
    for label, value in rows:
        if not value:
            continue
        cells = table.add_row().cells
        cells[0].width, cells[1].width = Cm(4.2), Cm(12.4)
        cells[0].text, cells[1].text = label, value
        _shade(cells[0], "EEF2FA")
        cells[0].paragraphs[0].runs[0].bold = True

    _para(doc)
    d = _para(doc, config.DISCLAIMER, italic=True, color=SLATE, size=9.5)
    d.paragraph_format.space_after = Pt(10)

    # -- summary
    _heading(doc, "Review summary", 1)
    summary = doc.add_table(rows=1, cols=4)
    summary.style = "Table Grid"
    for cell, (label, n) in zip(
        summary.rows[0].cells,
        [("Accepted", len(accepted)), ("Needs further investigation", len(investigate)),
         ("Rejected", len(rejected)), ("Not yet reviewed", len(pending))],
    ):
        cell.text = f"{n}\n{label}"
        _shade(cell, "EEF2FA")
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.paragraphs[0].runs[0].bold = True
    if pending:
        _para(doc, f"{len(pending)} suggestion(s) have not been reviewed and are not included below.",
              italic=True, color=SLATE, size=9.5)

    # -- sources
    _heading(doc, "Research sources processed", 1)
    if a.get("sources"):
        for s in a["sources"]:
            pages = [p["page"] for p in s["pages"] if p["page"] is not None]
            span = f", {len(pages)} page(s) with text" if pages else ""
            doc.add_paragraph(f"{s['filename']} ({s['file_type'].upper()}{span}, {s['n_chars']:,} characters)",
                              style="List Bullet")
    else:
        _para(doc, "No sources.", italic=True)

    # -- original assessment
    _heading(doc, "Original analyst assessment", 1)
    risks = list(a.get("analyst_map", {}).get("risk_factors", []))
    if (a.get("initial_events") or "").strip():
        risks.insert(0, f"Initial events noted by the analyst: {a['initial_events'].strip()}")
    _heading(doc, "Original risk factors", 2)
    if risks:
        for r in risks:
            doc.add_paragraph(r, style="List Bullet")
    else:
        _para(doc, "No risk factors were extracted. Run the analysis to extract them from the assessment text.",
              italic=True, color=SLATE)
    nodes = a.get("analyst_map", {}).get("nodes", [])
    if nodes:
        _heading(doc, "Causal mechanisms identified by the analyst", 2)
        by_id = {n["id"]: n["label"] for n in nodes}
        for e in a["analyst_map"]["edges"]:
            doc.add_paragraph(f"{by_id[e['source']]}  →  {by_id[e['target']]}", style="List Bullet")
    _heading(doc, "Analyst's text", 2)
    for block in (a.get("assessment_text") or "(empty)").split("\n\n"):
        _para(doc, block.strip())

    # -- revised assessment: accepted only
    _heading(doc, "Revised assessment: additions accepted by the analyst", 1)
    if accepted:
        _para(doc, "Only suggestions the analyst explicitly accepted appear in this section.",
              italic=True, color=SLATE, size=9.5)
        for i, s in enumerate(accepted, 1):
            _finding(doc, s, i, hypothesis=False)
    else:
        _para(doc, "No suggestions have been accepted.", italic=True)

    # -- outstanding hypotheses
    _heading(doc, "Outstanding hypotheses requiring further investigation", 1)
    if investigate:
        _para(doc, "These items are NOT part of the revised assessment. They are unverified hypotheses "
                   "that the analyst marked for follow-up.", italic=True, color=SLATE, size=9.5)
        for i, s in enumerate(investigate, 1):
            _finding(doc, s, i, hypothesis=True)
    else:
        _para(doc, "None.", italic=True)

    # -- decision log
    _heading(doc, "Review log", 1)
    log = doc.add_table(rows=1, cols=4)
    log.style = "Table Grid"
    for cell, text in zip(log.rows[0].cells, ["Suggestion", "Evidence class", "Decision", "Analyst note"]):
        cell.text = text
        _shade(cell, "EEF2FA")
        cell.paragraphs[0].runs[0].bold = True
    for s in a.get("suggestions", []):
        cells = log.add_row().cells
        cells[0].text, cells[1].text = s["title"], s["evidence_class"]
        cells[2].text, cells[3].text = config.DECISIONS[s["decision"]], s.get("analyst_note", "")
    if not a.get("suggestions"):
        _para(doc, "No suggestions were generated.", italic=True)

    footer = section.footer.paragraphs[0]
    fr = footer.add_run(config.DISCLAIMER)
    fr.font.size, fr.italic, fr.font.color.rgb = Pt(8), True, SLATE

    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


# ------------------------------------------------------------ JSON / CSV ----
def build_json(a: dict) -> str:
    """Full archive of the reviewed assessment (includes source text)."""
    return json.dumps(a, ensure_ascii=False, indent=2)


def build_csv(a: dict) -> str:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["id", "title", "decision", "evidence_class", "mechanism_type", "order",
                     "causal_chain", "citations", "analyst_note", "decided_at"])
    for s in a.get("suggestions", []):
        writer.writerow([
            s["id"], s["title"], s["decision"], s["evidence_class"], s["mechanism_type"], s["order"],
            " -> ".join(step["label"] for step in s["causal_chain"]),
            " | ".join(_cite(e) for e in s["evidence"]),
            s.get("analyst_note", ""), s.get("decided_at") or "",
        ])
    return out.getvalue()
