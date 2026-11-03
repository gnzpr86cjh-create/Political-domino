"""Political Domino — AI-assisted political risk analysis & causal gap detection.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import html
import re
import uuid

import streamlit as st
import streamlit.components.v1 as components

from political_domino import analysis, config, export, extraction, graph, llm, retrieval, storage, styles

st.set_page_config(page_title="Political Domino", page_icon="🎲", layout="wide")

NAV = [
    "Dashboard",
    "Assessment",
    "Sources",
    "Analyst Assessment",
    "Gap Detection",
    "Domino Map",
    "Review & Export",
]


# ----------------------------------------------------------------- helpers ----
def esc(value) -> str:
    return html.escape("" if value is None else str(value))


def header(title: str, subtitle: str = "") -> None:
    sub = f"<p>{esc(subtitle)}</p>" if subtitle else ""
    st.markdown(f'<div class="pd-head"><h1>{esc(title)}</h1>{sub}</div>', unsafe_allow_html=True)


def get_assessment() -> dict | None:
    aid = st.session_state.get("aid")
    return storage.load(aid) if aid else None


def make_client() -> llm.OllamaClient:
    return llm.OllamaClient(
        model=st.session_state.get("model_pick") or config.DEFAULT_MODEL,
        num_ctx=int(st.session_state.get("num_ctx", config.DEFAULT_NUM_CTX)),
    )


def go(page: str) -> None:
    st.session_state["_goto"] = page
    st.rerun()


def workflow_strip(a: dict) -> None:
    reviewed = sum(1 for s in a["suggestions"] if s["decision"] != "pending")
    steps = [
        ("Assessment created", True),
        (f"{len(a['sources'])} source(s) uploaded", bool(a["sources"])),
        ("Your analysis added", len((a["assessment_text"] or "").strip()) >= 80),
        ("Gaps analysed", bool(a["runs"])),
        (f"{reviewed} decision(s) made", reviewed > 0),
    ]
    items = "".join(f'<span class="{"done" if ok else ""}">{"✓ " if ok else ""}{esc(t)}</span>' for t, ok in steps)
    st.markdown(f'<div class="pd-flow">{items}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------- sidebar ----
def sidebar() -> str:
    with st.sidebar:
        st.markdown(styles.BRAND, unsafe_allow_html=True)
        st.session_state.setdefault("nav", "Dashboard")
        if "_goto" in st.session_state:
            st.session_state["nav"] = st.session_state.pop("_goto")
        choice = st.radio("Workspace", NAV, key="nav", label_visibility="collapsed")

        a = get_assessment()
        if a:
            st.caption(f"Open: **{a['name']}** · {a['company']}")

        st.markdown("---")
        st.markdown("**Local model (Ollama)**")
        status = llm.OllamaClient().server_status()
        if status.reachable:
            st.markdown('<span class="pd-ok">● Ollama is running</span>', unsafe_allow_html=True)
            if status.models:
                if st.session_state.get("model_pick") not in status.models:
                    st.session_state.pop("model_pick", None)
                default = next(
                    (m for m in status.models if m == config.DEFAULT_MODEL or m == config.DEFAULT_MODEL + ":latest"),
                    status.models[0],
                )
                st.selectbox("Model", status.models, index=status.models.index(default), key="model_pick")
                if not any(m.startswith(config.DEFAULT_MODEL.split(":")[0]) for m in status.models):
                    st.caption(f"Recommended: `ollama pull {config.DEFAULT_MODEL}`")
            else:
                st.warning(f"No models installed. Run `ollama pull {config.DEFAULT_MODEL}` in a terminal.")
        else:
            st.markdown('<span class="pd-bad">● Ollama is not reachable</span>', unsafe_allow_html=True)
            st.caption(
                "Install Ollama from ollama.com and start it. Everything except the AI analysis "
                "still works without it."
            )
        with st.expander("Advanced"):
            st.selectbox(
                "Context window (tokens)",
                config.NUM_CTX_OPTIONS,
                index=config.NUM_CTX_OPTIONS.index(config.DEFAULT_NUM_CTX),
                key="num_ctx",
                help="Larger windows let the model read more source passages but need more RAM "
                "and are slower. Use 8192 on small machines.",
            )
        st.caption(config.DISCLAIMER)
    return choice


# --------------------------------------------------------------- dashboard ----
def page_dashboard() -> None:
    header(
        "Assessments",
        "Research independently, then let Political Domino challenge the completeness of your causal reasoning.",
    )
    left, right = st.columns([2, 3], gap="large")

    with left:
        st.subheader("New assessment")
        with st.form("new_assessment"):
            name = st.text_input("Assessment name *")
            company = st.text_input("Company name *")
            question = st.text_area("Main political risk question or strategic concern *", height=110)
            submitted = st.form_submit_button("Create assessment", type="primary")
        if submitted:
            if not (name.strip() and company.strip() and question.strip()):
                st.error("Assessment name, company name and the main risk question are all required.")
            else:
                a = storage.new_assessment(name, company, question)
                st.session_state["aid"] = a["id"]
                go("Assessment")

    with right:
        st.subheader("Your assessments")
        items = storage.list_assessments()
        if not items:
            st.info("No assessments yet. Create your first one on the left.")
        for item in items:
            with st.container(border=True):
                c1, c2, c3 = st.columns([5, 1.2, 1.2])
                c1.markdown(
                    f"**{esc(item['name'])}**  \n{esc(item['company'])} · "
                    f"{item['n_sources']} source(s) · {item['n_suggestions']} suggestion(s), "
                    f"{item['n_accepted']} accepted  \n<small>Updated {esc(item['updated_at'])}</small>",
                    unsafe_allow_html=True,
                )
                if c2.button("Open", key=f"open_{item['id']}"):
                    st.session_state["aid"] = item["id"]
                    go("Assessment")
                if c3.button("Delete", key=f"del_{item['id']}"):
                    st.session_state["confirm_delete"] = item["id"]
                    st.rerun()
                if st.session_state.get("confirm_delete") == item["id"]:
                    st.warning("Delete this assessment, its sources and all review decisions? This cannot be undone.")
                    d1, d2 = st.columns(2)
                    if d1.button("Yes, delete", key=f"del_yes_{item['id']}", type="primary"):
                        storage.delete(item["id"])
                        if st.session_state.get("aid") == item["id"]:
                            st.session_state.pop("aid", None)
                        st.session_state.pop("confirm_delete", None)
                        st.rerun()
                    if d2.button("Cancel", key=f"del_no_{item['id']}"):
                        st.session_state.pop("confirm_delete", None)
                        st.rerun()


# -------------------------------------------------------------- assessment ----
def page_assessment(a: dict) -> None:
    header("Assessment workspace", "Define what is being assessed. You can edit this at any time.")
    workflow_strip(a)
    k = a["id"]
    with st.form(f"assessment_form_{k}"):
        name = st.text_input("Assessment name *", a["name"], key=f"f_name_{k}")
        company = st.text_input("Company name *", a["company"], key=f"f_company_{k}")
        question = st.text_area("Main political risk question or strategic concern *", a["question"],
                                height=100, key=f"f_question_{k}")
        c1, c2, c3 = st.columns(3)
        industry = c1.text_input("Industry", a["industry"], key=f"f_industry_{k}")
        market = c2.text_input("Geographic market", a["market"], key=f"f_market_{k}")
        horizon = c3.text_input("Time horizon", a["horizon"], placeholder="e.g. 3–5 years", key=f"f_horizon_{k}")
        background = st.text_area("Company background", a["background"], height=90, key=f"f_bg_{k}")
        pestle = st.multiselect("Relevant PESTLE dimensions", config.PESTLE, default=a["pestle"], key=f"f_pestle_{k}")
        events = st.text_area(
            "Initial events or risk factors", a["initial_events"], height=90, key=f"f_events_{k}",
            help="Optional. Events you already know are relevant; they are given to the model as context.",
        )
        saved = st.form_submit_button("Save changes", type="primary")
    if saved:
        if not (name.strip() and company.strip() and question.strip()):
            st.error("Assessment name, company name and the main risk question are required.")
        else:
            a.update(name=name.strip(), company=company.strip(), question=question.strip(),
                     industry=industry.strip(), market=market.strip(), horizon=horizon.strip(),
                     background=background.strip(), pestle=pestle, initial_events=events.strip())
            storage.save(a)
            st.success("Saved.")
    if st.button("Next: upload research sources"):
        go("Sources")


# ----------------------------------------------------------------- sources ----
def page_sources(a: dict) -> None:
    header(
        "Research sources",
        "Upload the documents you collected. They are the only factual evidence the AI may use. "
        "Nothing is fetched from the internet.",
    )
    workflow_strip(a)

    for kind, text in st.session_state.pop("_src_msgs", []):
        (st.success if kind == "success" else st.warning if kind == "warning" else st.error)(text)

    up_n = st.session_state.get("up_n", 0)
    files = st.file_uploader(
        "PDF (text-based), DOCX or TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key=f"src_up_{up_n}",
        help=f"Up to {config.MAX_SOURCES} sources, {config.MAX_FILE_MB} MB each. Scanned PDFs are not supported (no OCR).",
    )
    if st.button("Process uploaded files", type="primary", disabled=not files):
        msgs = []
        for f in files:
            known = {s["filename"] for s in a["sources"]}
            if len(a["sources"]) >= config.MAX_SOURCES and f.name not in known:
                msgs.append(("error", f"**{f.name}** was not added: the limit of {config.MAX_SOURCES} sources is reached."))
                continue
            try:
                extracted = extraction.extract_file(f.name, f.getvalue())
            except extraction.ExtractionError as exc:
                msgs.append(("error", f"**{f.name}** was not added. {exc}"))
                continue
            except Exception as exc:  # noqa: BLE001 - never crash the UI on a bad file
                msgs.append(("error", f"**{f.name}** was not added (unexpected {type(exc).__name__})."))
                continue
            source = storage.add_source(a, extracted)
            source["injection_hits"] = extraction.scan_for_injection(
                " ".join(p["text"] for p in extracted["pages"])
            )
            msgs.append(("success", f"**{f.name}** added as {source['id']} ({extracted['n_chars']:,} characters)."))
        storage.save(a)
        st.session_state["_src_msgs"] = msgs
        st.session_state["up_n"] = up_n + 1
        st.rerun()

    st.subheader(f"Processed sources ({len(a['sources'])}/{config.MAX_SOURCES})")
    if not a["sources"]:
        st.info("No sources yet. Upload at least one document to enable the analysis.")
    for s in a["sources"]:
        with st.container(border=True):
            c1, c2 = st.columns([6, 1])
            pages = [p["page"] for p in s["pages"] if p["page"] is not None]
            n_chunks = len(retrieval.chunk_sources([s]))
            c1.markdown(
                f"**{esc(s['filename'])}**  \n"
                f"<small>{esc(s['id'])} · {s['file_type'].upper()} · "
                f"{(str(len(pages)) + ' page(s) with text') if pages else 'no page numbers'} · "
                f"{s['n_chars']:,} characters · {n_chunks} passages</small>",
                unsafe_allow_html=True,
            )
            if c2.button("Remove", key=f"rm_{s['id']}"):
                storage.remove_source(a, s["id"])
                storage.save(a)
                st.rerun()
            for w in s.get("warnings", []):
                st.caption(f"⚠ {w}")
            if s.get("injection_hits"):
                st.warning(
                    "This document contains text that looks like instructions to an AI model "
                    "(for example: “" + s["injection_hits"][0] + "”). It is treated strictly as source "
                    "material and will not be obeyed."
                )
            with st.expander("Inspect extracted text"):
                labels = [f"Page {p['page']}" if p["page"] is not None else "Full text" for p in s["pages"]]
                pick = st.selectbox("Page", range(len(labels)), format_func=lambda i: labels[i], key=f"pg_{s['id']}")
                st.text_area("Extracted text", s["pages"][pick]["text"], height=240,
                             key=f"ins_{s['id']}_{pick}", disabled=True)
    if a["sources"] and st.button("Next: add your analysis"):
        go("Analyst Assessment")


# ------------------------------------------------------- analyst assessment ----
def page_analyst(a: dict) -> None:
    header(
        "Your existing analysis",
        "This is the object being reviewed. It is never treated as factual evidence — only your uploaded sources are.",
    )
    workflow_strip(a)
    ver = st.session_state.get("assess_ver", 0)
    tab_paste, tab_upload = st.tabs(["Paste or edit text", "Upload a document"])

    with tab_upload:
        file = st.file_uploader("PDF (text-based), DOCX or TXT", type=["pdf", "docx", "txt"], key=f"as_up_{ver}")
        if st.button("Use this document as my analysis", disabled=file is None):
            try:
                extracted = extraction.extract_file(file.name, file.getvalue())
            except extraction.ExtractionError as exc:
                st.error(f"**{file.name}** could not be used. {exc}")
            else:
                a["assessment_text"] = "\n\n".join(p["text"] for p in extracted["pages"])
                a["assessment_origin"] = extracted["filename"]
                storage.save(a)
                st.session_state["assess_ver"] = ver + 1
                st.session_state["_assess_msg"] = f"Loaded the text of {extracted['filename']}. Review it in the first tab."
                st.rerun()

    with tab_paste:
        if msg := st.session_state.pop("_assess_msg", None):
            st.success(msg)
        text = st.text_area(
            "Your political risk assessment",
            value=a["assessment_text"],
            height=380,
            key=f"assess_{a['id']}_{ver}",
            placeholder="Paste your risks, PESTLE factors, causal reasoning, expected consequences, "
            "scenarios and recommendations here…",
        )
        n = len(text.strip())
        st.caption(f"{n:,} characters" + (f" · loaded from {a['assessment_origin']}" if a["assessment_origin"] else ""))
        if n > config.MAX_ASSESSMENT_CHARS:
            st.warning(
                f"Only the first {config.MAX_ASSESSMENT_CHARS:,} characters are sent to the model, "
                "so that enough room is left for source passages. Put the causal reasoning first."
            )
        if st.button("Save my analysis", type="primary"):
            a["assessment_text"] = text.strip()
            a["assessment_origin"] = a["assessment_origin"] or "pasted text"
            storage.save(a)
            st.success("Saved.")
    if a["assessment_text"].strip() and st.button("Next: detect gaps"):
        go("Gap Detection")


# ---------------------------------------------------------- gap detection ----
EVIDENCE_CSS = {
    "Directly Supported": "pd-b-direct",
    "Inferred": "pd-b-inferred",
    "Insufficient Evidence": "pd-b-insufficient",
}


def block(label: str, text: str) -> None:
    st.markdown(f'<div class="pd-label">{esc(label)}</div><div>{esc(text)}</div>', unsafe_allow_html=True)


def decide(a: dict, sid: str, decision: str) -> None:
    storage.set_decision(a, sid, decision)
    storage.save(a)
    st.rerun()


def save_note(sid: str) -> None:
    a = get_assessment()
    s = storage.get_suggestion(a, sid) if a else None
    if s is not None:
        s["analyst_note"] = st.session_state.get(f"note_{sid}", "").strip()
        storage.save(a)


def render_suggestion(a: dict, s: dict) -> None:
    badges = (
        f'<span class="pd-badge {EVIDENCE_CSS[s["evidence_class"]]}">{esc(s["evidence_class"])}</span>'
        f'<span class="pd-badge pd-b-neutral">{esc(config.MECHANISM_TYPES.get(s["mechanism_type"], ""))}</span>'
        f'<span class="pd-badge pd-b-neutral">{esc(config.ORDERS.get(s["order"], ""))}</span>'
        + "".join(f'<span class="pd-badge pd-b-neutral">{esc(p)}</span>' for p in s.get("pestle", []))
        + f'<span class="pd-badge pd-b-{s["decision"]}">{esc(config.DECISIONS[s["decision"]])}</span>'
    )
    with st.container(border=True):
        st.markdown(badges, unsafe_allow_html=True)
        st.markdown(f'<div class="pd-title">{esc(s["title"])}</div>', unsafe_allow_html=True)
        if s["evidence_class"] == "Insufficient Evidence":
            st.markdown(
                '<div class="pd-hyp"><b>Unverified hypothesis.</b> The uploaded sources do not establish '
                "this relationship. Treat it as a question for further investigation, not as a finding.</div>",
                unsafe_allow_html=True,
            )

        block("Existing analytical context", s["existing_context"])
        ref = s.get("analyst_text_ref", {})
        if ref.get("verified"):
            st.markdown(
                f'<div class="pd-quote"><div class="src">From your assessment</div>'
                f'<div class="txt">{esc(ref["text"])}</div></div>',
                unsafe_allow_html=True,
            )
        block("Missing causal connection", s["missing_connection"])

        chain = "".join(
            ('<span class="pd-arrow">→</span>' if i else "")
            + f'<span class="pd-chip {"impact" if step["node_type"] == "Company Impact" else "event" if step["node_type"] == "Initial Event" else ""}">{esc(step["label"])}</span>'
            for i, step in enumerate(s["causal_chain"])
        )
        st.markdown(f'<div class="pd-label">Causal chain</div><div class="pd-chain">{chain}</div>', unsafe_allow_html=True)
        block("Causal explanation", s["explanation"])
        block("Company-specific implication", s["company_implication"])
        if s.get("countervailing_note"):
            block("Countervailing mechanisms / alternative reading", s["countervailing_note"])

        st.markdown('<div class="pd-label">Supporting evidence (verified against your uploaded sources)</div>', unsafe_allow_html=True)
        if s["evidence"]:
            for ev in s["evidence"]:
                page = f" · page {ev['page']}" if ev.get("page") is not None else ""
                approx = " · approximate match" if ev["match"] == "approximate" else " · exact match"
                st.markdown(
                    f'<div class="pd-quote"><div class="src">{esc(ev["filename"])}{page}{approx}</div>'
                    f'<div class="txt">“{esc(ev["excerpt"])}”</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<div class="pd-note">No citation could be verified in the uploaded sources.</div>', unsafe_allow_html=True)
        if s.get("evidence_rationale"):
            st.markdown(
                f'<div class="pd-note"><b>Why this classification (model-assigned):</b> {esc(s["evidence_rationale"])}</div>',
                unsafe_allow_html=True,
            )
        for note in s.get("validation_notes", []):
            st.markdown(f'<div class="pd-note">⚠ {esc(note)}</div>', unsafe_allow_html=True)

        st.markdown('<div class="pd-label">Your decision</div>', unsafe_allow_html=True)
        sid = s["id"]
        c1, c2, c3, c4 = st.columns([1, 1, 1.7, 1])
        if c1.button("Accept", key=f"acc_{sid}", type="primary" if s["decision"] != "accepted" else "secondary"):
            decide(a, sid, "accepted")
        if c2.button("Reject", key=f"rej_{sid}"):
            decide(a, sid, "rejected")
        if c3.button("Needs Further Investigation", key=f"inv_{sid}"):
            decide(a, sid, "investigate")
        if s["decision"] != "pending" and c4.button("Undo", key=f"und_{sid}"):
            decide(a, sid, "pending")
        st.text_input("Your note (optional)", value=s.get("analyst_note", ""), key=f"note_{sid}",
                      on_change=save_note, args=(sid,))


def run_analysis(a: dict, client: llm.OllamaClient) -> None:
    bar = st.progress(0.0, text="Starting…")

    def on_progress(stage: str, fraction: float, detail: str = "") -> None:
        bar.progress(min(max(fraction, 0.0), 1.0), text=stage + (f" — {detail}" if detail else ""))

    try:
        result = analysis.run_gap_analysis(client, a, on_progress)
    except analysis.AnalysisError as exc:
        bar.empty()
        st.error(f"Analysis stopped. {exc}  \n**No findings were generated.**")
        if exc.raw_output:
            with st.expander("Raw model output (diagnostic only — not a finding)"):
                st.code(exc.raw_output[:6000])
    except llm.BadModelOutput as exc:
        bar.empty()
        st.error(f"Analysis stopped. {exc}")
        if exc.raw:
            with st.expander("Raw model output (diagnostic only — not a finding)"):
                st.code(exc.raw[:6000])
    except llm.LLMError as exc:
        bar.empty()
        st.error(f"Analysis stopped. {exc}  \n**No findings were generated.**")
    except Exception as exc:  # noqa: BLE001 - surface, never fabricate
        bar.empty()
        st.error(f"Unexpected error ({type(exc).__name__}): {exc}  \n**No findings were generated.**")
    else:
        storage.record_run(a, result)
        storage.save(a)
        st.session_state["view_run"] = result["run"]["id"]
        bar.empty()
        n = result["run"]["n_suggestions"]
        if n:
            st.success(f"Done in {result['run']['duration_s']} s. {n} suggestion(s) are ready for your review below.")
        else:
            st.info("The analysis finished but no well-grounded suggestions passed validation. See the run details below.")


def page_gaps(a: dict) -> None:
    header(
        "Gap detection",
        "Political Domino compares your assessment with your uploaded sources and challenges the completeness "
        "of your causal reasoning. Nothing enters your assessment until you accept it.",
    )
    workflow_strip(a)

    problems = analysis.check_inputs(a)
    for p in problems:
        st.warning(p)
    client = make_client()
    status = client.server_status()
    if not status.reachable:
        st.error(
            f"Ollama is not reachable at {client.host}. Install it from https://ollama.com and start it, "
            "then reload this page. No analysis can run without it, and no placeholder results are shown."
        )
    elif not llm.model_installed(client.model, status.models):
        st.error(f"The model **{client.model}** is not installed. Run `ollama pull {client.model}` in a terminal.")

    if st.button("Analyze Potential Gaps", type="primary", disabled=bool(problems) or not status.reachable):
        run_analysis(a, client)
        a = get_assessment()

    if not a["runs"]:
        st.info("No analysis has been run yet.")
        return

    runs = list(reversed(a["runs"]))
    ids = [r["id"] for r in runs]
    default = ids.index(st.session_state["view_run"]) if st.session_state.get("view_run") in ids else 0
    run_id = st.selectbox(
        "Analysis run", ids, index=default,
        format_func=lambda rid: next(
            f"{r['created_at'].replace('T', ' ')} · {r['model']} · {r['n_suggestions']} suggestion(s)"
            for r in runs if r["id"] == rid
        ),
    )
    run = next(r for r in runs if r["id"] == run_id)
    st.session_state["view_run"] = run_id

    with st.expander("About this run: passages used, limitations, discarded items"):
        st.markdown(
            f"Model **{esc(run['model'])}**, context {run['num_ctx']:,} tokens, {run['duration_s']} s. "
            f"{len(run['passages_used'])} of {run['n_chunks_total']} source passages were given to the model.",
            unsafe_allow_html=True,
        )
        for w in run.get("warnings", []):
            st.warning(w)
        if run.get("source_limitations"):
            st.markdown(f"**Source limitations reported by the model:** {run['source_limitations']}")
        if run.get("dropped"):
            st.markdown("**Suggestions discarded by validation (not shown as findings):**")
            for d in run["dropped"]:
                st.markdown(f"- {d['title']} — {d['reason']}")
        used = ", ".join(
            f"{p['filename']}" + (f" p.{p['page']}" if p["page"] is not None else "") for p in run["passages_used"]
        )
        st.caption("Passages: " + used)

    suggestions = [s for s in a["suggestions"] if s["run_id"] == run_id]
    if not suggestions:
        st.info("This run produced no suggestions that passed validation.")
        return
    choice = st.radio(
        "Show", ["All", "Awaiting decision", "Accepted", "Needs further investigation", "Rejected"],
        horizontal=True, key=f"flt_{run_id}",
    )
    wanted = {"All": None, "Awaiting decision": "pending", "Accepted": "accepted",
              "Needs further investigation": "investigate", "Rejected": "rejected"}[choice]
    shown = [s for s in suggestions if wanted is None or s["decision"] == wanted]
    if not shown:
        st.info("Nothing in this view.")
    for s in shown:
        render_suggestion(a, s)
    if any(s["decision"] != "pending" for s in suggestions) and st.button("Next: view the Domino Map"):
        go("Domino Map")


# --------------------------------------------------------------- domino map ----
def page_map(a: dict) -> None:
    header(
        "Domino map",
        "How events cascade into company impacts. Solid nodes are yours; dashed violet nodes are AI suggestions "
        "awaiting a decision; accepted suggestions join your map.",
    )
    options = {
        "Awaiting decision": "pending",
        "Accepted": "accepted",
        "Needs further investigation": "investigate",
        "Rejected (history)": "rejected",
    }
    picked = st.multiselect("Show AI suggestions that are…", list(options), default=list(options)[:3])
    statuses = ("existing",) + tuple(options[p] for p in picked)
    g = graph.build_graph(a, statuses)
    components.html(graph.render_html(g, 640, a["id"]), height=660, scrolling=False)
    st.caption(
        "Click a node for its explanation and evidence. Drag nodes to rearrange them (positions are remembered in "
        "this browser). Accept or reject suggestions on the Gap Detection page; rejected ones stay in the review history."
    )
    if not a["analyst_map"]["nodes"] and not g["nodes"]:
        st.info("The map fills in after you run the gap analysis, or you can add nodes below.")

    with st.expander("Edit the map"):
        label_of = {n["id"]: f"{n['label']}  ({'AI' if n['origin'] == 'ai' else 'yours'})" for n in g["nodes"]}
        t_add, t_link, t_edit, t_unlink = st.tabs(["Add node", "Add link", "Edit or remove node", "Remove link"])

        with t_add:
            with st.form("add_node"):
                label = st.text_input("Label")
                ntype = st.selectbox("Type", config.NODE_TYPES, index=1)
                desc = st.text_area("Description", height=80)
                if st.form_submit_button("Add node") and label.strip():
                    a["manual_nodes"].append(
                        {"id": uuid.uuid4().hex[:6], "label": label.strip()[:90], "node_type": ntype,
                         "description": desc.strip()}
                    )
                    storage.save(a)
                    st.rerun()

        with t_link:
            if len(label_of) < 2:
                st.info("Add at least two nodes first.")
            else:
                with st.form("add_link"):
                    src = st.selectbox("From (cause)", list(label_of), format_func=label_of.get)
                    dst = st.selectbox("To (effect)", list(label_of), index=1, format_func=label_of.get)
                    text = st.text_input("Link label (optional)")
                    if st.form_submit_button("Add link"):
                        if src == dst:
                            st.error("Choose two different nodes.")
                        else:
                            a["manual_edges"].append({"source": src, "target": dst, "label": text.strip()[:40]})
                            storage.save(a)
                            st.rerun()

        with t_edit:
            editable = [n for n in g["nodes"] if n["origin"] in ("analyst", "manual")]
            ai_nodes = [n for n in g["nodes"] if n["origin"] == "ai"]
            if ai_nodes:
                st.caption("AI-suggested nodes cannot be edited here: accept or reject them on the Gap Detection page.")
            if not editable:
                st.info("No editable nodes yet.")
            else:
                pick = st.selectbox("Node", [n["id"] for n in editable], format_func=label_of.get, key="edit_pick")
                node = next(n for n in editable if n["id"] == pick)
                new_label = st.text_input("Label", node["label"], key=f"en_l_{pick}")
                new_type = st.selectbox("Type", config.NODE_TYPES, index=config.NODE_TYPES.index(node["node_type"]), key=f"en_t_{pick}")
                new_desc = st.text_area("Description", node["description"], key=f"en_d_{pick}", height=80)
                c1, c2 = st.columns(2)
                target_list = a["analyst_map"]["nodes"] if node["origin"] == "analyst" else a["manual_nodes"]
                record = next(n for n in target_list if n["id"] == node["ref"])
                if c1.button("Save node", key=f"en_s_{pick}") and new_label.strip():
                    record.update(label=new_label.strip()[:90], node_type=new_type, description=new_desc.strip())
                    storage.save(a)
                    st.rerun()
                if c2.button("Remove node", key=f"en_r_{pick}"):
                    target_list.remove(record)
                    if node["origin"] == "analyst":
                        a["analyst_map"]["edges"] = [
                            e for e in a["analyst_map"]["edges"] if node["ref"] not in (e["source"], e["target"])
                        ]
                    a["manual_edges"] = [e for e in a["manual_edges"] if pick not in (e["source"], e["target"])]
                    storage.save(a)
                    st.rerun()

        with t_unlink:
            links = [e for e in g["edges"] if e["origin"] in ("analyst", "manual")]
            if not links:
                st.info("No editable links.")
            else:
                idx = st.selectbox(
                    "Link", range(len(links)),
                    format_func=lambda i: f"{label_of.get(links[i]['source'], '?')}  →  {label_of.get(links[i]['target'], '?')}",
                    key="unlink_pick",
                )
                if st.button("Remove link"):
                    kind, position = links[idx]["ref"]
                    (a["analyst_map"]["edges"] if kind == "analyst" else a["manual_edges"]).pop(position)
                    storage.save(a)
                    st.rerun()


# ---------------------------------------------------------- review & export ----
def page_review(a: dict) -> None:
    header(
        "Review & export",
        "Only suggestions you accepted join the revised assessment. Items marked for follow-up stay separate.",
    )
    workflow_strip(a)
    accepted = export.decisions_of(a, "accepted")
    investigate = export.decisions_of(a, "investigate")
    rejected = export.decisions_of(a, "rejected")
    pending = export.decisions_of(a, "pending")
    m = st.columns(4)
    m[0].metric("Accepted", len(accepted))
    m[1].metric("Needs investigation", len(investigate))
    m[2].metric("Rejected", len(rejected))
    m[3].metric("Not yet reviewed", len(pending))
    if pending:
        st.warning(f"{len(pending)} suggestion(s) have no decision yet. They will not appear in the report.")

    st.subheader("Accepted additions")
    if not accepted:
        st.info("Nothing accepted yet.")
    for s in accepted:
        with st.expander(f"{s['title']}  ·  {s['evidence_class']}"):
            st.markdown("**Causal chain:** " + " → ".join(step["label"] for step in s["causal_chain"]))
            st.markdown(f"**Explanation:** {s['explanation']}")
            st.markdown(f"**Company implication:** {s['company_implication']}")
            for ev in s["evidence"]:
                page = f", page {ev['page']}" if ev.get("page") is not None else ""
                st.markdown(f"> “{ev['excerpt']}” — *{ev['filename']}{page}*")
            if s["evidence_class"] == "Insufficient Evidence":
                st.warning("Accepted as a working hypothesis: the sources do not establish it.")

    st.subheader("Outstanding hypotheses requiring further investigation")
    if not investigate:
        st.info("None.")
    for s in investigate:
        with st.expander(f"{s['title']}  ·  {s['evidence_class']}"):
            st.markdown(f"**Missing connection:** {s['missing_connection']}")
            st.markdown("**Causal chain:** " + " → ".join(step["label"] for step in s["causal_chain"]))

    st.subheader("Download")
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", a["name"])[:40] or "assessment"
    c1, c2, c3 = st.columns(3)
    try:
        c1.download_button(
            "Report (DOCX)", export.build_docx(a), file_name=f"{slug}_review.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", type="primary",
        )
    except Exception as exc:  # noqa: BLE001
        c1.error(f"The report could not be generated ({type(exc).__name__}).")
    c2.download_button("Archive (JSON)", export.build_json(a), file_name=f"{slug}.json", mime="application/json")
    c3.download_button("Decision log (CSV)", export.build_csv(a), file_name=f"{slug}_decisions.csv", mime="text/csv")


# -------------------------------------------------------------------- main ----
PAGES = {
    "Assessment": page_assessment,
    "Sources": page_sources,
    "Analyst Assessment": page_analyst,
    "Gap Detection": page_gaps,
    "Domino Map": page_map,
    "Review & Export": page_review,
}


def main() -> None:
    st.markdown(styles.CSS, unsafe_allow_html=True)
    nav = sidebar()
    a = get_assessment()
    if nav == "Dashboard":
        page_dashboard()
    elif a is None:
        header(nav)
        st.info("Open or create an assessment on the Dashboard first.")
        if st.button("Go to Dashboard"):
            go("Dashboard")
    else:
        PAGES[nav](a)
    st.markdown(f'<div class="pd-disclaimer">{esc(config.DISCLAIMER)}</div>', unsafe_allow_html=True)


main()
