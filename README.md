# Political Domino

AI-assisted political risk analysis and causal gap detection.

> The analyst researches. Political Domino challenges. The analyst decides.

Political Domino does not do research. You upload the sources you collected and the assessment you wrote;
a **local** language model (Ollama) looks for causal mechanisms your assessment may have missed, cites the
uploaded sources for each one, and you accept, reject or flag every suggestion. Nothing is fetched from the
internet and no API keys or accounts are needed.

*Political Domino is an AI-assisted analytical support tool. Its outputs are hypotheses and evidence-based
suggestions that require independent validation by the human analyst.*

## Install

You need Python 3.10+ and [Ollama](https://ollama.com).

```bash
# 1. Python dependencies
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Local model (one-time download, about 2.5 GB)
ollama pull qwen3:4b

# 3. Check everything
python check_setup.py
```

Ollama must be running (the desktop app, or `ollama serve`). After the model is downloaded the app works offline.

## Launch

```bash
streamlit run app.py          # or ./run.sh (macOS/Linux), run.bat (Windows)
```

Other model or address: `POLITICAL_DOMINO_MODEL=qwen3:8b` and `POLITICAL_DOMINO_OLLAMA_URL=http://host:11434`.
Data is stored as JSON in `data/assessments/` (override with `POLITICAL_DOMINO_DATA`).

## How it works

```
sources (PDF/DOCX/TXT) -> text extraction (page numbers kept) -> passages -> BM25 retrieval ─┐
analyst assessment ---------> LLM call 1: extract the analyst's own causal map ──────────────┼-> LLM call 2:
                                                                                              │   gap detection (JSON)
                    programmatic validation: every quote must exist in the source text <──────┘
                                         │
                      review cards (Accept / Reject / Needs Further Investigation) -> Domino Map -> DOCX/JSON/CSV
```

Safeguards:

- **Quotes are checked in code.** The model must quote a passage verbatim; the quote is searched in the real
  extracted text. Unfound quotes are discarded and the excerpt shown is always the source's own text.
- **Evidence class cannot be overstated.** If no citation survives verification, a suggestion claimed as
  "Directly Supported" or "Inferred" is shown as *Insufficient Evidence*, an unverified hypothesis.
- **Invalid output produces no findings.** Malformed JSON, a missing model or a stopped Ollama gives an error
  message, never placeholder results. Incomplete, one-step or duplicate suggestions are dropped and listed under
  "About this run".
- **Documents are data, not instructions.** Source and assessment text is fenced, the model is told to ignore
  instructions inside it, and instruction-like text in a source triggers a visible warning.
- **Nothing is merged automatically.** Only suggestions you accept enter the revised assessment in the export;
  "Needs Further Investigation" items stay in a separate section; rejected ones stay in the review log.

## Limits

- Up to 5 sources, 15 MB each. Optimised for about three text-based PDFs.
- No OCR: scanned PDFs are rejected with a message. DOCX and TXT have no page numbers (citations show the filename).
- Only the first 6,000 characters of your assessment are analysed; put the causal reasoning first.
- With a 4B model, expect imperfect reasoning and slow CPU inference (minutes). Verification protects the citations,
  not the quality of the reasoning, which is exactly why you review every suggestion.

## Tests

```bash
python -m unittest discover -s tests -t .
```

The suite (29 tests) runs the real client, retrieval, validation, storage, map and export code against a small
fake Ollama HTTP server in `tests/fake_ollama.py`. That fake exists only for tests; the app never ships canned results.
It covers: PDF/DOCX/TXT extraction, scanned/corrupt/empty/unsupported/oversized files, citation verification,
fabricated and misattributed quotes, invalid JSON, missing Ollama, missing model, server errors, and the
accept/reject/investigate flow into the map and the report.

It does **not** test the quality of a real model's reasoning, or the Streamlit UI itself. Do the manual test below.

## End-to-end manual test (about 10 minutes with a real model)

All files in `samples/` are **fictional** (invented country "Tarvos", mineral "tarvonite", company "Lumeria Drives").

1. **Dashboard:** create *Name* `Tarvos test`, *Company* `Lumeria Drives S.p.A.`, *Question*
   `How will Tarvos's export licensing of tarvonite affect Lumeria's profitability over three years?`
2. **Assessment:** optionally fill Industry `Industrial automation`, Market `Europe`, Horizon `3 years`; Save.
3. **Sources:** upload the three `samples/source*` files; click *Process uploaded files*. Open *Inspect extracted text*
   and check the page numbers of the two PDFs.
4. **Analyst Assessment:** paste `samples/analyst_assessment_lumeria.txt` (or upload the `.docx`); Save.
5. **Gap Detection:** click *Analyze Potential Gaps* and wait for both steps.
6. **Inspect:** each card shows context, missing connection, causal chain, explanation, company implication,
   countervailing note, evidence with filename and page, and the evidence class. Verify one excerpt in the PDF.
   Gaps worth expecting from the sources: licence lead time (up to 45 working days) vs 20-30 days of inventory,
   the 6-month fixed-volume licence, the 30% domestic quota, 18-24 months to substitute magnets, competitors'
   90-120 days of stock, customer late-delivery penalties.
7. **Decide:** Accept one, Reject one, mark one *Needs Further Investigation*; add a note.
8. **Domino Map:** your nodes are solid, pending suggestions dashed violet, accepted ones solid violet with a check.
   Click nodes, drag them, toggle *Rejected (history)*. Use *Edit the map* to add a node and a link.
9. **Review & Export:** download the DOCX. Accepted items appear under "Revised assessment", flagged ones under
   "Outstanding hypotheses", rejected only in the review log.

Error checks (each must show a clear message and **no findings**):

| Test | How |
|---|---|
| Ollama missing | Quit Ollama, reload, open Gap Detection |
| Model missing | Pick or set a model that is not installed |
| Scanned PDF | Upload `samples/error_cases/scanned_image_only.pdf` |
| Corrupt / empty / unsupported | `corrupt.pdf`, `empty.txt`, `whitespace_only.txt`, `notes.csv` |
| Prompt injection | Upload `injection_source.txt`; a warning appears and it is not obeyed |
| Too many sources | Upload a sixth source |

## Project layout

```
app.py                     Streamlit interface (7 pages)
check_setup.py             Pre-flight check
political_domino/
  config.py                Limits, model settings, vocabulary
  extraction.py            PDF/DOCX/TXT extraction, injection scan
  retrieval.py             Chunking and BM25 retrieval
  textmatch.py             Quote verification against source text
  llm.py                   Ollama client (JSON-schema constrained output)
  analysis.py              Prompts, pipeline, validation
  storage.py               Local JSON persistence
  graph.py                 Domino Map build, layout, offline SVG renderer
  export.py                DOCX, JSON, CSV
  styles.py                Visual identity
samples/                   Fictional test inputs (+ error_cases/)
tests/                     Unit and pipeline tests, fake Ollama server
```
