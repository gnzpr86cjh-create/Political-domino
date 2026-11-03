"""Central configuration for Political Domino."""
from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "Political Domino"
TAGLINE = "The analyst researches. Political Domino challenges. The analyst decides."
DISCLAIMER = (
    "Political Domino is an AI-assisted analytical support tool. Its outputs are "
    "hypotheses and evidence-based suggestions that require independent validation "
    "by the human analyst."
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("POLITICAL_DOMINO_DATA", str(BASE_DIR / "data")))


def normalize_host(raw: str | None) -> str:
    """Return a clean Ollama base URL (adds http:// if missing)."""
    raw = (raw or "").strip() or "http://localhost:11434"
    if not raw.startswith(("http://", "https://")):
        raw = "http://" + raw
    return raw.rstrip("/")


# POLITICAL_DOMINO_OLLAMA_URL wins over OLLAMA_HOST (which is often a *bind* address).
OLLAMA_URL = normalize_host(
    os.environ.get("POLITICAL_DOMINO_OLLAMA_URL") or os.environ.get("OLLAMA_HOST")
)
DEFAULT_MODEL = os.environ.get("POLITICAL_DOMINO_MODEL", "qwen3:4b")

# --- Processing limits -------------------------------------------------------
MAX_SOURCES = 5                 # optimised for ~3 text-based PDFs; 5 is the hard cap
MAX_FILE_MB = 15
MAX_SOURCE_CHARS = 600_000      # text beyond this is truncated (with a visible warning)
MIN_TOTAL_CHARS = 150           # below this a PDF is treated as scanned / empty
MIN_PAGE_CHARS = 30             # pages below this are reported as "little or no text"
MAX_ASSESSMENT_CHARS = 6_000    # analyst text sent to the model (rest is not analysed)
CHUNK_CHARS = 900
CHUNK_OVERLAP_SENTENCES = 1
MAX_SUGGESTIONS = 5
MAX_EVIDENCE_PER_SUGGESTION = 4
MAX_MAP_NODES = 14
MAX_MAP_EDGES = 18

# --- Model settings ----------------------------------------------------------
NUM_CTX_OPTIONS = [8192, 12288, 16384, 24576]
DEFAULT_NUM_CTX = 12288
GAP_NUM_PREDICT = 3200          # output budget (tokens) for the gap analysis call
MAP_NUM_PREDICT = 1500
TEMPERATURE = 0.2
REQUEST_TIMEOUT_S = 1200        # CPU inference of a 4B model can be slow

# --- Vocabulary --------------------------------------------------------------
PESTLE = ["Political", "Economic", "Social", "Technological", "Legal", "Environmental"]
NODE_TYPES = ["Initial Event", "Intermediate Mechanism", "Company Impact"]
EVIDENCE_CLASSES = ["Directly Supported", "Inferred", "Insufficient Evidence"]
MECHANISM_TYPES = {
    "overlooked_consequence": "Overlooked consequence",
    "interaction": "Cross-dimension interaction",
    "dependency": "Dependency between risks",
    "alternative_scenario": "Alternative scenario",
    "countervailing": "Countervailing mechanism",
    "company_vulnerability": "Company-specific vulnerability",
}
ORDERS = {"direct": "Direct", "second": "Second-order", "third": "Third-order"}
DECISIONS = {
    "pending": "Awaiting decision",
    "accepted": "Accepted",
    "rejected": "Rejected",
    "investigate": "Needs further investigation",
}
