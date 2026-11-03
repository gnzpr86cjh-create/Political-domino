"""Text extraction for PDF, DOCX and TXT files.

Every extractor returns the same structure::

    {
        "filename": str,
        "file_type": "pdf" | "docx" | "txt",
        "pages": [{"page": int | None, "text": str}, ...],
        "n_chars": int,
        "warnings": [str, ...],
    }

PDF page numbers are preserved. DOCX and TXT have no reliable pagination, so
``page`` is ``None`` for them.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

from . import config


class ExtractionError(Exception):
    """Raised with a user-facing message when a file cannot be used."""


SUPPORTED = {".pdf": "pdf", ".docx": "docx", ".txt": "txt"}


def clean_text(text: str) -> str:
    """Normalise whitespace and repair PDF line-break hyphenation."""
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=[A-Za-zÀ-ɏ])-\n(?=[a-zà-ɏ])", "", text)
    text = re.sub(r"[ \t ​]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------- PDF ----
def _extract_pdf(data: bytes) -> tuple[list[dict], list[str]]:
    from pypdf import PdfReader  # imported lazily so other formats work without it

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001 - pypdf raises many exception types
        raise ExtractionError(
            f"The PDF could not be opened ({type(exc).__name__}). The file may be corrupted."
        ) from exc

    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")
        except Exception:  # noqa: BLE001
            unlocked = 0
        if not unlocked:
            raise ExtractionError(
                "The PDF is password-protected. Remove the password and upload it again."
            )

    pages: list[dict] = []
    thin_pages: list[int] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:  # noqa: BLE001
            raw = ""
        text = clean_text(raw)
        if len(text) < config.MIN_PAGE_CHARS:
            thin_pages.append(number)
            continue
        pages.append({"page": number, "text": text})

    total = sum(len(p["text"]) for p in pages)
    if total < config.MIN_TOTAL_CHARS:
        raise ExtractionError(
            "No extractable text was found. This looks like a scanned or image-only PDF. "
            "Political Domino does not run OCR: upload a text-based PDF, or export the "
            "content to DOCX or TXT."
        )

    warnings: list[str] = []
    if thin_pages:
        shown = ", ".join(str(p) for p in thin_pages[:12])
        more = " …" if len(thin_pages) > 12 else ""
        warnings.append(
            f"Little or no extractable text on page(s) {shown}{more}. "
            "These pages may be scans, images or charts and are not available as evidence."
        )
    return pages, warnings


# -------------------------------------------------------------------- DOCX ----
def _extract_docx(data: bytes) -> tuple[list[dict], list[str]]:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        doc = Document(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(
            f"The DOCX could not be opened ({type(exc).__name__}). "
            "The file may be corrupted or not a real Word document."
        ) from exc

    blocks: list[str] = []
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            text = Paragraph(child, doc).text.strip()
            if text:
                blocks.append(text)
        elif tag == "tbl":
            table = Table(child, doc)
            for row in table.rows:
                cells = []
                for cell in row.cells:
                    value = cell.text.strip().replace("\n", " ")
                    if value and (not cells or cells[-1] != value):  # skip merged-cell repeats
                        cells.append(value)
                if cells:
                    blocks.append(" | ".join(cells))

    text = clean_text("\n\n".join(blocks))
    if len(text) < config.MIN_TOTAL_CHARS // 3:
        raise ExtractionError("The DOCX contains no (or almost no) text.")
    return [{"page": None, "text": text}], [
        "DOCX files have no reliable page numbers; citations will show the filename only."
    ]


# --------------------------------------------------------------------- TXT ----
def _extract_txt(data: bytes) -> tuple[list[dict], list[str]]:
    encodings = ["utf-8-sig", "cp1252", "latin-1"]
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        encodings.insert(0, "utf-16")
    text = None
    for encoding in encodings:
        try:
            text = data.decode(encoding)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if text is None:
        raise ExtractionError("The text file could not be decoded.")
    text = clean_text(text)
    if len(text) < config.MIN_TOTAL_CHARS // 3:
        raise ExtractionError("The text file is empty or contains almost no text.")
    return [{"page": None, "text": text}], [
        "TXT files have no page numbers; citations will show the filename only."
    ]


# ------------------------------------------------------------------ public ----
def extract_file(filename: str, data: bytes) -> dict:
    """Extract text from an uploaded file or raise :class:`ExtractionError`."""
    suffix = Path(filename).suffix.lower()
    file_type = SUPPORTED.get(suffix)
    if file_type is None:
        raise ExtractionError(
            f"Unsupported file type '{suffix or 'none'}'. Upload PDF (text-based), DOCX or TXT."
        )
    if not data:
        raise ExtractionError("The file is empty (0 bytes).")
    if len(data) > config.MAX_FILE_MB * 1024 * 1024:
        raise ExtractionError(
            f"The file is larger than {config.MAX_FILE_MB} MB. Split it or upload an excerpt."
        )

    extractor = {"pdf": _extract_pdf, "docx": _extract_docx, "txt": _extract_txt}[file_type]
    pages, warnings = extractor(data)

    total = sum(len(p["text"]) for p in pages)
    if total > config.MAX_SOURCE_CHARS:
        kept: list[dict] = []
        used = 0
        for p in pages:
            room = config.MAX_SOURCE_CHARS - used
            if room <= 0:
                break
            text = p["text"][:room]
            kept.append({"page": p["page"], "text": text})
            used += len(text)
        pages = kept
        warnings.append(
            f"The document is very long ({total:,} characters). Only the first "
            f"{config.MAX_SOURCE_CHARS:,} characters were kept; later content cannot be cited."
        )

    return {
        "filename": Path(filename).name,
        "file_type": file_type,
        "pages": pages,
        "n_chars": sum(len(p["text"]) for p in pages),
        "warnings": warnings,
    }


# ------------------------------------------------------- instruction safety ----
_INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
    r"disregard (all |any |the )?(previous|prior|above|earlier|your) (instructions|prompts?|rules)",
    r"(reveal|show|print) (your|the) (system )?prompt",
    r"you are now (a|an|the) ",
    r"new instructions?:",
    r"override (the )?(analytical )?rules",
    r"(classify|label|mark) (everything|all|this) as directly supported",
]


def scan_for_injection(text: str) -> list[str]:
    """Return short snippets that look like instructions aimed at an AI model.

    The text is *never* removed or obeyed: the analyst is only warned, and the
    model is told that all source text is data.
    """
    hits: list[str] = []
    for pattern in _INJECTION_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            start = max(0, match.start() - 20)
            snippet = re.sub(r"\s+", " ", text[start : match.end() + 40]).strip()
            hits.append(snippet)
            if len(hits) >= 5:
                return hits
    return hits
