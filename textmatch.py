"""Verify that a quotation really occurs in a piece of text.

This is the safeguard against fabricated citations: the model proposes a quote
and a passage id, and nothing is shown to the analyst unless the quote is found
in the real extracted text. The *displayed* excerpt is always taken from the
source text, never from the model's wording.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

_TRANSLATE = str.maketrans(
    {
        "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
        "‘": "'", "’": "'", "‚": "'",
        "–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-",
        " ": " ", "​": "",
    }
)

MIN_QUOTE_CHARS = 15
_ELLIPSIS = re.compile(r"\s*(?:\[\s*\.{3}\s*\]|\.{3,}|…)\s*")


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Lower-case, collapse whitespace, unify quotes/dashes.

    Returns the normalised string and, for each normalised character, the index
    of the original character it came from (so spans can be mapped back).
    """
    out: list[str] = []
    index: list[int] = []
    previous_space = True
    for i, ch in enumerate(text):
        ch = ch.translate(_TRANSLATE)
        if ch == "":
            continue
        if ch.isspace():
            if previous_space:
                continue
            out.append(" ")
            index.append(i)
            previous_space = True
            continue
        for lowered in ch.lower():
            out.append(lowered)
            index.append(i)
        previous_space = False
    while out and out[-1] == " ":
        out.pop()
        index.pop()
    return "".join(out), index


def _find_segment(segment: str, text: str, threshold: float) -> tuple[str, str] | None:
    """Return (status, original-text-span) or None."""
    n_text, idx = normalize_with_map(text)
    n_seg, _ = normalize_with_map(segment)
    if len(n_seg) < MIN_QUOTE_CHARS or not n_text:
        return None

    pos = n_text.find(n_seg)
    if pos >= 0:
        return "exact", text[idx[pos] : idx[pos + len(n_seg) - 1] + 1]

    # Approximate: slide a window of the quote's length over the text.
    length = len(n_seg)
    if length > len(n_text):
        return None
    matcher = SequenceMatcher(None, autojunk=False)
    matcher.set_seq2(n_seg)
    best_ratio, best_start = 0.0, -1
    stride = max(4, length // 10)
    for start in range(0, len(n_text) - length + 1, stride):
        matcher.set_seq1(n_text[start : start + length])
        if matcher.real_quick_ratio() < threshold or matcher.quick_ratio() < threshold:
            continue
        ratio = matcher.ratio()
        if ratio > best_ratio:
            best_ratio, best_start = ratio, start
    if best_ratio < threshold:
        return None

    # Refine the span around the best window and snap it to word boundaries.
    lo = max(0, best_start - stride)
    hi = min(len(n_text), best_start + length + stride)
    while lo > 0 and n_text[lo - 1] != " ":
        lo -= 1
    while hi < len(n_text) and n_text[hi] != " ":
        hi += 1
    return "approximate", text[idx[lo] : idx[hi - 1] + 1]


def find_quote(quote: str, text: str, threshold: float = 0.92) -> dict:
    """Locate ``quote`` inside ``text``.

    Returns ``{"status": "exact" | "approximate" | "not_found", "excerpt": str}``.
    A quote containing ellipses ("…", "...", "[...]") is split and *every*
    segment must be found. ``excerpt`` is real text copied from ``text``.
    """
    quote = (quote or "").strip().strip("\"'“”")
    if not quote or not text:
        return {"status": "not_found", "excerpt": ""}

    segments = [s for s in _ELLIPSIS.split(quote) if s.strip()]
    if not segments:
        return {"status": "not_found", "excerpt": ""}

    spans: list[str] = []
    statuses: list[str] = []
    for segment in segments:
        found = _find_segment(segment, text, threshold)
        if found is None:
            return {"status": "not_found", "excerpt": ""}
        statuses.append(found[0])
        spans.append(found[1])

    status = "exact" if all(s == "exact" for s in statuses) else "approximate"
    return {"status": status, "excerpt": " [...] ".join(spans)}
