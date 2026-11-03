"""Source segmentation and simple local retrieval (BM25, no vector database)."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from . import config


@dataclass(frozen=True)
class Chunk:
    chunk_id: str        # e.g. "S1-P3-C2" (PDF) or "S2-C1" (no pages)
    source_id: str       # e.g. "S1"
    filename: str
    page: int | None
    text: str


# ---------------------------------------------------------------- chunking ----
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:])\s+(?=[A-Z0-9À-Ý\"“(\[])")


def _split_long(text: str, limit: int) -> list[str]:
    parts: list[str] = []
    while len(text) > limit:
        cut = text.rfind(" ", 0, limit)
        if cut < limit * 0.5:
            cut = limit
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        parts.append(text)
    return parts


def _sentences(page_text: str, limit: int) -> list[str]:
    out: list[str] = []
    for paragraph in re.split(r"\n\s*\n", page_text):
        flat = re.sub(r"\s+", " ", paragraph).strip()
        if not flat:
            continue
        for sentence in _SENTENCE_SPLIT.split(flat):
            sentence = sentence.strip()
            if sentence:
                out.extend(_split_long(sentence, limit))
    return out


def chunk_page(
    page_text: str,
    max_chars: int = config.CHUNK_CHARS,
    overlap_sentences: int = config.CHUNK_OVERLAP_SENTENCES,
) -> list[str]:
    """Pack sentences into chunks of at most ``max_chars`` (never crossing a page)."""
    sentences = _sentences(page_text, max_chars)
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in sentences:
        added = len(sentence) + (1 if current else 0)
        if current and size + added > max_chars:
            chunks.append(" ".join(current))
            carry = current[-overlap_sentences:] if overlap_sentences else []
            carry = [s for s in carry if len(s) < max_chars / 3]
            current = list(carry)
            size = len(" ".join(current))
            added = len(sentence) + (1 if current else 0)
        current.append(sentence)
        size += added
    if current:
        last = " ".join(current)
        if not chunks or last != chunks[-1]:
            chunks.append(last)
    return chunks


def chunk_sources(sources: list[dict]) -> list[Chunk]:
    """Deterministically segment all sources. Chunk ids are stable between runs."""
    chunks: list[Chunk] = []
    for source in sources:
        counter = 0
        for page in source.get("pages", []):
            for piece in chunk_page(page["text"]):
                counter += 1
                if page["page"] is not None:
                    cid = f"{source['id']}-P{page['page']}-C{counter}"
                else:
                    cid = f"{source['id']}-C{counter}"
                chunks.append(Chunk(cid, source["id"], source["filename"], page["page"], piece))
    return chunks


# --------------------------------------------------------------- retrieval ----
_STOP = set(
    """the and for with that this from are was were has have had not but its their they them his her
    which will would could should may might can also into than then over under between about after
    before because while where when what who whom whose such these those been being more most other
    some any all each per via our your out off one two new use used using
    che con per una uno gli dei del della delle dello degli nel nella non sono come anche più
    alla alle allo agli dal dalla dalle sul sulla tra fra""".split()
)


def _stem(word: str) -> str:
    for suffix in ("ations", "ation", "ing", "ies", "ied", "ed", "es", "s", "ly"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)] + ("y" if suffix in ("ies", "ied") else "")
    return word


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[^\W_]{3,}", text.lower())
    return [_stem(w) for w in words if w not in _STOP]


def rank_chunks(chunks: list[Chunk], query: str, k1: float = 1.5, b: float = 0.75):
    """Return ``[(score, chunk), ...]`` sorted best-first using BM25."""
    if not chunks:
        return []
    docs = [Counter(tokenize(c.text)) for c in chunks]
    lengths = [sum(d.values()) or 1 for d in docs]
    avg = sum(lengths) / len(lengths)
    n = len(chunks)
    df: Counter = Counter()
    for d in docs:
        df.update(d.keys())
    q_terms = set(tokenize(query))
    scored = []
    for chunk, doc, length in zip(chunks, docs, lengths):
        score = 0.0
        for term in q_terms:
            tf = doc.get(term, 0)
            if not tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * length / avg))
        scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def select_evidence(
    chunks: list[Chunk],
    query: str,
    char_budget: int,
    min_per_source: int = 2,
) -> list[Chunk]:
    """Pick the most relevant chunks within ``char_budget`` characters.

    Every source is guaranteed some representation so that one long document
    cannot crowd out the others. The result is returned in reading order.
    """
    ranked = rank_chunks(chunks, query)
    chosen: dict[str, Chunk] = {}
    used = 0

    def take(chunk: Chunk) -> bool:
        nonlocal used
        if chunk.chunk_id in chosen:
            return False
        if used + len(chunk.text) > char_budget and chosen:
            return False
        chosen[chunk.chunk_id] = chunk
        used += len(chunk.text)
        return True

    by_source: dict[str, list[tuple[float, Chunk]]] = {}
    for score, chunk in ranked:
        by_source.setdefault(chunk.source_id, []).append((score, chunk))
    for items in by_source.values():
        for score, chunk in items[:min_per_source]:
            take(chunk)
    for score, chunk in ranked:
        if score <= 0 and len(chosen) >= len(by_source) * min_per_source:
            break
        take(chunk)

    order = {c.chunk_id: i for i, c in enumerate(chunks)}
    return sorted(chosen.values(), key=lambda c: order[c.chunk_id])
