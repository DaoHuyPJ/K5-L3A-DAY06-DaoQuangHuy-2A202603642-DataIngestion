"""Fixed-size (strategy A) and structure-aware (strategy B/C) chunking.

Every chunk keeps document_id, source, page (= unit number) and the character
span inside that unit, so it can always be traced back to the source text.
Chapter/section/article metadata is derived from the unit's paragraph offsets,
never guessed.
"""
from __future__ import annotations

import hashlib
import re

from .models import Chunk, DocumentPage


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _span_metadata(page: DocumentPage, start: int, end: int) -> dict:
    """Metadata of a [start, end) span: chapter + the articles/sections it overlaps."""
    paras = [p for p in page.metadata["paragraphs"] if p["start"] < end and p["end"] > start]
    articles = sorted({p["article"] for p in paras if p["article"] is not None})
    sections = sorted({p["section"] for p in paras if p["section"]}, key=int)
    return {
        "unit_id": page.metadata["unit_id"],
        "chapter": page.metadata["chapter"],
        "chapter_title": page.metadata["chapter_title"],
        "section": sections[0] if len(sections) == 1 else None,
        "articles": articles,
        "char_start": start,
        "char_end": end,
    }


def _make_chunk(strategy: str, page: DocumentPage, index: int, text: str,
                start: int, end: int, **extra) -> Chunk:
    meta = _span_metadata(page, start, end)
    meta.update(extra)
    meta["content_hash"] = _hash(text)
    meta["char_count"] = len(text)
    return Chunk(
        chunk_id=f"{strategy}:{page.document_id}:u{page.page:03d}:c{index:03d}",
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata=meta,
    )


# --------------------------------------------------------------------------- #
# Strategy A — fixed-size character windows
# --------------------------------------------------------------------------- #
def fixed_size(page: DocumentPage, size: int, overlap: int, strategy: str = "A") -> list[Chunk]:
    """Split the unit text into windows of ``size`` chars, stepping ``size - overlap``.

    Deliberately structure-blind (it may cut mid-word / mid-article): this is
    the baseline.  Windows never cross a unit, so page metadata stays exact.
    """
    if not 0 <= overlap < size:
        raise ValueError("require 0 <= overlap < size")
    text = page.text
    chunks: list[Chunk] = []
    step = size - overlap
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end]
        if piece.strip():
            chunks.append(_make_chunk(strategy, page, len(chunks), piece, start, end))
        if end == len(text):
            break
        start += step
    return chunks


# --------------------------------------------------------------------------- #
# Strategy B — structure-aware: paragraphs grouped within one article
# --------------------------------------------------------------------------- #
_SENTENCE_SPLIT = re.compile(r"(?<=[.;:])\s+")


def _split_long(text: str, offset: int, max_size: int) -> list[tuple[str, int, int]]:
    """Split one oversized paragraph at sentence/clause ends (fallback: hard cut).

    Returns (text, absolute_start, absolute_end) where absolute = offset + local.
    """
    spans, last = [], 0
    for m in _SENTENCE_SPLIT.finditer(text):
        spans.append((last, m.start()))
        last = m.end()
    spans.append((last, len(text)))

    pieces: list[tuple[int, int]] = []
    cur_s = cur_e = None
    for s, e in spans:
        if cur_s is None:
            cur_s, cur_e = s, e
        elif e - cur_s > max_size:
            pieces.append((cur_s, cur_e))
            cur_s, cur_e = s, e
        else:
            cur_e = e
        while cur_e - cur_s > max_size:          # a single sentence longer than max_size
            pieces.append((cur_s, cur_s + max_size))
            cur_s += max_size
    if cur_s is not None and cur_e > cur_s:
        pieces.append((cur_s, cur_e))
    return [(text[s:e], s + offset, e + offset) for s, e in pieces]


def by_paragraph(page: DocumentPage, max_size: int, strategy: str = "B") -> list[Chunk]:
    """Group consecutive paragraphs of the same article up to ``max_size`` chars.

    - Chunks never cross an article ("Điều") boundary.
    - Chapter/section headings stay with the first chunk of the next article.
    - A paragraph longer than ``max_size`` is split at sentence/clause ends.
    - Continuation chunks of an article get the article heading prepended
      ("Điều N. Tên điều") so they remain self-explanatory; the span metadata
      still points to the original characters, and ``context_prefix`` records
      the added heading.
    """
    text = page.text
    paras = page.metadata["paragraphs"]

    # segments: (article, start, end, article_heading_text)
    segments: list[tuple[int | None, int, int]] = []
    pending_heading_start = None
    for p in paras:
        if p["kind"] in ("chapter", "chapter_title", "section"):
            if pending_heading_start is None:
                pending_heading_start = p["start"]
            continue
        start = pending_heading_start if pending_heading_start is not None else p["start"]
        pending_heading_start = None
        if p["end"] - start > max_size and p["kind"] != "article":
            for _, s, e in _split_long(text[p["start"]:p["end"]], p["start"], max_size):
                segments.append((p["article"], s, e))
        else:
            segments.append((p["article"], start, p["end"]))

    headings = {p["article"]: text[p["start"]:p["end"]] for p in paras if p["kind"] == "article"}

    chunks: list[Chunk] = []
    group: list[tuple[int | None, int, int]] = []

    def emit() -> None:
        if not group:
            return
        article = group[0][0]
        start, end = group[0][1], group[-1][2]
        body = text[start:end]
        heading = headings.get(article, "")
        prefix = None
        if heading and not body.lstrip().startswith(heading) and heading not in body:
            prefix = heading
            body = f"{heading}\n{body}"
        chunks.append(_make_chunk(strategy, page, len(chunks), body, start, end,
                                  context_prefix=prefix))
        group.clear()

    for seg in segments:
        if group and (seg[0] != group[0][0] or seg[2] - group[0][1] > max_size):
            emit()
        group.append(seg)
    emit()
    return chunks
