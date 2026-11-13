"""Read and normalize source data.

Corpus: Bộ luật Lao động 2019 (45/2019/QH14). The official PDF published on
chinhphu.vn is a scanned image (no text layer), so the text is taken from the
HTML full-text copy on pbgdpl.cantho.gov.vn and the PDF is only used as a
reference / to log why it was not used.

Because HTML has no physical pages, the text is divided (per GUIDE.md) into
sequential "units" of at least 1,500 normalized characters.  Units are cut only
at article ("Điều") boundaries and never cross a chapter ("Chương") boundary,
so every unit has exactly one trustworthy `chapter` value for filtering.
"""
from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import DocumentPage

MIN_UNIT_CHARS = 1500

_CHAPTER_RE = re.compile(r"^Chương ([IVXLC]+)$")
_SECTION_RE = re.compile(r"^Mục (\d+)\.\s*(.+)$")
_ARTICLE_RE = re.compile(r"^Điều (\d+)\.\s*(.*)$")


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
def normalize_text(text: str) -> str:
    """Normalize Unicode (NFC) and whitespace while keeping line structure.

    - NFC so that composed/decomposed Vietnamese diacritics compare equal.
    - Non-breaking / zero-width spaces are turned into normal spaces or removed.
    - Runs of spaces collapse to one; blank lines collapse; lines are stripped.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", " ").replace("​", "").replace("﻿", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def normalize_for_match(text: str) -> str:
    """Looser form used only for evidence matching (case/whitespace-insensitive)."""
    text = unicodedata.normalize("NFC", text).lower()
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- #
# PDF reader (used to check the official PDF; it turns out to be a scan)
# --------------------------------------------------------------------------- #
def load_pdf(path: Path, document_id: str, source: str) -> tuple[list[DocumentPage], list[dict[str, Any]]]:
    """Extract each physical page; return (pages, error_log).

    Pages with no extractable text are *not* silently dropped: they are logged
    with reason ``no_text_layer`` so a later OCR step (or a replacement source)
    can be documented.
    """
    import pymupdf  # imported lazily: only needed for PDFs

    pages: list[DocumentPage] = []
    errors: list[dict[str, Any]] = []
    with pymupdf.open(path) as doc:
        for index, page in enumerate(doc, start=1):
            text = normalize_text(page.get_text())
            if len(text) < 50:
                errors.append({"document_id": document_id, "page": index,
                               "error": "no_text_layer", "chars": len(text)})
                continue
            pages.append(DocumentPage(document_id, source, index, text, {"char_count": len(text)}))
    return pages, errors


# --------------------------------------------------------------------------- #
# HTML law reader
# --------------------------------------------------------------------------- #
@dataclass
class Paragraph:
    text: str
    kind: str                     # chapter | chapter_title | section | article | body
    chapter: str | None = None
    chapter_title: str | None = None
    section: str | None = None
    section_title: str | None = None
    article: int | None = None
    article_title: str | None = None


def _html_paragraphs(raw_html: str) -> list[str]:
    """Return normalized text of every <p> inside the law body."""
    start = raw_html.find("QUỐC HỘI")
    if start < 0:
        raise ValueError("Cannot find the law header ('QUỐC HỘI') in the HTML")
    body = raw_html[start - 2000:]
    out = []
    for fragment in re.findall(r"<p[^>]*>(.*?)</p>", body, flags=re.S):
        text = re.sub(r"<br\s*/?>", "\n", fragment)
        text = html.unescape(re.sub(r"<[^>]+>", "", text))
        text = normalize_text(text)
        if text:
            out.append(text)
    return out


def parse_law_html(path: Path) -> tuple[list[Paragraph], dict[str, Any]]:
    """Parse the law into structured paragraphs (chapter/section/article aware).

    The preamble (header, "Căn cứ ...") before "Chương I" and the signature /
    site footer after Điều 220 are excluded and reported in the log.
    """
    raw = path.read_text(encoding="utf-8")
    texts = _html_paragraphs(raw)

    first = next(i for i, t in enumerate(texts) if t == "Chương I")
    last = next(i for i, t in enumerate(texts) if t.startswith("Bộ luật này được Quốc hội"))
    log = {
        "html_paragraphs_total": len(texts),
        "excluded_preamble": texts[:first],
        "excluded_trailer": texts[last + 1:],
    }
    texts = texts[first:last + 1]

    paragraphs: list[Paragraph] = []
    chapter = chapter_title = section = section_title = None
    article = article_title = None
    expect_chapter_title = False
    for text in texts:
        if expect_chapter_title:
            chapter_title = text
            expect_chapter_title = False
            paragraphs.append(Paragraph(text, "chapter_title", chapter, chapter_title))
            continue
        if m := _CHAPTER_RE.match(text):
            chapter, chapter_title = m.group(1), None
            section = section_title = None
            article = article_title = None
            expect_chapter_title = True
            paragraphs.append(Paragraph(text, "chapter", chapter))
            continue
        if m := _SECTION_RE.match(text):
            section, section_title = m.group(1), m.group(2).strip()
            article = article_title = None
            paragraphs.append(Paragraph(text, "section", chapter, chapter_title, section, section_title))
            continue
        kind = "body"
        if m := _ARTICLE_RE.match(text):
            article, article_title = int(m.group(1)), m.group(2).strip()
            kind = "article"
        paragraphs.append(Paragraph(text, kind, chapter, chapter_title, section, section_title,
                                    article, article_title))

    articles = sorted({p.article for p in paragraphs if p.article is not None})
    if articles != list(range(1, 221)):
        missing = sorted(set(range(1, 221)) - set(articles))
        raise ValueError(f"Article sequence broken; missing: {missing[:20]}")
    log["articles_found"] = len(articles)
    log["chapters_found"] = len({p.chapter for p in paragraphs})
    return paragraphs, log


def build_units(paragraphs: list[Paragraph], document_id: str, source: str,
                min_chars: int = MIN_UNIT_CHARS) -> list[DocumentPage]:
    """Group whole articles into sequential units of >= ``min_chars`` characters.

    Chapter/section headings attach to the article that follows them.  A unit
    never crosses a chapter; if a chapter's last group is shorter than
    ``min_chars`` it is merged into the previous unit of the same chapter.  A
    chapter whose entire text is shorter than ``min_chars`` gives one short unit,
    which is reported (``counts_toward_30``=False) rather than padded.
    """
    # 1. blocks = [headings..., article heading, article body...]
    blocks: list[list[Paragraph]] = []
    pending: list[Paragraph] = []
    for p in paragraphs:
        if p.kind in ("chapter", "chapter_title", "section"):
            pending.append(p)
        elif p.kind == "article":
            blocks.append(pending + [p])
            pending = []
        else:
            blocks[-1].append(p)

    # 2. group blocks by chapter, then into units
    def joined_len(group: list[list[Paragraph]]) -> int:
        """Exact length of the unit text ("\\n".join of all paragraphs)."""
        texts = [p.text for block in group for p in block]
        return sum(map(len, texts)) + max(len(texts) - 1, 0)

    groups: list[list[list[Paragraph]]] = []
    current: list[list[Paragraph]] = []
    current_chapter = None

    def merge_short_tail() -> None:
        nonlocal current
        if (current and joined_len(current) < min_chars and groups
                and groups[-1][0][-1].chapter == current_chapter):
            groups[-1].extend(current)
            current = []

    def flush() -> None:
        nonlocal current
        if current:
            groups.append(current)
        current = []

    for block in blocks:
        chapter = block[-1].chapter
        if chapter != current_chapter:
            merge_short_tail()
            flush()
            current_chapter = chapter
        current.append(block)
        if joined_len(current) >= min_chars:
            flush()
    merge_short_tail()
    flush()

    # 3. materialize units with paragraph offsets for traceability
    units: list[DocumentPage] = []
    for number, group in enumerate(groups, start=1):
        paras = [p for block in group for p in block]
        offsets, pieces, pos = [], [], 0
        for p in paras:
            offsets.append({"start": pos, "end": pos + len(p.text), "kind": p.kind,
                            "article": p.article, "section": p.section})
            pieces.append(p.text)
            pos += len(p.text) + 1
        text = "\n".join(pieces)
        body_articles = sorted({p.article for p in paras if p.article is not None})
        chapter_p = paras[-1]
        units.append(DocumentPage(
            document_id=document_id,
            source=source,
            page=number,
            text=text,
            metadata={
                "unit_id": f"{document_id}-u{number:03d}",
                "chapter": chapter_p.chapter,
                "chapter_title": chapter_p.chapter_title,
                "sections": sorted({p.section for p in paras if p.section}, key=int),
                "articles": body_articles,
                "article_range": f"{body_articles[0]}-{body_articles[-1]}",
                "char_count": len(text),
                "counts_toward_30": len(text) >= min_chars,
                "paragraphs": offsets,
            },
        ))
    return units
