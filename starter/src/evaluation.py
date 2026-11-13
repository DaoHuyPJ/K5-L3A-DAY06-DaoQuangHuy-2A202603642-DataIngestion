"""Evaluate retrieved evidence and preserve per-query traces.

Definition (GUIDE.md §4.1): a retrieved chunk is a *hit* when it belongs to a
gold source/page AND contains the **whole** gold evidence quote (after the
same case/whitespace/Unicode normalization on both sides).  Any documented
alternative gold (``gold_alternatives``) is accepted as well.

This module never infers coherence or grounding labels; those are manual.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from .ingestion import normalize_for_match


def gold_options(question: dict[str, Any]) -> list[dict[str, Any]]:
    """Primary gold + documented alternatives as a flat list of (source, pages, quote)."""
    pages = question["gold_page"]
    options = [{
        "source_id": question["gold_source_id"],
        "pages": pages if isinstance(pages, list) else [pages],
        "evidence_quote": question["evidence_quote"],
    }]
    for alt in question.get("gold_alternatives", []):
        alt_pages = alt["gold_page"]
        options.append({
            "source_id": alt["gold_source_id"],
            "pages": alt_pages if isinstance(alt_pages, list) else [alt_pages],
            "evidence_quote": alt["evidence_quote"],
        })
    return options


def chunk_is_hit(question: dict[str, Any], hit: dict[str, Any]) -> bool:
    text = normalize_for_match(hit["text"])
    for gold in gold_options(question):
        if (hit["document_id"] == gold["source_id"] and hit["page"] in gold["pages"]
                and normalize_for_match(gold["evidence_quote"]) in text):
            return True
    return False


def _partial_evidence(question: dict[str, Any], hit: dict[str, Any]) -> bool:
    """True when the chunk is on a gold page and holds a large piece (>=40 chars
    prefix or suffix) of the evidence but not all of it -> likely split evidence."""
    text = normalize_for_match(hit["text"])
    for gold in gold_options(question):
        if hit["document_id"] != gold["source_id"] or hit["page"] not in gold["pages"]:
            continue
        quote = normalize_for_match(gold["evidence_quote"])
        if quote in text:
            continue
        k = min(40, len(quote) // 2)
        if quote[:k] in text or quote[-k:] in text:
            return True
    return False


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]],
                   strategy: str, applied_filter: dict[str, Any] | None) -> dict[str, Any]:
    """Score one query and return a full trace record."""
    flags = [chunk_is_hit(question, h) for h in hits]
    partial = [(not f) and _partial_evidence(question, h) for f, h in zip(flags, hits)]
    # Compliance is measured against the filter the *question* declares, for
    # every strategy: for A/B (no filter applied) it shows how often an
    # unfiltered search leaves the requested scope.
    question_filter = question.get("filter") or None
    filter_ok = None
    if question_filter:
        filter_ok = bool(hits) and all(
            all(h.get(k) == v for k, v in question_filter.items()) for h in hits)
    return {
        "strategy": strategy,
        "id": question["id"],
        "split": question["split"],
        "type": question.get("type"),
        "question": question["question"],
        "expected_answer": question["expected_answer"],
        "gold": gold_options(question),
        "question_filter": question.get("filter"),
        "applied_filter": applied_filter,
        "filter_compliant": filter_ok,
        "hit_at_1": bool(flags[:1] and flags[0]),
        "hit_at_3": any(flags[:3]),
        "first_hit_rank": next((i + 1 for i, f in enumerate(flags) if f), None),
        "evidence_split_suspected": any(partial[:3]) and not any(flags[:3]),
        "retrieved": [
            {
                "rank": h["rank"],
                "chunk_id": h["chunk_id"],
                "score": h["score"],
                "document_id": h["document_id"],
                "source": h["source"],
                "page": h["page"],
                "chapter": h["chapter"],
                "section": h.get("section"),
                "articles": h["articles"],
                "char_span": [h["char_start"], h["char_end"]],
                "is_evidence_hit": f,
                "partial_evidence": p,
                "text": h["text"],
            }
            for h, f, p in zip(hits, flags, partial)
        ],
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1 / Evidence Hit@3 by strategy and split, as n/N."""
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for t in traces:
        buckets[(t["strategy"], t["split"])].append(t)
        buckets[(t["strategy"], "all")].append(t)
    out: dict[str, Any] = {}
    for (strategy, split), items in sorted(buckets.items()):
        n = len(items)
        h1 = sum(t["hit_at_1"] for t in items)
        h3 = sum(t["hit_at_3"] for t in items)
        filtered = [t for t in items if t["question_filter"]]
        out.setdefault(strategy, {})[split] = {
            "N": n,
            "hit_at_1": h1,
            "hit_at_3": h3,
            "hit_at_1_rate": round(h1 / n, 4) if n else None,
            "hit_at_3_rate": round(h3 / n, 4) if n else None,
            "hit_at_1_str": f"{h1}/{n}",
            "hit_at_3_str": f"{h3}/{n}",
            "filter_queries": len(filtered),
            "filter_compliant": sum(bool(t["filter_compliant"]) for t in filtered),
            "evidence_split_suspected": [t["id"] for t in items if t["evidence_split_suspected"]],
            "missed_ids": [t["id"] for t in items if not t["hit_at_3"]],
        }
    return out
