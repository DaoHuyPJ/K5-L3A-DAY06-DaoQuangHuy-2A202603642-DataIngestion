"""Optional extension: hybrid retrieval = dense (Chroma) + BM25, fused with RRF.

Same chunks, metadata and filters as strategy C; only the ranking changes, so
"C vs C+hybrid" isolates the effect of adding a keyword signal.

BM25 is implemented here (Okapi BM25, k1=1.5, b=0.75) on syllable tokens:
Vietnamese words are space-separated syllables, so ``\\w+`` after NFC +
lowercase gives one token per syllable.  No stop-word list and no word
segmentation — a known limitation.

Fusion (weighted Reciprocal Rank Fusion):
    score(d) = 1/(k + rank_dense(d)) + w * 1/(k + rank_bm25(d)),  k = 60
over the top-N (N=20) candidates of each list; w is the only tuned parameter and
is chosen on the dev split.  w = 0 reproduces strategy C exactly.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np

from .ingestion import normalize_for_match

RRF_K = 60
CANDIDATES = 20


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", normalize_for_match(text))


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in docs]
        self.lengths = np.array([sum(c.values()) for c in self.docs], dtype=np.float64)
        self.avgdl = float(self.lengths.mean())
        df: Counter[str] = Counter()
        for c in self.docs:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs))
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avgdl)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([c.get(term, 0) for c in self.docs], dtype=np.float64)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out


class HybridRetriever:
    """Wraps the B collection + an in-memory BM25 index over the same chunks."""

    def __init__(self, store, chunks: list[dict[str, Any]], index: str = "B") -> None:
        self.store = store
        self.index = index
        self.chunks = chunks
        self.bm25 = BM25([c["text"] for c in chunks])
        self.by_id = {c["chunk_id"]: c for c in chunks}

    def _row(self, chunk: dict[str, Any]) -> dict[str, Any]:
        m = chunk["metadata"]
        return {"chunk_id": chunk["chunk_id"], "document_id": chunk["document_id"],
                "source": chunk["source"], "page": chunk["page"], "chapter": m["chapter"],
                "chapter_title": m.get("chapter_title"), "section": m.get("section"),
                "articles": m["articles"], "char_start": m["char_start"], "char_end": m["char_end"],
                "text": chunk["text"]}

    def search(self, query: str, query_vector: list[float], weight: float, top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        dense = self.store.search(self.index, query_vector, top_k=CANDIDATES, filters=filters)
        dense_rank = {r["chunk_id"]: i for i, r in enumerate(dense, start=1)}

        allowed = np.array([all(c["metadata"].get(k) == v for k, v in (filters or {}).items())
                            for c in self.chunks])
        bm = np.where(allowed, self.bm25.scores(query), -np.inf)
        order = [i for i in np.argsort(-bm)[:CANDIDATES] if np.isfinite(bm[i]) and bm[i] > 0]
        bm_rank = {self.chunks[i]["chunk_id"]: r for r, i in enumerate(order, start=1)}

        fused: dict[str, float] = {}
        for cid in set(dense_rank) | set(bm_rank):
            s = 0.0
            if cid in dense_rank:
                s += 1.0 / (RRF_K + dense_rank[cid])
            if cid in bm_rank:
                s += weight / (RRF_K + bm_rank[cid])
            fused[cid] = s
        # ties broken by dense rank, then chunk id, for determinism
        ranked = sorted(fused, key=lambda c: (-fused[c], dense_rank.get(c, 10**6), c))[:top_k]
        rows = []
        for rank, cid in enumerate(ranked, start=1):
            row = self._row(self.by_id[cid])
            row.update({"rank": rank, "score": round(fused[cid], 6),
                        "dense_rank": dense_rank.get(cid), "bm25_rank": bm_rank.get(cid)})
            rows.append(row)
        return rows
