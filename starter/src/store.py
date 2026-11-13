"""Persistent vector store: Chroma (local, on-disk), cosine metric, metadata filters.

Why Chroma instead of the starter's PostgreSQL + pgvector: the lab machine has
no Docker and not enough disk for Docker Desktop + WSL2.  Chroma's
PersistentClient stores data on disk (survives restarts), supports vector
search and `where` metadata filters, which is what GUIDE.md requires.

One collection per index ("bld2019_A", "bld2019_B"); strategy C searches the B
collection with a `where` filter.  Chroma's search is approximate (HNSW), so
``exact_search`` provides a brute-force reference used to verify that the ANN
top-k equals the exact top-k.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .models import Chunk

ALLOWED_FILTERS = {"chapter", "document_id", "section"}
COLLECTION_PREFIX = "bld2019_"


def _where(strategy_filters: dict[str, Any] | None) -> dict[str, Any] | None:
    items = list((strategy_filters or {}).items())
    for key, _ in items:
        if key not in ALLOWED_FILTERS:
            raise ValueError(f"filter '{key}' not allowed; allowed: {sorted(ALLOWED_FILTERS)}")
    if not items:
        return None
    if len(items) == 1:
        k, v = items[0]
        return {k: {"$eq": v}}
    return {"$and": [{k: {"$eq": v}} for k, v in items]}


class ChromaStore:
    def __init__(self, path: str | Path, dimensions: int, model_name: str) -> None:
        import chromadb

        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.dimensions = dimensions
        self.model_name = model_name
        self.client = chromadb.PersistentClient(path=str(self.path))

    # ------------------------------------------------------------------ schema
    def collection(self, strategy: str):
        """Get/create the collection; refuse to mix models or dimensions."""
        col = self.client.get_or_create_collection(
            name=f"{COLLECTION_PREFIX}{strategy.strip('_')}",
            embedding_function=None,            # vectors are always supplied by our embedder
            metadata={"hnsw:space": "cosine", "embedding_model": self.model_name,
                      "dimensions": self.dimensions},
        )
        meta = col.metadata or {}
        if meta.get("embedding_model") != self.model_name or meta.get("dimensions") != self.dimensions:
            raise RuntimeError(f"Collection {col.name} was built with {meta.get('embedding_model')} "
                               f"({meta.get('dimensions')}d); current model is {self.model_name} "
                               f"({self.dimensions}d). Delete the collection instead of mixing models.")
        if meta.get("hnsw:space") != "cosine":
            raise RuntimeError(f"Collection {col.name} does not use cosine distance")
        return col

    def create_schema(self) -> None:
        """Chroma is schemaless; nothing to do beyond creating collections lazily."""

    def versions(self) -> dict[str, str]:
        import chromadb
        return {"chromadb": chromadb.__version__, "path": str(self.path)}

    # ------------------------------------------------------------------ write
    @staticmethod
    def _metadata(chunk: Chunk) -> dict[str, Any]:
        m = chunk.metadata
        return {   # Chroma metadata values must be scalars and non-null
            "document_id": chunk.document_id,
            "source": chunk.source,
            "page": int(chunk.page),
            "unit_id": m["unit_id"],
            "chapter": m["chapter"],
            "chapter_title": m.get("chapter_title") or "",
            "section": m.get("section") or "",
            "articles": ",".join(map(str, m["articles"])),
            "char_start": int(m["char_start"]),
            "char_end": int(m["char_end"]),
            "content_hash": m["content_hash"],
            "context_prefix": m.get("context_prefix") or "",
        }

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> dict[str, int]:
        """Idempotent write keyed by chunk_id.

        - Same chunk_id + same content_hash -> skipped (no duplicate).
        - Same chunk_id + new content_hash  -> updated.
        - chunk_ids of the same documents that are no longer produced are
          deleted, so a changed source leaves no stale chunks.
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        if any(len(v) != self.dimensions for v in vectors):
            raise ValueError(f"vector dimension != {self.dimensions}")
        ids = [c.chunk_id for c in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate chunk_id in input")

        col = self.collection(strategy)
        doc_ids = sorted({c.document_id for c in chunks})
        existing: dict[str, str] = {}
        for doc in doc_ids:
            got = col.get(where={"document_id": {"$eq": doc}}, include=["metadatas"])
            existing.update({i: m["content_hash"] for i, m in zip(got["ids"], got["metadatas"])})

        stats = {"inserted": 0, "updated": 0, "unchanged": 0, "deleted": 0}
        todo = []
        for chunk, vector in zip(chunks, vectors):
            old = existing.get(chunk.chunk_id)
            if old == chunk.metadata["content_hash"]:
                stats["unchanged"] += 1
                continue
            stats["updated" if old else "inserted"] += 1
            todo.append((chunk, vector))
        for start in range(0, len(todo), 256):
            batch = todo[start:start + 256]
            col.upsert(ids=[c.chunk_id for c, _ in batch],
                       embeddings=[v for _, v in batch],
                       documents=[c.text for c, _ in batch],
                       metadatas=[self._metadata(c) for c, _ in batch])
        stale = sorted(set(existing) - set(ids))
        if stale:
            col.delete(ids=stale)
        stats["deleted"] = len(stale)
        return stats

    def existing_hashes(self, strategy: str) -> dict[str, str]:
        got = self.collection(strategy).get(include=["metadatas"])
        return {i: m["content_hash"] for i, m in zip(got["ids"], got["metadatas"])}

    def count(self, strategy: str) -> dict[str, int]:
        hashes = self.existing_hashes(strategy)
        return {"rows": self.collection(strategy).count(), "distinct_chunk_ids": len(hashes),
                "distinct_hashes": len(set(hashes.values()))}

    def delete_strategy(self, strategy: str) -> int:
        col = self.collection(strategy)
        n = col.count()
        self.client.delete_collection(col.name)
        return n

    # ------------------------------------------------------------------ read
    @staticmethod
    def _row(chunk_id: str, text: str, meta: dict[str, Any], score: float, rank: int) -> dict[str, Any]:
        return {
            "rank": rank,
            "chunk_id": chunk_id,
            "document_id": meta["document_id"],
            "source": meta["source"],
            "page": meta["page"],
            "chapter": meta["chapter"],
            "chapter_title": meta["chapter_title"],
            "section": meta["section"] or None,
            "articles": [int(a) for a in meta["articles"].split(",") if a],
            "char_start": meta["char_start"],
            "char_end": meta["char_end"],
            "text": text,
            "score": round(float(score), 6),
        }

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """HNSW cosine search; the metadata filter is applied by Chroma (`where`).

        score = 1 - cosine distance (vectors are L2-normalized).
        """
        if len(query_vector) != self.dimensions:
            raise ValueError("query vector dimension mismatch")
        res = self.collection(strategy).query(
            query_embeddings=[query_vector], n_results=top_k, where=_where(filters),
            include=["documents", "metadatas", "distances"])
        return [self._row(i, d, m, 1.0 - dist, r)
                for r, (i, d, m, dist) in enumerate(zip(res["ids"][0], res["documents"][0],
                                                        res["metadatas"][0], res["distances"][0]), start=1)]

    def exact_search(self, strategy: str, query_vector: list[float], top_k: int = 3,
                     filters: dict[str, Any] | None = None) -> list[str]:
        """Brute-force reference (numpy dot product over all stored vectors that
        pass the filter).  Used only to verify the ANN results."""
        got = self.collection(strategy).get(where=_where(filters), include=["embeddings"])
        if not got["ids"]:
            return []
        matrix = np.asarray(got["embeddings"], dtype=np.float32)
        scores = matrix @ np.asarray(query_vector, dtype=np.float32)
        order = np.argsort(-scores)[:top_k]
        return [got["ids"][i] for i in order]
