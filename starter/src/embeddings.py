"""Real local embedding model (Sentence Transformers). No mock fallback."""
from __future__ import annotations

from typing import Protocol

DEFAULT_MODEL = "intfloat/multilingual-e5-small"


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class E5Embedder:
    """Adapter for the multilingual E5 family.

    E5 is trained with asymmetric prefixes: documents are encoded as
    ``"passage: <text>"`` and queries as ``"query: <text>"``.  Vectors are
    L2-normalized so cosine distance in the store equals 1 - dot product.
    Inputs longer than the model's max_seq_length (512 tokens) are truncated by
    the tokenizer; ``truncated_count`` reports how many documents were affected.

    If the model cannot be loaded, the exception propagates — the pipeline must
    fail loudly instead of falling back to random / mock vectors.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None,
                 batch_size: int = 32, local_path: str | None = None) -> None:
        """``local_path``: optional folder holding a downloaded copy of
        ``model_name`` (with a REVISION file); otherwise loaded from the HF hub."""
        from pathlib import Path

        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.revision = None
        source = model_name
        if local_path:
            path = Path(local_path)
            if not (path / "modules.json").exists():
                raise FileNotFoundError(f"EMBEDDING_MODEL_PATH {path} does not contain a model")
            rev = path / "REVISION"
            self.revision = rev.read_text().strip() if rev.exists() else None
            source = str(path)
        self.model = SentenceTransformer(source, device=device)
        get_dim = getattr(self.model, "get_embedding_dimension", None) or \
            self.model.get_sentence_embedding_dimension
        dims = get_dim()
        if not dims:
            raise RuntimeError(f"Cannot determine embedding dimension for {model_name}")
        self.dimensions = int(dims)
        self.max_seq_length = int(self.model.max_seq_length)
        self.truncated_count = 0

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=len(texts) > 64)
        if vectors.shape != (len(texts), self.dimensions):
            raise RuntimeError(f"Unexpected embedding shape {vectors.shape}")
        return vectors.tolist()

    def count_truncated(self, texts: list[str]) -> int:
        tok = self.model.tokenizer
        return sum(len(tok(t, add_special_tokens=True)["input_ids"]) > self.max_seq_length for t in texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        self.truncated_count = self.count_truncated(prefixed)
        return self._encode(prefixed)

    def embed_query(self, question: str) -> list[float]:
        return self._encode([f"query: {question}"])[0]

    def describe(self) -> dict:
        import sentence_transformers
        import torch
        return {
            "provider": "local / sentence-transformers",
            "model": self.model_name,
            "revision": self.revision,
            "dimensions": self.dimensions,
            "max_seq_length": self.max_seq_length,
            "document_prefix": "passage: ",
            "query_prefix": "query: ",
            "normalized": True,
            "device": str(self.model.device),
            "sentence_transformers_version": sentence_transformers.__version__,
            "torch_version": torch.__version__,
            "cost_usd": 0.0,
        }


# Backwards-compatible name used by the starter skeleton
MyEmbedder = E5Embedder
