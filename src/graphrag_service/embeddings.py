"""Deterministic hashing embeddings (no model download) and framework wrappers.

`HashingEmbedder` is a bag-of-words feature hasher with sublinear term frequency. It is a
stand-in for a neural embedding model so the whole project runs offline and reproducibly.
The same vectors feed LlamaIndex, LangChain and the native pipeline.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Protocol

import numpy as np
from langchain_core.embeddings import Embeddings
from llama_index.core.base.embeddings.base import BaseEmbedding

from .text import tokens


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _one(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float64)
        for tok, count in Counter(tokens(text)).items():
            digest = hashlib.md5(tok.encode()).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[index] += sign * (1.0 + math.log(count))
        norm = float(np.linalg.norm(vec))
        return (vec / norm if norm else vec).tolist()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]


class LangChainHashingEmbeddings(Embeddings):
    """LangChain `Embeddings` adapter around any `Embedder`."""

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or HashingEmbedder()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embedder.embed([text])[0]


class LlamaIndexHashingEmbedding(BaseEmbedding):
    """LlamaIndex `BaseEmbedding` adapter around the hashing embedder."""

    dim: int = 512

    def _vec(self, text: str) -> list[float]:
        return HashingEmbedder(self.dim).embed([text])[0]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._vec(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._vec(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._vec(text)
