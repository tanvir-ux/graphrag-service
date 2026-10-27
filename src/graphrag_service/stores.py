"""Vector stores behind one small interface: a local numpy store and a Pinecone adapter."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class Hit:
    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(Protocol):
    def upsert(
        self, ids: list[str], vectors: list[list[float]], metadata: list[dict[str, Any]]
    ) -> None: ...

    def query(self, vector: list[float], top_k: int) -> list[Hit]: ...


class LocalVectorStore:
    """Exact cosine search with numpy. Vectors are assumed L2-normalised."""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._meta: list[dict[str, Any]] = []
        self._matrix = np.zeros((0, 0))

    def upsert(
        self, ids: list[str], vectors: list[list[float]], metadata: list[dict[str, Any]]
    ) -> None:
        index = {i: n for n, i in enumerate(self._ids)}
        for id_, vec, meta in zip(ids, vectors, metadata, strict=True):
            row = np.asarray(vec, dtype=np.float64)
            if id_ in index:
                self._matrix[index[id_]] = row
                self._meta[index[id_]] = meta
            else:
                self._matrix = row[None, :] if not self._ids else np.vstack([self._matrix, row])
                index[id_] = len(self._ids)
                self._ids.append(id_)
                self._meta.append(meta)

    def query(self, vector: list[float], top_k: int) -> list[Hit]:
        if not self._ids:
            return []
        scores = self._matrix @ np.asarray(vector, dtype=np.float64)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [Hit(self._ids[i], float(scores[i]), self._meta[i]) for i in order]


class PineconeVectorStore:
    """Adapter for a Pinecone index.

    Pass an `index` object (anything with `upsert` and `query`, for example `pc.Index(name)`),
    or call `PineconeVectorStore.from_env()` which needs PINECONE_API_KEY and PINECONE_INDEX.
    Only the unit tests with a fake index were run for this repository; the live service was
    not called because no Pinecone key was available when this was written.
    """

    def __init__(self, index: Any, namespace: str = "graphrag-demo") -> None:
        self.index = index
        self.namespace = namespace

    @classmethod
    def from_env(cls) -> PineconeVectorStore:
        from pinecone import Pinecone

        client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        return cls(
            client.Index(os.environ["PINECONE_INDEX"]),
            os.getenv("PINECONE_NAMESPACE", "graphrag-demo"),
        )

    def upsert(
        self, ids: list[str], vectors: list[list[float]], metadata: list[dict[str, Any]]
    ) -> None:
        records = [
            {"id": i, "values": v, "metadata": m}
            for i, v, m in zip(ids, vectors, metadata, strict=True)
        ]
        self.index.upsert(vectors=records, namespace=self.namespace)

    def query(self, vector: list[float], top_k: int) -> list[Hit]:
        res = self.index.query(
            vector=vector, top_k=top_k, include_metadata=True, namespace=self.namespace
        )
        matches = res["matches"] if isinstance(res, dict) else res.matches
        out = []
        for m in matches:
            get = m.get if isinstance(m, dict) else lambda k, d=None, m=m: getattr(m, k, d)
            out.append(Hit(get("id"), float(get("score")), dict(get("metadata") or {})))
        return out


def get_vector_store() -> VectorStore:
    """Pinecone only when PINECONE_API_KEY and PINECONE_INDEX exist, otherwise local."""
    if os.getenv("PINECONE_API_KEY") and os.getenv("PINECONE_INDEX"):
        return PineconeVectorStore.from_env()
    return LocalVectorStore()
