"""Plain vector RAG: the native pipeline, plus LlamaIndex and LangChain implementations.

All three retrieve whole documents (each is only a few sentences) with the same embeddings
and the same top_k, so they should behave almost identically. They exist to show the same
retrieval step expressed in each framework and behind the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from langchain_core.documents import Document as LCDocument
from langchain_core.vectorstores import InMemoryVectorStore
from llama_index.core import Document as LIDocument
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

from ..corpus import DOCS, Document
from ..embeddings import (
    Embedder,
    HashingEmbedder,
    LangChainHashingEmbeddings,
    LlamaIndexHashingEmbedding,
)
from ..stores import LocalVectorStore, VectorStore
from ..trace import traced


@dataclass
class Retrieved:
    doc_id: str
    text: str
    score: float


class VectorRetriever(Protocol):
    name: str

    def retrieve(self, question: str, top_k: int = 3) -> list[Retrieved]: ...


class NativeRetriever:
    name = "vector_native"

    def __init__(
        self,
        docs: list[Document] | None = None,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.docs = {d.id: d for d in (docs or DOCS)}
        self.embedder = embedder or HashingEmbedder()
        self.store = store or LocalVectorStore()
        ids = list(self.docs)
        self.store.upsert(
            ids,
            self.embedder.embed([self.docs[i].text for i in ids]),
            [{"doc_id": i, "title": self.docs[i].title} for i in ids],
        )

    @traced("retrieve.native", kind="retrieval")
    def retrieve(self, question: str, top_k: int = 3) -> list[Retrieved]:
        hits = self.store.query(self.embedder.embed([question])[0], top_k)
        return [
            Retrieved(
                h.metadata.get("doc_id", h.id),
                self.docs[h.metadata.get("doc_id", h.id)].text,
                h.score,
            )
            for h in hits
        ]


class LlamaIndexRetriever:
    name = "vector_llamaindex"

    def __init__(self, docs: list[Document] | None = None, dim: int = 512) -> None:
        documents = [
            LIDocument(text=d.text, metadata={"doc_id": d.id, "title": d.title}, doc_id=d.id)
            for d in (docs or DOCS)
        ]
        self.index = VectorStoreIndex.from_documents(
            documents,
            embed_model=LlamaIndexHashingEmbedding(dim=dim),
            transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=0)],
        )

    @traced("retrieve.llamaindex", kind="retrieval")
    def retrieve(self, question: str, top_k: int = 3) -> list[Retrieved]:
        nodes = self.index.as_retriever(similarity_top_k=top_k).retrieve(question)
        return [
            Retrieved(n.node.metadata["doc_id"], n.node.get_content(), float(n.score or 0.0))
            for n in nodes
        ]


class LangChainRetriever:
    name = "vector_langchain"

    def __init__(self, docs: list[Document] | None = None, dim: int = 512) -> None:
        self.store = InMemoryVectorStore(LangChainHashingEmbeddings(HashingEmbedder(dim)))
        self.store.add_documents(
            [
                LCDocument(
                    page_content=d.text, metadata={"doc_id": d.id, "title": d.title}, id=d.id
                )
                for d in (docs or DOCS)
            ]
        )

    @traced("retrieve.langchain", kind="retrieval")
    def retrieve(self, question: str, top_k: int = 3) -> list[Retrieved]:
        pairs = self.store.similarity_search_with_score(question, k=top_k)
        return [Retrieved(d.metadata["doc_id"], d.page_content, float(s)) for d, s in pairs]
