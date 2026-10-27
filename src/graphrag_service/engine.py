"""The query engine: four retrieval modes behind one call, sharing one generator."""

from __future__ import annotations

import time
from typing import Any

from .corpus import DOCS
from .graph.build import KnowledgeGraph
from .graph.retrieve import graph_retrieve
from .llm import LLM, MockLLM
from .rag.vector import LangChainRetriever, LlamaIndexRetriever, NativeRetriever
from .trace import RECORDER

MODES = ("vector_native", "vector_llamaindex", "vector_langchain", "graphrag")
TOP_K = 3


class Engine:
    def __init__(self, llm: LLM | None = None, store=None, embedder=None) -> None:
        self.llm = llm or MockLLM()
        self.kg = KnowledgeGraph.build()
        self.retrievers = {
            "vector_native": NativeRetriever(store=store, embedder=embedder),
            "vector_llamaindex": LlamaIndexRetriever(),
            "vector_langchain": LangChainRetriever(),
        }

    def query(
        self, question: str, mode: str = "graphrag", llm: LLM | None = None
    ) -> dict[str, Any]:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose one of {', '.join(MODES)}")
        llm = llm or self.llm
        started = time.perf_counter()
        RECORDER.add("query", mode, input=question)
        graph: dict[str, Any] = {"nodes": [], "edges": []}
        evidence: list[str] = []
        strategy = "vector"
        if mode == "graphrag":
            res = graph_retrieve(self.kg, question)
            contexts = res.contexts
            evidence = res.evidence
            strategy = res.strategy
            graph = {
                "nodes": sorted(res.nodes),
                "edges": [
                    {"source": s, "target": t, "relation": r} for s, t, r in sorted(res.edges)
                ],
            }
        else:
            contexts = [
                {"doc_id": r.doc_id, "text": r.text, "score": round(r.score, 3), "source": "vector"}
                for r in self.retrievers[mode].retrieve(question, TOP_K)
            ]
        answer = llm.generate(question, [c["text"] for c in contexts])
        return {
            "question": question,
            "mode": mode,
            "strategy": strategy,
            "llm": llm.name,
            "answer": answer,
            "contexts": contexts,
            "evidence": evidence,
            "graph": graph,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    def doc_count(self) -> int:
        return len(DOCS)
