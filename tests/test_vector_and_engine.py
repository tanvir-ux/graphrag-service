import pytest

from graphrag_service.corpus import DOCS
from graphrag_service.engine import MODES, TOP_K, Engine
from graphrag_service.rag.vector import NativeRetriever
from graphrag_service.stores import Hit


def test_all_vector_backends_return_top_k_documents(engine):
    for name in ("vector_native", "vector_llamaindex", "vector_langchain"):
        res = engine.query("Where is Helios Robotics headquartered?", name)
        assert len(res["contexts"]) == TOP_K
        assert res["contexts"][0]["doc_id"] == "d01"
        assert res["strategy"] == "vector"


def test_backends_agree_on_the_top_document(engine):
    for q in ("Who founded Nimbus Vision?", "Which university did Anika Rao attend?"):
        tops = {engine.query(q, m)["contexts"][0]["doc_id"] for m in MODES if m != "graphrag"}
        assert len(tops) == 1


def test_engine_rejects_unknown_mode(engine):
    with pytest.raises(ValueError, match="unknown mode"):
        engine.query("hello there", "nope")


def test_query_result_shape(engine):
    res = engine.query("Which university did the founder of Helios Robotics attend?", "graphrag")
    assert {"answer", "contexts", "evidence", "graph", "latency_ms", "llm"} <= set(res)
    assert "Stanford University" in res["answer"]
    assert res["graph"]["edges"] and res["graph"]["nodes"]


def test_vector_rag_misses_a_multi_hop_fact(engine):
    q = "Which university did the founder of the company that supplies Helios Robotics attend?"
    assert "Carnegie Mellon" not in engine.query(q, "vector_native")["answer"]
    assert "Carnegie Mellon" in engine.query(q, "graphrag")["answer"]


def test_native_retriever_works_with_any_vector_store():
    class Spy:
        def __init__(self):
            self.upserted = []

        def upsert(self, ids, vectors, metadata):
            self.upserted = list(ids)

        def query(self, vector, top_k):
            return [Hit("d05", 0.5, {"doc_id": "d05"})]

    spy = Spy()
    retriever = NativeRetriever(store=spy)
    assert spy.upserted == [d.id for d in DOCS]
    assert [r.doc_id for r in retriever.retrieve("anything")] == ["d05"]


def test_engine_is_deterministic():
    a, b = Engine(), Engine()
    q = "In which city is the company that Helios Robotics acquired based?"
    ra, rb = a.query(q, "graphrag"), b.query(q, "graphrag")
    assert ra["answer"] == rb["answer"] and ra["evidence"] == rb["evidence"]
