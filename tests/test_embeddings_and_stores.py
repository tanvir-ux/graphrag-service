import math
from types import SimpleNamespace

from graphrag_service.embeddings import (
    HashingEmbedder,
    LangChainHashingEmbeddings,
    LlamaIndexHashingEmbedding,
)
from graphrag_service.stores import (
    Hit,
    LocalVectorStore,
    PineconeVectorStore,
    get_vector_store,
)


def test_hashing_embedder_is_deterministic_and_normalised():
    emb = HashingEmbedder(dim=128)
    a, b = emb.embed(["Maria Chen founded Helios Robotics"] * 2)
    assert a == b
    assert math.isclose(sum(x * x for x in a), 1.0, rel_tol=1e-9)
    assert len(a) == 128


def test_similar_text_scores_higher_than_unrelated():
    emb = HashingEmbedder()
    q, close, far = emb.embed(
        ["who founded Helios Robotics", "Maria Chen founded Helios Robotics", "Seattle rain"]
    )
    dot = lambda x, y: sum(i * j for i, j in zip(x, y, strict=True))  # noqa: E731
    assert dot(q, close) > dot(q, far)


def test_framework_wrappers_share_the_same_vectors():
    text = "Voltix Batteries supplies Helios Robotics"
    base = HashingEmbedder(512).embed([text])[0]
    assert LangChainHashingEmbeddings().embed_query(text) == base
    assert LangChainHashingEmbeddings().embed_documents([text]) == [base]
    assert LlamaIndexHashingEmbedding(dim=512).get_text_embedding(text) == base
    assert LlamaIndexHashingEmbedding(dim=512).get_query_embedding(text) == base


def test_local_store_ranks_and_overwrites():
    store = LocalVectorStore()
    assert store.query([1.0, 0.0], 3) == []
    store.upsert(["a", "b"], [[1.0, 0.0], [0.0, 1.0]], [{"n": 1}, {"n": 2}])
    hits = store.query([0.9, 0.1], 2)
    assert [h.id for h in hits] == ["a", "b"]
    store.upsert(["a"], [[0.0, 1.0]], [{"n": 3}])
    hits = store.query([0.0, 1.0], 2)
    assert {h.id for h in hits} == {"a", "b"} and hits[0].score == 1.0
    assert [h.metadata["n"] for h in hits if h.id == "a"] == [3]


class FakeIndex:
    """Stands in for a pinecone.Index; records calls and returns canned matches."""

    def __init__(self, as_dict: bool) -> None:
        self.as_dict, self.calls = as_dict, []

    def upsert(self, **kwargs):
        self.calls.append(("upsert", kwargs))

    def query(self, **kwargs):
        self.calls.append(("query", kwargs))
        matches = [{"id": "d01", "score": 0.9, "metadata": {"doc_id": "d01"}}]
        if self.as_dict:
            return {"matches": matches}
        return SimpleNamespace(matches=[SimpleNamespace(**m) for m in matches])


def test_pinecone_adapter_with_fake_index():
    for as_dict in (True, False):
        index = FakeIndex(as_dict)
        store = PineconeVectorStore(index, namespace="ns")
        store.upsert(["d01"], [[0.1, 0.2]], [{"doc_id": "d01"}])
        hits = store.query([0.1, 0.2], top_k=2)
        assert hits == [Hit("d01", 0.9, {"doc_id": "d01"})]
        kind, kwargs = index.calls[0]
        assert kind == "upsert" and kwargs["namespace"] == "ns"
        assert kwargs["vectors"] == [
            {"id": "d01", "values": [0.1, 0.2], "metadata": {"doc_id": "d01"}}
        ]
        assert index.calls[1][1]["top_k"] == 2 and index.calls[1][1]["include_metadata"]


def test_vector_store_selection_follows_environment(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    monkeypatch.delenv("PINECONE_INDEX", raising=False)
    assert isinstance(get_vector_store(), LocalVectorStore)
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    assert isinstance(get_vector_store(), LocalVectorStore)  # index name still missing
    created = {}

    class FakePinecone:
        def __init__(self, api_key):
            created["key"] = api_key

        def Index(self, name):  # noqa: N802 - mirrors the Pinecone client
            created["index"] = name
            return FakeIndex(True)

    import pinecone

    monkeypatch.setattr(pinecone, "Pinecone", FakePinecone)
    monkeypatch.setenv("PINECONE_INDEX", "helios")
    assert isinstance(get_vector_store(), PineconeVectorStore)
    assert created == {"key": "test-key", "index": "helios"}
