import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from graphrag_service import api as api_module
from graphrag_service.api import create_app
from graphrag_service.cli import main
from graphrag_service.evaluation.dataset import QUESTIONS
from graphrag_service.evaluation.harness import run_eval, score_one, to_markdown
from graphrag_service.llm import MockLLM

ROOT = Path(__file__).resolve().parent.parent


def test_dataset_is_well_formed():
    assert len(QUESTIONS) == 17 and len({q.id for q in QUESTIONS}) == 17
    cats = {q.category for q in QUESTIONS}
    assert cats == {"single_hop", "multi_hop", "global", "paraphrase"}
    assert all(q.gold_docs and q.gold_answers for q in QUESTIONS)


def test_score_one_counts_recall_and_all_fragments():
    q = QUESTIONS[2]  # two gold fragments, one gold doc
    res = {"answer": "Atlas Arm only", "contexts": [{"doc_id": "d01", "text": "x"}]}
    s = score_one(q, res)
    assert s["doc_recall"] == 1.0 and s["answer_correct"] is False
    res["answer"] = "atlas arm and orbit rover"
    assert score_one(q, res)["answer_correct"] is True


@pytest.fixture(scope="module")
def report(engine):
    return run_eval(engine, llm=MockLLM())


def test_eval_report_structure(report):
    assert report["n_questions"] == 17 and len(report["rows"]) == 17 * 4
    assert set(report["summary"]) == {
        "vector_native",
        "vector_llamaindex",
        "vector_langchain",
        "graphrag",
    }
    assert "| graphrag | multi_hop |" in to_markdown(report)


def test_graphrag_beats_vector_rag_on_multi_hop_but_not_on_paraphrases(report):
    s = report["summary"]
    assert (
        s["graphrag"]["multi_hop"]["answer_accuracy"]
        > s["vector_native"]["multi_hop"]["answer_accuracy"]
    )
    assert (
        s["graphrag"]["global"]["answer_accuracy"] > s["vector_native"]["global"]["answer_accuracy"]
    )
    # the honest limitation: rule-based GraphRAG is brittle on unseen phrasing
    assert (
        s["graphrag"]["paraphrase"]["doc_recall"] < s["vector_native"]["paraphrase"]["doc_recall"]
    )
    # and it uses far less context
    assert (
        s["graphrag"]["all"]["avg_context_chars"] < s["vector_native"]["all"]["avg_context_chars"]
    )


def test_framework_backends_have_matching_single_hop_quality(report):
    s = report["summary"]
    assert {s[m]["single_hop"]["answer_accuracy"] for m in s if m != "graphrag"} == {1.0}


@pytest.fixture()
def client():
    return TestClient(create_app())


def test_api_endpoints(client):
    assert client.get("/api/health").json()["status"] == "ok"
    ex = client.get("/api/examples").json()
    assert len(ex["questions"]) == 17 and len(ex["documents"]) == 15 and not ex["gemini_enabled"]
    g = client.get("/api/graph").json()
    assert len(g["nodes"]) == 26 and len(g["edges"]) == 30 and g["summaries"]
    res = client.post(
        "/api/query",
        json={"question": "Where is Mars Survey Institute located?", "mode": "graphrag"},
    ).json()
    assert "Pasadena" in res["answer"] and res["trace"]
    assert client.get("/").status_code == 200
    assert client.get("/recordings.json").status_code == 200


def test_api_validation(client):
    bad = client.post("/api/query", json={"question": "Where is Austin?", "mode": "nope"})
    assert bad.status_code == 422
    assert client.post("/api/query", json={"question": "hi"}).status_code == 422
    short = client.post("/api/query", json={"question": "x" * 301})
    assert short.status_code == 422


def test_gemini_mode_is_disabled_by_default(client, monkeypatch):
    monkeypatch.delenv("ENABLE_GEMINI_DEMO", raising=False)
    r = client.post("/api/query", json={"question": "Where is Austin?", "llm": "gemini"})
    assert r.status_code == 403


def test_gemini_mode_is_rate_limited_and_hides_provider_errors(monkeypatch):
    monkeypatch.setenv("ENABLE_GEMINI_DEMO", "1")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-test-key")
    monkeypatch.setenv("GEMINI_DEMO_PER_IP_HOURLY", "1")

    class Boom:
        name = "boom"

        def generate(self, question, contexts):
            raise RuntimeError("secret detail dummy-test-key")

    monkeypatch.setattr(api_module, "get_llm", lambda mode=None: Boom())
    c = TestClient(create_app())
    body = {"question": "Where is Mars Survey Institute located?", "llm": "gemini"}
    first = c.post("/api/query", json=body)
    assert first.status_code == 502 and "dummy-test-key" not in first.text
    assert c.post("/api/query", json=body).status_code == 429


def test_cli_ask_and_graph(capsys):
    assert main(["ask", "Where is Helios Robotics headquartered?"]) == 0
    out = capsys.readouterr().out
    assert "Austin" in out and "evidence:" in out
    assert main(["graph"]) == 0
    assert "26 entities, 30 relations" in capsys.readouterr().out
    assert main(["eval", "--limit", "3"]) == 0
    assert "| graphrag |" in capsys.readouterr().out


def test_docs_copy_matches_packaged_web_files():
    web = ROOT / "src" / "graphrag_service" / "web"
    for name in ("index.html", "recordings.json"):
        assert (ROOT / "docs" / name).read_bytes() == (web / name).read_bytes(), name
    assert (ROOT / "docs" / ".nojekyll").exists()


def test_recordings_cover_every_example_and_mode():
    rec = json.loads((ROOT / "docs" / "recordings.json").read_text())
    for q in rec["examples"]["questions"]:
        for mode in ("vector_native", "vector_llamaindex", "vector_langchain", "graphrag"):
            assert f"{mode}::{q['text']}" in rec["queries"]
    assert rec["eval"]["n_questions"] == 17 and rec["graph"]["nodes"]
