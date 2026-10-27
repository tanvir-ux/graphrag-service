import io
import json
from types import SimpleNamespace

import boto3
import httpx
import pytest
from botocore.response import StreamingBody
from botocore.stub import Stubber

from graphrag_service.adapters.azure_openai import AzureOpenAIEmbedder, AzureOpenAILLM
from graphrag_service.adapters.bedrock import BedrockEmbedder, BedrockLLM
from graphrag_service.llm import MockLLM, OpenAICompatLLM, build_prompt, get_llm

CONTEXTS = ["Maria Chen founded Helios Robotics.", "Maria Chen studied at Stanford University."]
QUESTION = "Which university did the founder of Helios Robotics attend?"


def test_mock_llm_is_extractive_and_deterministic():
    llm = MockLLM()
    answer = llm.generate(QUESTION, CONTEXTS)
    assert answer == llm.generate(QUESTION, CONTEXTS)
    assert "Stanford University" in answer
    assert llm.generate("anything", []).startswith("I could not find")


def test_mock_llm_caps_sentences_and_removes_duplicates():
    contexts = [f"Fact number {i} about robots." for i in range(10)] + [
        "Fact number 0 about robots."
    ]
    answer = MockLLM(max_sentences=3).generate("robots facts", contexts)
    assert answer.count("Fact number") == 3


def test_prompt_contains_question_and_context():
    prompt = build_prompt(QUESTION, CONTEXTS)
    assert QUESTION in prompt and "- Maria Chen founded Helios Robotics." in prompt


def test_openai_compatible_llm_with_fake_client():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        msg = SimpleNamespace(content=" Stanford University. ")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    llm = OpenAICompatLLM("http://x", "k", "some-model", client=fake)
    assert llm.generate(QUESTION, CONTEXTS) == "Stanford University."
    assert calls[0]["model"] == "some-model" and calls[0]["temperature"] == 0


def test_get_llm_modes(monkeypatch):
    monkeypatch.delenv("GRAPHRAG_LLM", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert isinstance(get_llm(), MockLLM)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        get_llm("gemini")
    with pytest.raises(ValueError):
        get_llm("nonsense")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-test-key")
    gem = get_llm("gemini")
    assert isinstance(gem, OpenAICompatLLM) and gem.name.startswith("gemini")


def _azure_transport(seen):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if "embeddings" in request.url.path:
            body = {
                "object": "list",
                "model": "m",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
                "data": [
                    {"object": "embedding", "index": i, "embedding": [float(i), 1.0]}
                    for i in range(len(json.loads(request.content)["input"]))
                ],
            }
        else:
            body = {
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Stanford University"},
                    }
                ],
            }
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


def test_azure_openai_chat_request_shape_and_parsing():
    seen: list[httpx.Request] = []
    llm = AzureOpenAILLM(
        "https://example-resource.openai.azure.com",
        "dummy-key",
        "gpt4o-deploy",
        http_client=httpx.Client(transport=_azure_transport(seen)),
    )
    assert llm.generate(QUESTION, CONTEXTS) == "Stanford University"
    req = seen[0]
    assert req.url.path == "/openai/deployments/gpt4o-deploy/chat/completions"
    assert req.url.params["api-version"] == "2024-10-21"
    assert req.headers["api-key"] == "dummy-key"
    assert QUESTION in json.loads(req.content)["messages"][0]["content"]


def test_azure_openai_embedder():
    seen: list[httpx.Request] = []
    emb = AzureOpenAIEmbedder(
        "https://example-resource.openai.azure.com",
        "dummy-key",
        "embed-deploy",
        http_client=httpx.Client(transport=_azure_transport(seen)),
    )
    assert emb.embed(["a", "b"]) == [[0.0, 1.0], [1.0, 1.0]]
    assert seen[0].url.path == "/openai/deployments/embed-deploy/embeddings"


def test_azure_from_env(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example-resource.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "dep")
    assert get_llm("azure").name == "azure:dep"


def _bedrock_client():
    return boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="dummy",
        aws_secret_access_key="dummy",  # noqa: S106 - not real
    )


def test_bedrock_converse_is_validated_against_the_service_model():
    client = _bedrock_client()
    model = "anthropic.claude-3-haiku-20240307-v1:0"
    with Stubber(client) as stub:
        stub.add_response(
            "converse",
            {
                "output": {"message": {"role": "assistant", "content": [{"text": " Stanford "}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
                "metrics": {"latencyMs": 1},
            },
            expected_params={
                "modelId": model,
                "messages": [
                    {"role": "user", "content": [{"text": build_prompt(QUESTION, CONTEXTS)}]}
                ],
                "inferenceConfig": {"temperature": 0.0, "maxTokens": 400},
            },
        )
        assert BedrockLLM(model, client).generate(QUESTION, CONTEXTS) == "Stanford"
        stub.assert_no_pending_responses()


def test_bedrock_titan_embedder():
    client = _bedrock_client()
    payload = json.dumps({"embedding": [0.5, 0.5], "inputTextTokenCount": 3}).encode()
    with Stubber(client) as stub:
        stub.add_response(
            "invoke_model",
            {
                "body": StreamingBody(io.BytesIO(payload), len(payload)),
                "contentType": "application/json",
            },
            expected_params={
                "modelId": "amazon.titan-embed-text-v2:0",
                "body": json.dumps({"inputText": "hello", "dimensions": 1024, "normalize": True}),
                "contentType": "application/json",
                "accept": "application/json",
            },
        )
        assert BedrockEmbedder(client=client).embed(["hello"]) == [[0.5, 0.5]]
