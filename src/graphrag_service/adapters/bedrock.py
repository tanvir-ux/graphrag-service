"""Amazon Bedrock adapters: Converse API for generation, Titan for embeddings.

Configuration: AWS credentials and region from the standard AWS environment, plus
BEDROCK_MODEL_ID (default `anthropic.claude-3-haiku-20240307-v1:0`) and
BEDROCK_EMBEDDING_MODEL_ID (default `amazon.titan-embed-text-v2:0`).

NOT executed against live Bedrock. The unit tests use botocore's `Stubber` on a real
`bedrock-runtime` client, so request parameters and response shapes are validated against the
AWS service model, but no AWS call was made.
"""

from __future__ import annotations

import json
import os
from typing import Any

from ..llm import build_prompt
from ..trace import traced


def make_client(region: str | None = None):
    import boto3

    return boto3.client(
        "bedrock-runtime", region_name=region or os.getenv("AWS_REGION", "us-east-1")
    )


class BedrockLLM:
    def __init__(self, model_id: str, client: Any | None = None) -> None:
        self.name = f"bedrock:{model_id}"
        self.model_id = model_id
        self.client = client or make_client()

    @classmethod
    def from_env(cls) -> BedrockLLM:
        return cls(os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"))

    @traced("llm.bedrock", kind="llm")
    def generate(self, question: str, contexts: list[str]) -> str:
        res = self.client.converse(
            modelId=self.model_id,
            messages=[{"role": "user", "content": [{"text": build_prompt(question, contexts)}]}],
            inferenceConfig={"temperature": 0.0, "maxTokens": 400},
        )
        blocks = res["output"]["message"]["content"]
        return "".join(b.get("text", "") for b in blocks).strip()


class BedrockEmbedder:
    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v2:0",
        dim: int = 1024,
        client: Any | None = None,
    ) -> None:
        self.dim = dim
        self.model_id = model_id
        self.client = client or make_client()

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            res = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({"inputText": text, "dimensions": self.dim, "normalize": True}),
                contentType="application/json",
                accept="application/json",
            )
            out.append(json.loads(res["body"].read())["embedding"])
        return out
