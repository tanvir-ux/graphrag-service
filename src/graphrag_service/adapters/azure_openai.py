"""Azure OpenAI adapters (chat completions and embeddings).

Configuration: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT,
AZURE_OPENAI_EMBEDDING_DEPLOYMENT and optionally AZURE_OPENAI_API_VERSION.

NOT executed against a live Azure OpenAI resource. The unit tests drive the real `openai`
client through a mocked HTTP transport and check the request shape and response parsing.
"""

from __future__ import annotations

import os
from typing import Any

from ..llm import build_prompt
from ..trace import traced


def _client(endpoint: str, api_key: str, api_version: str, http_client: Any = None):
    from openai import AzureOpenAI

    return AzureOpenAI(
        azure_endpoint=endpoint, api_key=api_key, api_version=api_version, http_client=http_client
    )


class AzureOpenAILLM:
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment: str,
        api_version: str = "2024-10-21",
        http_client: Any = None,
    ) -> None:
        self.name = f"azure:{deployment}"
        self.deployment = deployment
        self.client = _client(endpoint, api_key, api_version, http_client)

    @classmethod
    def from_env(cls) -> AzureOpenAILLM:
        return cls(
            os.environ["AZURE_OPENAI_ENDPOINT"],
            os.environ["AZURE_OPENAI_API_KEY"],
            os.environ["AZURE_OPENAI_DEPLOYMENT"],
            os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )

    @traced("llm.azure_openai", kind="llm")
    def generate(self, question: str, contexts: list[str]) -> str:
        res = self.client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "user", "content": build_prompt(question, contexts)}],
            temperature=0,
        )
        return (res.choices[0].message.content or "").strip()


class AzureOpenAIEmbedder:
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment: str,
        dim: int = 1536,
        api_version: str = "2024-10-21",
        http_client: Any = None,
    ) -> None:
        self.dim = dim
        self.deployment = deployment
        self.client = _client(endpoint, api_key, api_version, http_client)

    @classmethod
    def from_env(cls) -> AzureOpenAIEmbedder:
        return cls(
            os.environ["AZURE_OPENAI_ENDPOINT"],
            os.environ["AZURE_OPENAI_API_KEY"],
            os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        res = self.client.embeddings.create(model=self.deployment, input=texts)
        return [item.embedding for item in res.data]
