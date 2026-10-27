"""Generators: a deterministic extractive mock plus real-provider adapters."""

from __future__ import annotations

import os
from typing import Protocol

from .text import sentences, tokens
from .trace import traced

PROMPT = (
    "Answer the question using only the context. Be concise (one or two sentences). "
    "If the context is not enough, say so.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
)

GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


class LLM(Protocol):
    name: str

    def generate(self, question: str, contexts: list[str]) -> str: ...


def build_prompt(question: str, contexts: list[str]) -> str:
    return PROMPT.format(context="\n".join(f"- {c}" for c in contexts), question=question)


class MockLLM:
    """Extractive, deterministic 'generator'.

    It splits the retrieved context into sentences, ranks them by word overlap with the
    question (ties keep retrieval order) and returns the best few. It cannot reason, so any
    quality difference between pipelines comes from what they retrieve, not from the model.
    """

    name = "mock-extractive"

    def __init__(self, max_sentences: int = 5) -> None:
        self.max_sentences = max_sentences

    @traced("llm.mock", kind="llm")
    def generate(self, question: str, contexts: list[str]) -> str:
        q = set(tokens(question))
        seen: set[str] = set()
        scored: list[tuple[int, int, str]] = []
        order = 0
        for ctx in contexts:
            for sent in sentences(ctx):
                if sent in seen:
                    continue
                seen.add(sent)
                scored.append((len(q & set(tokens(sent))), order, sent))
                order += 1
        if not scored:
            return "I could not find an answer in the retrieved context."
        # keep the top sentences, but present them in retrieval order
        top = sorted(
            sorted(scored, key=lambda t: (-t[0], t[1]))[: self.max_sentences], key=lambda t: t[1]
        )
        return " ".join(s for _, _, s in top)


class OpenAICompatLLM:
    """Any OpenAI-compatible chat endpoint (used for Gemini's compatibility API)."""

    def __init__(self, base_url: str, api_key: str, model: str, client=None) -> None:
        from openai import OpenAI

        self.name = model
        self.model = model
        self.client = client or OpenAI(base_url=base_url, api_key=api_key, timeout=60)

    @traced("llm.openai_compat", kind="llm")
    def generate(self, question: str, contexts: list[str]) -> str:
        res = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": build_prompt(question, contexts)}],
            temperature=0,
        )
        return (res.choices[0].message.content or "").strip()


def get_llm(mode: str | None = None) -> LLM:
    """mock (default), gemini, azure or bedrock, chosen by argument or GRAPHRAG_LLM."""
    mode = (mode or os.getenv("GRAPHRAG_LLM", "mock")).lower()
    if mode == "mock":
        return MockLLM()
    if mode == "gemini":
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("gemini mode requires GEMINI_API_KEY")
        model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        return OpenAICompatLLM(GEMINI_OPENAI_BASE, key, model)
    if mode == "azure":
        from .adapters.azure_openai import AzureOpenAILLM

        return AzureOpenAILLM.from_env()
    if mode == "bedrock":
        from .adapters.bedrock import BedrockLLM

        return BedrockLLM.from_env()
    raise ValueError(f"unknown LLM mode: {mode}")
