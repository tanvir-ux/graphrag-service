"""Small text helpers shared by embeddings, extraction and generation."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an and are as at be by did do does for from has have how in is it its of on or that the
    their there this to was were what when where which who whom whose why with would about into
    than then them they also can could should will it's i me my we you your he she his
    her""".split()
)

_WORD = re.compile(r"[a-z0-9]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def stem(word: str) -> str:
    """A deliberately tiny stemmer: enough to match plurals and simple verb forms."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def tokens(text: str) -> list[str]:
    return [stem(w) for w in _WORD.findall(text.lower()) if w not in STOPWORDS]


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text.strip()) if s.strip()]
