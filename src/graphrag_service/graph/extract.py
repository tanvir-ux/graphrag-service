"""Rule-based triple extraction (deterministic, no LLM).

A sentence like "Maria Chen founded Helios Robotics." becomes
(Maria Chen, FOUNDED, Helios Robotics). The extractor looks at adjacent capitalised entity
mentions and the verb phrase between them. It is tuned to the demo corpus; the `Extractor`
protocol is where an LLM-based extractor would plug in for real documents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from ..corpus import Document
from ..text import sentences

_ENTITY = re.compile(r"[A-Z][\w-]*(?: (?:of )?[A-Z][\w-]*)*")

VERBS: dict[str, str] = {
    "founded": "FOUNDED",
    "acquired": "ACQUIRED",
    "manufactures": "MANUFACTURES",
    "supplies": "SUPPLIES",
    "uses": "USES",
    "invested in": "INVESTED_IN",
    "partnered with": "PARTNERED_WITH",
    "studied at": "STUDIED_AT",
    "works at": "WORKS_AT",
    "leads": "LEADS",
    "is headquartered in": "HEADQUARTERED_IN",
    "is located in": "LOCATED_IN",
}

# entity type implied by the role an entity plays in a relation
SUBJECT_TYPE = {
    "FOUNDED": "person",
    "STUDIED_AT": "person",
    "WORKS_AT": "person",
    "LEADS": "person",
}
OBJECT_TYPE = {
    "STUDIED_AT": "university",
    "HEADQUARTERED_IN": "place",
    "LOCATED_IN": "place",
    "MANUFACTURES": "product",
    "USES": "product",
}


@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    object: str
    doc_id: str
    sentence: str


class Extractor(Protocol):
    def extract(self, doc: Document) -> list[Triple]: ...


class RuleExtractor:
    def extract(self, doc: Document) -> list[Triple]:
        triples: list[Triple] = []
        for sentence in sentences(doc.text):
            mentions = list(_ENTITY.finditer(sentence))
            for left, right in zip(mentions, mentions[1:], strict=False):
                between = sentence[left.end() : right.start()].strip().lower()
                between = between.removeprefix("the ").strip()
                relation = VERBS.get(between)
                if relation:
                    triples.append(Triple(left.group(), relation, right.group(), doc.id, sentence))
        return triples
