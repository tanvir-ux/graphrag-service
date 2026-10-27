"""Knowledge graph construction, community detection and community summaries."""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from ..corpus import DOCS, Document
from .extract import OBJECT_TYPE, SUBJECT_TYPE, Extractor, RuleExtractor, Triple


@dataclass
class KnowledgeGraph:
    graph: nx.MultiDiGraph = field(default_factory=nx.MultiDiGraph)
    triples: list[Triple] = field(default_factory=list)
    communities: list[set[str]] = field(default_factory=list)
    summaries: list[str] = field(default_factory=list)

    @classmethod
    def build(
        cls, docs: list[Document] | None = None, extractor: Extractor | None = None
    ) -> KnowledgeGraph:
        extractor = extractor or RuleExtractor()
        kg = cls()
        seen: set[tuple[str, str, str]] = set()
        for doc in docs or DOCS:
            for t in extractor.extract(doc):
                key = (t.subject, t.relation, t.object)
                if key in seen:
                    continue
                seen.add(key)
                kg.triples.append(t)
                kg._add(t)
        kg._detect_communities()
        return kg

    def _add(self, t: Triple) -> None:
        for name, role in ((t.subject, SUBJECT_TYPE), (t.object, OBJECT_TYPE)):
            current = self.graph.nodes[name].get("type") if name in self.graph else None
            kind = role.get(t.relation) or current or "organization"
            if current in ("person", "university", "place", "product") and kind == "organization":
                kind = current
            self.graph.add_node(name, type=kind)
        self.graph.add_edge(
            t.subject, t.object, relation=t.relation, doc_id=t.doc_id, sentence=t.sentence
        )

    def _detect_communities(self) -> None:
        undirected = nx.Graph(self.graph)
        parts = nx.community.louvain_communities(undirected, seed=42)
        self.communities = sorted((set(p) for p in parts), key=lambda s: sorted(s)[0])
        self.summaries = [self._summarise(i, members) for i, members in enumerate(self.communities)]

    def _summarise(self, index: int, members: set[str]) -> str:
        """Templated community report (a real system would ask an LLM to write this)."""
        by_type: dict[str, list[str]] = {}
        for name in sorted(members):
            by_type.setdefault(self.graph.nodes[name]["type"], []).append(name)
        plural = {
            "person": "people",
            "place": "places",
            "university": "universities",
            "product": "products",
            "organization": "organizations",
        }
        parts = [f"{plural.get(k, k)}: {', '.join(n)}" for k, n in sorted(by_type.items())]
        facts = [t.sentence for t in self.triples if t.subject in members and t.object in members]
        return f"Community {index + 1} covers {'; '.join(parts)}. " + " ".join(dict.fromkeys(facts))

    def community_of(self, name: str) -> int | None:
        for i, members in enumerate(self.communities):
            if name in members:
                return i
        return None

    def to_json(self) -> dict:
        return {
            "nodes": [
                {"id": n, "type": d["type"], "community": self.community_of(n)}
                for n, d in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, "relation": d["relation"], "doc_id": d["doc_id"]}
                for u, v, d in self.graph.edges(data=True)
            ],
        }
