"""GraphRAG retrieval: entity linking, relation-guided path search, community reports."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..text import stem, tokens
from .build import KnowledgeGraph

# question word prefixes (after stemming) -> relations they hint at
HINTS: dict[str, tuple[str, ...]] = {
    "found": ("FOUNDED",),
    "creat": ("FOUNDED",),
    "attend": ("STUDIED_AT",),
    "univers": ("STUDIED_AT",),
    "studi": ("STUDIED_AT",),
    "school": ("STUDIED_AT",),
    "graduat": ("STUDIED_AT",),
    "where": ("HEADQUARTERED_IN", "LOCATED_IN"),
    "city": ("HEADQUARTERED_IN", "LOCATED_IN"),
    "locat": ("HEADQUARTERED_IN", "LOCATED_IN"),
    "bas": ("HEADQUARTERED_IN", "LOCATED_IN"),
    "headquarter": ("HEADQUARTERED_IN",),
    "mak": ("MANUFACTURES",),
    "made": ("MANUFACTURES",),
    "manufactur": ("MANUFACTURES",),
    "produc": ("MANUFACTURES",),
    "acquir": ("ACQUIRED",),
    "suppl": ("SUPPLIES",),
    "vendor": ("SUPPLIES",),
    "invest": ("INVESTED_IN",),
    "backer": ("INVESTED_IN",),
    "us": ("USES",),
    "chip": ("USES", "MANUFACTURES"),
    "partner": ("PARTNERED_WITH",),
    "work": ("WORKS_AT",),
    "employ": ("WORKS_AT",),
    "lead": ("LEADS",),
}
OVERVIEW_WORDS = ("summar", "overview", "main", "across", "theme", "ecosystem", "relationship")


@dataclass
class GraphResult:
    contexts: list[dict]
    evidence: list[str]
    nodes: set[str] = field(default_factory=set)
    edges: set[tuple[str, str, str]] = field(default_factory=set)
    strategy: str = "paths"


def link_entities(kg: KnowledgeGraph, question: str) -> list[str]:
    """Find graph entities mentioned in the question (longest name first, no overlaps)."""
    lowered = question.lower()
    taken = [False] * len(lowered)
    found: list[tuple[int, str]] = []
    for name in sorted(kg.graph.nodes, key=len, reverse=True):
        for m in re.finditer(re.escape(name.lower()), lowered):
            if not any(taken[m.start() : m.end()]):
                for i in range(m.start(), m.end()):
                    taken[i] = True
                found.append((m.start(), name))
                break
    return [name for _, name in sorted(found)]


def hinted_relations(question: str) -> set[str]:
    relations: set[str] = set()
    # stop words such as "where" matter here, so this does not use the filtered tokenizer
    for tok in (stem(w) for w in re.findall(r"[a-z]+", question.lower())):
        for prefix, rels in HINTS.items():
            if tok.startswith(prefix):
                relations.update(rels)
    return relations


def _neighbours(kg: KnowledgeGraph, node: str):
    for _, v, d in kg.graph.out_edges(node, data=True):
        yield v, (node, v, d["relation"]), d, "→"
    for u, _, d in kg.graph.in_edges(node, data=True):
        yield u, (u, node, d["relation"]), d, "←"


def search_paths(
    kg: KnowledgeGraph, seeds: list[str], relations: set[str], max_hops: int = 4, top: int = 4
) -> list[tuple[float, list]]:
    """Enumerate simple paths from the seeds and keep those that best match the question.

    Score = distinct question-relevant relations covered - 0.5 * unmatched edges
    - 0.05 * length. A path must end on a matched edge and may contain one unmatched edge.
    Only paths within 0.11 of the best score are kept, so a path that covers every
    relation the question asks about beats paths that cover fewer of them.
    """
    results: list[tuple[float, list]] = []

    def score_of(steps: list) -> float:
        matched = {s[3]["relation"] for s in steps if s[4]}
        unmatched = sum(1 for s in steps if not s[4])
        return len(matched) - 0.5 * unmatched - 0.05 * len(steps)

    def walk(node: str, visited: set[str], steps: list, unmatched: int) -> None:
        if steps and steps[-1][4]:
            results.append((score_of(steps), list(steps)))
        if len(steps) >= max_hops:
            return
        for nxt, edge, data, arrow in _neighbours(kg, node):
            if nxt in visited:
                continue
            hit = data["relation"] in relations
            if not hit and unmatched >= 1:
                continue
            steps.append((node, nxt, edge, data, hit, arrow))
            walk(nxt, visited | {nxt}, steps, unmatched + (not hit))
            steps.pop()

    for seed in seeds:
        walk(seed, {seed}, [], 0)
    if not results:
        return []
    results.sort(key=lambda r: (-r[0], len(r[1])))
    best = results[0][0]
    picked, used = [], set()
    for score, steps in results:
        if score < best - 0.11:
            break
        key = frozenset(s[2] for s in steps)
        if key in used:
            continue
        used.add(key)
        picked.append((score, steps))
        if len(picked) == top:
            break
    return picked


def _describe(steps: list) -> str:
    text = steps[0][0]
    for _, nxt, edge, _, _, arrow in steps:
        text += f" {arrow}{edge[2]}{arrow} {nxt}" if arrow == "→" else f" {arrow}{edge[2]}— {nxt}"
    return text


def graph_retrieve(kg: KnowledgeGraph, question: str, max_communities: int = 2) -> GraphResult:
    seeds = link_entities(kg, question)
    relations = hinted_relations(question)
    q = " ".join(tokens(question))
    overview = any(w in q for w in OVERVIEW_WORDS)
    contexts: list[dict] = []
    evidence: list[str] = []
    result = GraphResult(contexts, evidence)

    def add_context(sentence: str, doc_id: str, score: float, source: str) -> None:
        if all(c["text"] != sentence for c in contexts):
            contexts.append(
                {"doc_id": doc_id, "text": sentence, "score": round(score, 3), "source": source}
            )

    if seeds and overview:
        result.strategy = "overview"
        comm_ids = []
        edge_sentences: list[tuple[str, str]] = []
        for seed in seeds:
            linked = []
            for nxt, edge, data, _arrow in _neighbours(kg, seed):
                result.nodes.update({seed, nxt})
                result.edges.add(edge)
                linked.append(nxt)
                edge_sentences.append((data["sentence"], data["doc_id"]))
            add_context(
                f"{seed} is directly connected to: {', '.join(dict.fromkeys(linked))}.",
                "neighbourhood",
                1.0,
                f"1-hop neighbourhood of {seed}",
            )
            evidence.append(f"neighbourhood of {seed}: {', '.join(dict.fromkeys(linked))}")
            c = kg.community_of(seed)
            if c is not None and c not in comm_ids:
                comm_ids.append(c)
        for sentence, doc_id in edge_sentences:
            add_context(sentence, doc_id, 0.5, "graph edge")
        for c in comm_ids[:max_communities]:
            add_context(kg.summaries[c], "community", 1.0, f"community report {c + 1}")
            evidence.append(f"community {c + 1}: {', '.join(sorted(kg.communities[c]))}")
        return result
    if seeds and relations:
        for score, steps in search_paths(kg, seeds, relations):
            evidence.append(f"{_describe(steps)}  (score {score:.2f})")
            for _, _, edge, data, _, _ in steps:
                result.nodes.update({edge[0], edge[1]})
                result.edges.add(edge)
                add_context(data["sentence"], data["doc_id"], score, "graph path")
        if contexts:
            return result
    # fall back to community reports ranked by word overlap with the question
    result.strategy = "communities"
    qset = set(tokens(question))
    ranked = sorted(
        range(len(kg.communities)),
        key=lambda i: -len(qset & set(tokens(kg.summaries[i]))),
    )
    for c in ranked[:max_communities]:
        overlap = len(qset & set(tokens(kg.summaries[c])))
        if overlap:
            add_context(kg.summaries[c], "community", overlap, f"community report {c + 1}")
            evidence.append(f"community {c + 1}: {', '.join(sorted(kg.communities[c]))}")
            result.nodes.update(kg.communities[c])
    return result
