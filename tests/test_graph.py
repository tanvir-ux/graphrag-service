from graphrag_service.corpus import DOC_BY_ID, DOCS
from graphrag_service.graph.build import KnowledgeGraph
from graphrag_service.graph.extract import RuleExtractor
from graphrag_service.graph.retrieve import (
    graph_retrieve,
    hinted_relations,
    link_entities,
    search_paths,
)


def test_extractor_reads_simple_sentences():
    triples = RuleExtractor().extract(DOC_BY_ID["d03"])
    keys = {(t.subject, t.relation, t.object) for t in triples}
    assert ("Tomas Ruiz", "FOUNDED", "Nimbus Vision") in keys
    assert ("Nimbus Vision", "LOCATED_IN", "Boston") in keys
    assert ("Helios Robotics", "ACQUIRED", "Nimbus Vision") in keys
    assert all(t.sentence and t.doc_id == "d03" for t in triples)


def test_distractor_document_yields_no_triples():
    assert RuleExtractor().extract(DOC_BY_ID["d15"]) == []


def test_graph_shape_and_types(kg):
    assert kg.graph.number_of_nodes() == 26
    assert kg.graph.number_of_edges() == 30
    assert kg.graph.nodes["Maria Chen"]["type"] == "person"
    assert kg.graph.nodes["Austin"]["type"] == "place"
    assert kg.graph.nodes["Stanford University"]["type"] == "university"
    assert kg.graph.nodes["Orbit Rover"]["type"] == "product"
    assert kg.graph.nodes["Helios Robotics"]["type"] == "organization"


def test_duplicate_facts_are_stored_once(kg):
    keys = [(t.subject, t.relation, t.object) for t in kg.triples]
    assert len(keys) == len(set(keys))
    assert len(DOCS) == 15


def test_communities_are_deterministic_and_cover_every_node():
    a, b = KnowledgeGraph.build(), KnowledgeGraph.build()
    assert a.communities == b.communities and a.summaries == b.summaries
    assert set().union(*a.communities) == set(a.graph.nodes)
    assert all(s.startswith(f"Community {i + 1} covers") for i, s in enumerate(a.summaries))


def test_entity_linking_prefers_longest_names(kg):
    q = "Did Maria Chen study at Massachusetts Institute of Technology or Stanford University?"
    assert link_entities(kg, q) == [
        "Maria Chen",
        "Massachusetts Institute of Technology",
        "Stanford University",
    ]


def test_relation_hints_include_stop_word_where():
    assert {"HEADQUARTERED_IN", "LOCATED_IN"} <= hinted_relations("Where is it?")
    assert "FOUNDED" in hinted_relations("Who is the founder?")
    assert hinted_relations("Tell me a joke") == set()


def test_multi_hop_path_is_found_and_ranked_first(kg):
    paths = search_paths(kg, ["Helios Robotics"], {"FOUNDED", "STUDIED_AT"})
    assert len(paths) == 1
    steps = paths[0][1]
    assert [s[1] for s in steps] == ["Maria Chen", "Stanford University"]


def test_graph_retrieve_returns_only_the_needed_sentences(kg):
    res = graph_retrieve(kg, "Which university did the founder of Helios Robotics attend?")
    assert res.strategy == "paths"
    assert [c["text"] for c in res.contexts] == [
        "Maria Chen founded Helios Robotics.",
        "Maria Chen studied at Stanford University.",
    ]
    assert ("Maria Chen", "Stanford University", "STUDIED_AT") in res.edges
    assert "Stanford University" in res.nodes


def test_three_hop_question(kg):
    res = graph_retrieve(
        kg, "Which university did the founder of the company that Helios Robotics acquired attend?"
    )
    text = " ".join(c["text"] for c in res.contexts)
    assert "Tomas Ruiz founded Nimbus Vision" in text
    assert "Massachusetts Institute of Technology" in text
    assert "Stanford" not in text


def test_overview_strategy_lists_neighbours(kg):
    res = graph_retrieve(kg, "Give an overview of the companies connected to Helios Robotics.")
    assert res.strategy == "overview"
    first = res.contexts[0]["text"]
    for name in ("Nimbus Vision", "Voltix Batteries", "Northwind Labs", "Greenfield Capital"):
        assert name in first


def test_unknown_phrasing_falls_back_to_community_reports(kg):
    res = graph_retrieve(kg, "Which place is home to the firm that Helios Robotics bought?")
    assert res.strategy == "communities"
    assert all(c["doc_id"] == "community" for c in res.contexts)


def test_no_entities_and_no_overlap_gives_empty_context(kg):
    res = graph_retrieve(kg, "zzz qqq")
    assert res.contexts == [] and res.evidence == []
