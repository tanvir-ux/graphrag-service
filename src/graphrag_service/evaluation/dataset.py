"""Evaluation questions with gold evidence documents and gold answer fragments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    id: str
    category: str  # single_hop | multi_hop | global | paraphrase
    text: str
    gold_docs: tuple[str, ...]
    gold_answers: tuple[str, ...]  # every fragment must appear in the answer


QUESTIONS: list[Question] = [
    Question("q01", "single_hop", "Where is Helios Robotics headquartered?", ("d01",), ("Austin",)),
    Question("q02", "single_hop", "Who founded Nimbus Vision?", ("d03",), ("Tomas Ruiz",)),
    Question(
        "q03",
        "single_hop",
        "Which products does Helios Robotics manufacture?",
        ("d01",),
        ("Atlas Arm", "Orbit Rover"),
    ),
    Question(
        "q04",
        "single_hop",
        "Which university did Anika Rao attend?",
        ("d06",),
        ("Carnegie Mellon University",),
    ),
    Question(
        "q05", "single_hop", "Where is Mars Survey Institute located?", ("d09",), ("Pasadena",)
    ),
    Question(
        "q06",
        "multi_hop",
        "Which university did the founder of Helios Robotics attend?",
        ("d01", "d02"),
        ("Stanford University",),
    ),
    Question(
        "q07",
        "multi_hop",
        "Which university did the founder of the company that supplies Helios Robotics attend?",
        ("d05", "d06"),
        ("Carnegie Mellon University",),
    ),
    Question(
        "q08",
        "multi_hop",
        "In which city is the company that Helios Robotics acquired based?",
        ("d03",),
        ("Boston",),
    ),
    Question(
        "q09",
        "multi_hop",
        "Who invested in both Helios Robotics and Voltix Batteries?",
        ("d12",),
        ("Greenfield Capital",),
    ),
    Question(
        "q10",
        "multi_hop",
        "Which institute uses a product made by the company Maria Chen founded, "
        "and where is it located?",
        ("d01", "d08", "d09"),
        ("Mars Survey Institute", "Pasadena"),
    ),
    Question(
        "q11",
        "multi_hop",
        "What chip does the Orbit Rover use, and which company manufactures it?",
        ("d08", "d10"),
        ("Quanta Chip", "Northwind Labs"),
    ),
    Question(
        "q12",
        "multi_hop",
        "Which university did the founder of the company that Helios Robotics acquired attend?",
        ("d03", "d04"),
        ("Massachusetts Institute of Technology",),
    ),
    Question(
        "q13",
        "global",
        "Give an overview of the companies connected to Helios Robotics.",
        ("d03", "d05", "d10", "d12"),
        ("Nimbus Vision", "Voltix Batteries", "Northwind Labs", "Greenfield Capital"),
    ),
    Question(
        "q14",
        "global",
        "Summarize the main relationships around Voltix Batteries.",
        ("d05", "d06", "d12"),
        ("Anika Rao", "Helios Robotics", "Greenfield Capital", "Voltix Cells"),
    ),
    # Paraphrases that avoid the relation words the graph retriever's lexicon knows.
    # They are here on purpose: they show where the rule-based GraphRAG breaks.
    Question(
        "q15",
        "paraphrase",
        "Which place is home to the firm that Helios Robotics bought?",
        ("d03",),
        ("Boston",),
    ),
    Question(
        "q16",
        "paraphrase",
        "Who bankrolled the maker of the Atlas Arm?",
        ("d01", "d12"),
        ("Greenfield Capital",),
    ),
    Question(
        "q17",
        "paraphrase",
        "Which college did the person who started Nimbus Vision go to?",
        ("d03", "d04"),
        ("Massachusetts Institute of Technology",),
    ),
]


def score_one_text(q: Question, answer: str) -> bool:
    """True when the answer contains every gold fragment (case-insensitive)."""
    lowered = answer.lower()
    return all(frag.lower() in lowered for frag in q.gold_answers)
