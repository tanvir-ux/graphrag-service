"""Evaluation harness: retrieval recall and answer correctness per pipeline."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from ..engine import MODES, Engine
from ..llm import LLM
from .dataset import QUESTIONS, Question, score_one_text


def score_one(q: Question, result: dict[str, Any]) -> dict[str, Any]:
    retrieved = {c["doc_id"] for c in result["contexts"]}
    recall = sum(d in retrieved for d in q.gold_docs) / len(q.gold_docs)
    correct = score_one_text(q, result["answer"])
    return {
        "doc_recall": round(recall, 3),
        "answer_correct": correct,
        "context_chars": sum(len(c["text"]) for c in result["contexts"]),
        "n_contexts": len(result["contexts"]),
    }


def run_eval(
    engine: Engine | None = None,
    modes: tuple[str, ...] = MODES,
    llm: LLM | None = None,
    questions: list[Question] | None = None,
) -> dict[str, Any]:
    engine = engine or Engine()
    questions = questions or QUESTIONS
    rows: list[dict[str, Any]] = []
    for q in questions:
        for mode in modes:
            res = engine.query(q.text, mode, llm=llm)
            rows.append(
                {
                    "id": q.id,
                    "category": q.category,
                    "question": q.text,
                    "mode": mode,
                    "answer": res["answer"],
                    **score_one(q, res),
                }
            )
    return {
        "llm": (llm or engine.llm).name,
        "n_questions": len(questions),
        "rows": rows,
        "summary": summarise(rows, modes),
    }


def summarise(rows: list[dict[str, Any]], modes: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for mode in modes:
        buckets: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            if r["mode"] == mode:
                buckets["all"].append(r)
                buckets[r["category"]].append(r)
        out[mode] = {
            cat: {
                "n": len(rs),
                "doc_recall": round(sum(r["doc_recall"] for r in rs) / len(rs), 3),
                "answer_accuracy": round(sum(r["answer_correct"] for r in rs) / len(rs), 3),
                "avg_context_chars": round(sum(r["context_chars"] for r in rs) / len(rs)),
            }
            for cat, rs in buckets.items()
        }
    return out


def to_markdown(report: dict[str, Any]) -> str:
    lines = [f"Generator: `{report['llm']}`, {report['n_questions']} questions.", ""]
    lines += [
        "| pipeline | category | n | doc recall | answer accuracy | avg context chars |",
        "|---|---|---|---|---|---|",
    ]
    for mode, cats in report["summary"].items():
        for cat in ("single_hop", "multi_hop", "global", "paraphrase", "all"):
            if cat in cats:
                c = cats[cat]
                lines.append(
                    f"| {mode} | {cat} | {c['n']} | {c['doc_recall']:.2f} | "
                    f"{c['answer_accuracy']:.2f} | {c['avg_context_chars']} |"
                )
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    report = run_eval()
    print(to_markdown(report))
    print(json.dumps(report["summary"], indent=2))
