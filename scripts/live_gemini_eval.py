"""Optional live check: answer a few questions with a real Gemini model, plain RAG vs GraphRAG.

Needs GEMINI_API_KEY (never printed or stored). Uses at most 2 x N requests, one attempt each,
because the free tier is small. Writes a plain-text log to docs/demo/live-gemini.txt.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from graphrag_service.engine import Engine
from graphrag_service.evaluation.dataset import QUESTIONS, score_one_text
from graphrag_service.llm import get_llm

IDS = ("q06", "q07", "q08", "q09", "q12", "q13")
OUT = Path(__file__).resolve().parent.parent / "docs" / "demo" / "live-gemini.txt"


def main() -> int:
    llm = get_llm("gemini")
    engine = Engine()
    lines = [
        f"live Gemini check, model={llm.name}, {datetime.now().astimezone():%Y-%m-%d %H:%M %Z}"
    ]
    ok = total = 0
    for q in (q for q in QUESTIONS if q.id in IDS):
        lines.append(f"\n{q.id} {q.text}\n  expected: {' + '.join(q.gold_answers)}")
        for mode in ("vector_llamaindex", "graphrag"):
            total += 1
            try:
                res = engine.query(q.text, mode, llm=llm)
            except Exception as exc:  # report provider errors honestly, do not retry
                lines.append(f"  {mode:<18} ERROR {type(exc).__name__}")
                continue
            good = score_one_text(q, res["answer"])
            ok += good
            lines.append(f"  {mode:<18} {'OK ' if good else 'MISS'} {res['answer'][:300]!r}")
    lines.append(f"\ncorrect answers: {ok}/{total} (errors count as not correct)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
