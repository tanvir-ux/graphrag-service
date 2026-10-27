"""Command line: `graphrag serve | ask | eval | graph`."""

from __future__ import annotations

import argparse
import json

from .engine import MODES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="graphrag", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    serve = sub.add_parser("serve", help="start the FastAPI service and web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question")
    ask.add_argument("--mode", choices=MODES, default="graphrag")
    ask.add_argument("--llm", default="mock", help="mock | gemini | azure | bedrock")
    ev = sub.add_parser("eval", help="run the evaluation harness")
    ev.add_argument("--llm", default="mock")
    ev.add_argument("--json", action="store_true", help="print the full JSON report")
    ev.add_argument("--limit", type=int, default=0, help="only the first N questions")
    sub.add_parser("graph", help="print the knowledge graph statistics")
    args = parser.parse_args(argv)

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("graphrag_service.api:app", host=args.host, port=args.port)
        return 0

    from .engine import Engine
    from .llm import get_llm

    engine = Engine()
    if args.cmd == "graph":
        kg = engine.kg
        print(
            f"{kg.graph.number_of_nodes()} entities, {kg.graph.number_of_edges()} relations, "
            f"{len(kg.communities)} communities"
        )
        for i, summary in enumerate(kg.summaries):
            print(f"\n[{i + 1}] {summary}")
        return 0
    if args.cmd == "ask":
        res = engine.query(args.question, args.mode, llm=get_llm(args.llm))
        print(f"mode: {res['mode']} ({res['strategy']})   generator: {res['llm']}")
        print("\ncontext:")
        for c in res["contexts"]:
            print(f"  [{c['doc_id']}] {c['text']}")
        if res["evidence"]:
            print("\nevidence:")
            for e in res["evidence"]:
                print(f"  {e}")
        print(f"\nanswer: {res['answer']}")
        return 0
    from .evaluation.dataset import QUESTIONS
    from .evaluation.harness import run_eval, to_markdown

    questions = QUESTIONS[: args.limit] if args.limit else None
    report = run_eval(engine, llm=get_llm(args.llm), questions=questions)
    print(json.dumps(report, indent=2) if args.json else to_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
