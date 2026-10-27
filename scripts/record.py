"""Record real runs (mock generator) into web/recordings.json for the static GitHub Pages demo.

Usage: python scripts/record.py
Writes src/graphrag_service/web/recordings.json and mirrors the web files into docs/.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from graphrag_service.api import create_app
from graphrag_service.engine import MODES

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "src" / "graphrag_service" / "web"
DOCS = ROOT / "docs"


def main() -> None:
    client = TestClient(create_app())
    examples = client.get("/api/examples").json()
    examples["gemini_enabled"] = False
    queries = {}
    for q in examples["questions"]:
        for mode in MODES:
            res = client.post("/api/query", json={"question": q["text"], "mode": mode}).json()
            queries[f"{mode}::{q['text']}"] = res
    recording = {
        "note": "Recorded from the real service with the deterministic mock generator.",
        "examples": examples,
        "graph": client.get("/api/graph").json(),
        "eval": client.get("/api/eval").json(),
        "queries": queries,
    }
    (WEB / "recordings.json").write_text(json.dumps(recording, separators=(",", ":")))
    DOCS.mkdir(exist_ok=True)
    for name in ("index.html", "recordings.json"):
        shutil.copy(WEB / name, DOCS / name)
    (DOCS / ".nojekyll").touch()
    print(f"recorded {len(queries)} queries")


if __name__ == "__main__":
    main()
