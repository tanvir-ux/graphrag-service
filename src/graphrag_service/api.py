"""FastAPI service: query, graph, evaluation and the web UI."""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .corpus import DOCS
from .engine import MODES, Engine
from .evaluation.dataset import QUESTIONS
from .evaluation.harness import run_eval
from .llm import get_llm
from .ratelimit import RateLimiter
from .trace import RECORDER

WEB_DIR = Path(__file__).parent / "web"


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=300)
    mode: str = Field(default="graphrag", examples=list(MODES))
    llm: str = Field(default="mock", pattern="^(mock|gemini)$")


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return Engine()


def gemini_enabled() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") and os.getenv("ENABLE_GEMINI_DEMO"))


def create_app() -> FastAPI:
    app = FastAPI(title="graphrag-service", version=__version__)
    limiter = RateLimiter(
        per_key=int(os.getenv("GEMINI_DEMO_PER_IP_HOURLY", "5")),
        total=int(os.getenv("GEMINI_DEMO_TOTAL_HOURLY", "30")),
    )
    lock = threading.Lock()

    @app.get("/api/health")
    async def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/api/examples")
    async def examples() -> dict:
        return {
            "modes": list(MODES),
            "questions": [
                {
                    "id": q.id,
                    "category": q.category,
                    "text": q.text,
                    "gold_answers": list(q.gold_answers),
                }
                for q in QUESTIONS
            ],
            "documents": [{"id": d.id, "title": d.title, "text": d.text} for d in DOCS],
            "gemini_enabled": gemini_enabled(),
        }

    @app.get("/api/graph")
    async def graph() -> dict:
        kg = get_engine().kg
        return {**kg.to_json(), "summaries": kg.summaries}

    def _run(req: QueryRequest) -> dict:
        with lock:
            RECORDER.reset()
            engine = get_engine()
            llm = get_llm("gemini") if req.llm == "gemini" else None
            result = engine.query(req.question, req.mode, llm=llm)
            result["trace"] = RECORDER.snapshot()
            return result

    @app.post("/api/query")
    async def query(req: QueryRequest, request: Request) -> JSONResponse:
        if req.mode not in MODES:
            raise HTTPException(422, f"mode must be one of {', '.join(MODES)}")
        if req.llm == "gemini":
            if not gemini_enabled():
                raise HTTPException(403, "live Gemini mode is disabled on this server")
            client = request.client.host if request.client else "unknown"
            if not limiter.allow(client):
                raise HTTPException(429, "live Gemini demo is rate limited; try again later")
        try:
            return JSONResponse(await run_in_threadpool(_run, req))
        except Exception as exc:  # provider errors should not become a stack trace for visitors
            if req.llm == "gemini":
                raise HTTPException(502, f"Gemini request failed: {type(exc).__name__}") from exc
            raise

    @app.get("/api/eval")
    async def evaluate() -> dict:
        return await run_in_threadpool(lambda: run_eval(get_engine()))

    if WEB_DIR.exists():
        app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")

        @app.get("/recordings.json", include_in_schema=False)
        async def recordings() -> FileResponse:
            return FileResponse(WEB_DIR / "recordings.json")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")

    return app


app = create_app()
