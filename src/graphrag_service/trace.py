"""Tracing: a local span recorder plus optional Langfuse / LangSmith hooks.

The local recorder always runs and feeds the web UI. Langfuse and LangSmith wrappers are
applied only when their environment variables are present. They are covered by unit tests
with fakes; they were NOT executed against the real hosted services in this repository.
"""

from __future__ import annotations

import functools
import inspect
import os
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class TraceRecorder:
    """Thread-safe, append-only list of events for the current run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []
        self._t0 = time.perf_counter()

    def reset(self) -> None:
        with self._lock:
            self._events = []
            self._t0 = time.perf_counter()

    def add(self, kind: str, name: str, **data: Any) -> None:
        with self._lock:
            self._events.append(
                {
                    "i": len(self._events),
                    "kind": kind,
                    "name": name,
                    "t_ms": round((time.perf_counter() - self._t0) * 1000, 1),
                    **data,
                }
            )

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(e) for e in self._events]


RECORDER = TraceRecorder()


def langfuse_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def langsmith_enabled() -> bool:
    return bool(os.getenv("LANGSMITH_API_KEY")) and os.getenv("LANGSMITH_TRACING", "").lower() in {
        "1",
        "true",
        "yes",
    }


def _external_wrappers(name: str) -> list[Callable[[F], F]]:
    wrappers: list[Callable[[F], F]] = []
    if langfuse_enabled():
        from langfuse import observe

        wrappers.append(observe(name=name))
    if langsmith_enabled():
        from langsmith import traceable

        wrappers.append(traceable(name=name))
    return wrappers


def traced(name: str, kind: str = "span") -> Callable[[F], F]:
    """Decorator: record a local event and apply Langfuse/LangSmith wrappers when configured."""

    def deco(fn: F) -> F:
        target: Any = fn
        for wrap in _external_wrappers(name):
            target = wrap(target)

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                out = await target(*args, **kwargs)
                RECORDER.add(
                    kind, name, input=_short(_inputs(fn, args, kwargs)), output=_short(out)
                )
                return out

            return awrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            out = target(*args, **kwargs)
            RECORDER.add(kind, name, input=_short(_inputs(fn, args, kwargs)), output=_short(out))
            return out

        return wrapper  # type: ignore[return-value]

    return deco


def _inputs(fn: Callable[..., Any], args: tuple, kwargs: dict) -> Any:
    """Call inputs for the trace, without the `self` of a method."""
    params = list(inspect.signature(fn).parameters)
    if args and params and params[0] in ("self", "cls"):
        args = args[1:]
    return kwargs or (args[0] if len(args) == 1 else args)


def _short(value: Any, limit: int = 400) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"
