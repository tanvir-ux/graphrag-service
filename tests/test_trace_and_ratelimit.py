import asyncio
import sys
import types

import pytest

from graphrag_service import trace
from graphrag_service.ratelimit import RateLimiter


def test_traced_records_sync_and_async_without_self():
    trace.RECORDER.reset()

    class Thing:
        @trace.traced("t.method", kind="tool")
        def add(self, a):
            return a + 1

    @trace.traced("t.async")
    async def aadd(a, b):
        return a + b

    assert Thing().add(1) == 2
    assert asyncio.run(aadd(2, 3)) == 5
    events = trace.RECORDER.snapshot()
    assert [e["name"] for e in events] == ["t.method", "t.async"]
    assert events[0]["input"] == "1" and "Thing object" not in events[0]["input"]


def test_integrations_disabled_without_env(monkeypatch):
    for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGSMITH_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert not trace.langfuse_enabled() and not trace.langsmith_enabled()


def test_langfuse_and_langsmith_wrappers_are_applied(monkeypatch):
    """Verified with fake SDKs; not executed against the hosted services."""
    seen = []

    def fake(label):
        def factory(name):
            def deco(fn):
                seen.append((label, name))
                return fn

            return deco

        return factory

    monkeypatch.setitem(sys.modules, "langfuse", types.SimpleNamespace(observe=fake("langfuse")))
    monkeypatch.setitem(
        sys.modules, "langsmith", types.SimpleNamespace(traceable=fake("langsmith"))
    )
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    @trace.traced("hooked")
    def fn():
        return 1

    assert fn() == 1
    assert ("langfuse", "hooked") in seen and ("langsmith", "hooked") in seen


@pytest.mark.parametrize(
    "per_key,total,expected", [(2, 10, [True, True, False]), (5, 1, [True, False, False])]
)
def test_rate_limiter(per_key, total, expected):
    rl = RateLimiter(per_key=per_key, total=total, window=60)
    assert [rl.allow("ip", now=float(i)) for i in range(3)] == expected


def test_rate_limiter_window_expires():
    rl = RateLimiter(per_key=1, total=5, window=10)
    assert rl.allow("a", now=0) and not rl.allow("a", now=5) and rl.allow("a", now=11)
