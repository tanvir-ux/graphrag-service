"""Tiny in-memory sliding-window rate limiter used to protect the optional live-LLM mode."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    """Allow at most `per_key` hits per key and `total` hits overall within `window` seconds."""

    def __init__(self, per_key: int, total: int, window: float = 3600.0) -> None:
        self.per_key, self.total, self.window = per_key, total, window
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._all: deque[float] = deque()
        self._lock = threading.Lock()

    def _trim(self, q: deque[float], now: float) -> None:
        while q and now - q[0] > self.window:
            q.popleft()

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            mine = self._hits[key]
            self._trim(mine, now)
            self._trim(self._all, now)
            if len(mine) >= self.per_key or len(self._all) >= self.total:
                return False
            mine.append(now)
            self._all.append(now)
            return True
