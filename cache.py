"""Stage 10: Redis cache (Memurai on Windows).

Caches repeated/expensive results to cut latency and external calls:
  - web-search responses (short TTL — results go stale),
  - text embeddings (long-lived — deterministic for a given model+text).

Best-effort: if Redis is unreachable, every method no-ops / returns None and the
assistant runs uncached. Connection from ATLAS_REDIS_URL in .env
(default redis://localhost:6379/0).

Run directly for a standalone test:
    python cache.py
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from config import CacheConfig


def key(namespace: str, *parts: str) -> str:
    """Build a namespaced cache key, hashing the variable parts."""
    digest = hashlib.sha1("\x00".join(parts).encode("utf-8")).hexdigest()
    return f"atlas:{namespace}:{digest}"


class Cache:
    def __init__(self, cfg: CacheConfig):
        self.cfg = cfg
        self.enabled = False
        self.disabled_reason = ""
        self._r = None
        if not cfg.enable_cache:
            self.disabled_reason = "disabled in config"
            return
        try:
            import redis

            self._r = redis.Redis.from_url(
                cfg.url, socket_connect_timeout=3, decode_responses=True
            )
            self._r.ping()
            self.enabled = True
        except Exception as e:
            self.disabled_reason = str(e).strip().splitlines()[0] if str(e) else "connect failed"

    def get(self, k: str) -> Optional[str]:
        if not self.enabled:
            return None
        try:
            return self._r.get(k)
        except Exception:
            return None

    def set(self, k: str, value: str, ttl: Optional[int] = None) -> None:
        if not self.enabled:
            return
        try:
            # redis rejects ex=0; treat 0/None as "no expiry".
            self._r.set(k, value, ex=ttl if ttl else None)
        except Exception:
            pass

    def get_json(self, k: str) -> Optional[Any]:
        raw = self.get(k)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def set_json(self, k: str, obj: Any, ttl: Optional[int] = None) -> None:
        self.set(k, json.dumps(obj), ttl)

    def reset(self) -> int:
        """Delete all of Atlas's cache keys (atlas:*). Returns the count removed.

        Scans by prefix rather than flushing the DB, so other apps sharing this
        Redis instance are untouched.
        """
        if not self.enabled:
            return 0
        try:
            removed = 0
            for k in self._r.scan_iter(match="atlas:*", count=500):
                self._r.delete(k)
                removed += 1
            return removed
        except Exception:
            return 0

    def close(self) -> None:
        if self._r is not None:
            try:
                self._r.close()
            except Exception:
                pass


if __name__ == "__main__":
    c = Cache(CacheConfig())
    print("enabled:", c.enabled, "| reason:", c.disabled_reason or "-")
    if c.enabled:
        k = key("test", "hello")
        c.set(k, "world", ttl=60)
        print("roundtrip:", c.get(k))
        c.set_json(key("test", "vec"), [0.1, 0.2, 0.3], ttl=60)
        print("json roundtrip:", c.get_json(key("test", "vec")))
    c.close()
