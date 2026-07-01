"""Stage 8: persistent semantic memory — embedded Qdrant + fastembed.

Qdrant runs in-process (`QdrantClient(path=...)`) and persists to disk, so
there's no server daemon to run. fastembed (BAAI/bge-small-en-v1.5, 384-d)
produces the vectors on CPU.

Each turn's exchange is embedded and stored; before each turn the user's query
is embedded and the most similar past memories are retrieved and injected into
the LLM context, giving Atlas recall across sessions.

Best-effort by design: if the backend can't start or a call fails, methods
degrade to no-ops / empty results so the assistant keeps working without memory.

Run directly for a standalone test:
    python memory.py
"""

from __future__ import annotations

import time
import uuid
from typing import List, Optional

from config import MemoryConfig


class Memory:
    def __init__(self, cfg: MemoryConfig, cache=None):
        self.cfg = cfg
        self.cache = cache  # optional embedding cache (see cache.py)
        self.enabled = False
        self.disabled_reason = ""
        self._embed = None
        self._client = None
        if not cfg.enable_memory:
            self.disabled_reason = "disabled in config"
            return
        try:
            from fastembed import TextEmbedding
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            self._embed = TextEmbedding(model_name=cfg.embed_model)
            dim = len(self._vector("probe"))
            self._client = QdrantClient(path=cfg.qdrant_path)
            if not self._client.collection_exists(cfg.collection):
                self._client.create_collection(
                    cfg.collection,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
            self.enabled = True
        except Exception as e:
            # The most common failure on Windows: another Atlas instance still
            # holds the embedded store's exclusive lock. Make it actionable.
            if "already accessed" in str(e):
                self.disabled_reason = (
                    "the memory store is locked by another running Atlas "
                    "instance — close any other 'python main.py' and retry"
                )
            else:
                self.disabled_reason = str(e)

    def _vector(self, text: str) -> list[float]:
        # Cache embeddings (deterministic per model+text) to skip recompute.
        if self.cache is not None and self.cache.enabled:
            import cache as cache_mod

            ck = cache_mod.key("embed", self.cfg.embed_model, text)
            cached = self.cache.get_json(ck)
            if cached is not None:
                return cached
            vec = list(self._embed.embed([text]))[0].tolist()
            self.cache.set_json(ck, vec, ttl=self.cache.cfg.embed_ttl)
            return vec
        return list(self._embed.embed([text]))[0].tolist()

    def remember(self, text: str, kind: str = "exchange") -> None:
        """Store a memory. No-op if memory is disabled or text is empty."""
        if not self.enabled or not text.strip():
            return
        try:
            from qdrant_client.models import PointStruct

            self._client.upsert(
                self.cfg.collection,
                [
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=self._vector(text),
                        payload={"text": text, "kind": kind, "ts": time.time()},
                    )
                ],
            )
        except Exception as e:
            print(f"[memory] store failed ({e})")

    def recall(self, query: str, k: Optional[int] = None) -> List[str]:
        """Return up to k past memories most relevant to query (above threshold)."""
        if not self.enabled or not query.strip():
            return []
        try:
            hits = self._client.query_points(
                self.cfg.collection,
                query=self._vector(query),
                limit=k or self.cfg.recall_k,
                score_threshold=self.cfg.score_threshold,
            ).points
            return [h.payload["text"] for h in hits]
        except Exception as e:
            print(f"[memory] recall failed ({e})")
            return []

    def note(self, text: str) -> bool:
        """Store something the user explicitly asked Atlas to remember.

        Notes are kept apart from ordinary exchanges and are *always* surfaced
        into context (see notes()), so directives like "always answer briefly"
        or facts like "call me Boss" reliably stick instead of only being
        recalled when a turn happens to be similar.
        """
        if not self.enabled or not text.strip():
            return False
        self.remember(text.strip(), kind="note")
        return True

    def notes(self, limit: int = 25) -> List[str]:
        """Return recent explicit notes (oldest-first), for always-on injection."""
        if not self.enabled:
            return []
        try:
            from qdrant_client.models import (FieldCondition, Filter,
                                              MatchValue)
            points, _ = self._client.scroll(
                self.cfg.collection,
                scroll_filter=Filter(must=[FieldCondition(
                    key="kind", match=MatchValue(value="note"))]),
                limit=500, with_payload=True,
            )
            ordered = sorted(points, key=lambda p: p.payload.get("ts", 0))
            return [p.payload["text"] for p in ordered[-limit:]]
        except Exception as e:
            print(f"[memory] notes failed ({e})")
            return []

    def forget(self, text: str, min_score: float = 0.60) -> Optional[str]:
        """Delete the note most similar to `text`. Returns the deleted text.

        The threshold is deliberately high (bge-small cosine: real matches land
        ~0.64+, unrelated queries ~0.5) so it won't delete the wrong note.
        """
        if not self.enabled or not text.strip():
            return None
        try:
            from qdrant_client.models import (FieldCondition, Filter,
                                              MatchValue)
            hits = self._client.query_points(
                self.cfg.collection,
                query=self._vector(text),
                query_filter=Filter(must=[FieldCondition(
                    key="kind", match=MatchValue(value="note"))]),
                limit=1, with_payload=True,
            ).points
            if not hits or hits[0].score < min_score:
                return None
            self._client.delete(self.cfg.collection,
                                points_selector=[hits[0].id])
            return hits[0].payload.get("text", "")
        except Exception as e:
            print(f"[memory] forget failed ({e})")
            return None

    def count(self) -> int:
        """Number of stored memories (0 if disabled)."""
        if not self.enabled:
            return 0
        try:
            return self._client.count(self.cfg.collection).count
        except Exception:
            return 0

    def reset(self) -> bool:
        """Delete every stored memory. Deletes all points (an empty filter
        matches everything) rather than dropping the collection — in Qdrant's
        embedded/local mode, drop+recreate leaves the on-disk points behind."""
        if not self.enabled:
            return False
        try:
            from qdrant_client import models

            self._client.delete(
                self.cfg.collection,
                points_selector=models.FilterSelector(filter=models.Filter()),
            )
            return True
        except Exception as e:
            print(f"[memory] reset failed ({e})")
            return False

    def close(self) -> None:
        """Release the embedded store's file lock (call on shutdown)."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass


if __name__ == "__main__":
    # Standalone Step A test: store a few facts, then recall by meaning.
    mem = Memory(MemoryConfig())
    print("enabled:", mem.enabled)
    for fact in [
        "The user owns a cat named Pixel.",
        "The user prefers tea over coffee.",
        "The user is building a voice assistant called Atlas.",
    ]:
        mem.remember(fact, kind="fact")
    for q in ["what pet do I have", "what am I working on"]:
        print(f"\nrecall {q!r}:")
        for m in mem.recall(q):
            print("  -", m)
    mem.close()
