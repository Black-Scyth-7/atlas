"""Stage 11: RAG over your documents.

Indexes your own files (txt / md / pdf) into a SEPARATE embedded Qdrant
collection (distinct from conversation memory) and retrieves the most relevant
chunks for a query. Exposed to the LLM as the `search_documents` tool, so Atlas
can answer questions grounded in your notes/files.

Embeddings reuse fastembed (bge-small) and the Redis embedding cache. Index with
ingest.py; query via DocStore.search().

Best-effort: if the store can't open, it reports disabled and the tool is simply
not offered.
"""

from __future__ import annotations

import glob
import json
import os
import uuid
from typing import List, Optional, Tuple

from config import RAGConfig

# Stable namespace so re-ingesting a file overwrites its chunks instead of
# duplicating them (point id = uuid5(source, chunk_index)).
_NS = uuid.UUID("a71a5000-0000-4000-8000-a71a5d0c0000")

_TEXT_EXTS = {".txt", ".md", ".markdown", ".rst", ".text"}
_PDF_EXTS = {".pdf"}


def read_document(path: str) -> str:
    """Extract plain text from a supported file (txt/md/pdf). '' if unsupported."""
    ext = os.path.splitext(path)[1].lower()
    if ext in _TEXT_EXTS:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext in _PDF_EXTS:
        from pypdf import PdfReader

        reader = PdfReader(path)
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    return ""


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """Split text into ~size-char chunks with overlap, breaking on paragraphs."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buf = ""
    for para in paragraphs:
        if buf and len(buf) + len(para) + 2 > size:
            chunks.append(buf)
            buf = buf[-overlap:] if overlap else ""  # carry tail for context
        buf = f"{buf}\n\n{para}".strip() if buf else para
        # A single huge paragraph: hard-split it.
        while len(buf) > size:
            chunks.append(buf[:size])
            buf = buf[size - overlap:]
    if buf.strip():
        chunks.append(buf)
    return chunks


class DocStore:
    def __init__(self, cfg: RAGConfig, cache=None):
        self.cfg = cfg
        self.cache = cache
        self.enabled = False
        self.disabled_reason = ""
        self._embed = None
        self._client = None
        if not cfg.enable_rag:
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
            # Track source file mtimes so incremental ingest can skip unchanged files.
            self._manifest_path = os.path.join(cfg.qdrant_path, ".manifest.json")
            self._manifest = self._load_manifest()
            self.enabled = True
        except Exception as e:
            if "already accessed" in str(e):
                self.disabled_reason = (
                    "the document store is locked by another process "
                    "(is ingest.py or another Atlas running?)"
                )
            else:
                self.disabled_reason = str(e).strip().splitlines()[0] if str(e) else "init failed"

    def _vector(self, text: str) -> list[float]:
        if self.cache is not None and getattr(self.cache, "enabled", False):
            import cache as cache_mod

            ck = cache_mod.key("embed", self.cfg.embed_model, text)
            cached = self.cache.get_json(ck)
            if cached is not None:
                return cached
            vec = list(self._embed.embed([text]))[0].tolist()
            self.cache.set_json(ck, vec, ttl=self.cache.cfg.embed_ttl)
            return vec
        return list(self._embed.embed([text]))[0].tolist()

    def _load_manifest(self) -> dict:
        try:
            with open(self._manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_manifest(self) -> None:
        try:
            with open(self._manifest_path, "w", encoding="utf-8") as f:
                json.dump(self._manifest, f)
        except Exception:
            pass

    def ingest_file(self, path: str, force: bool = True) -> int:
        """Index one file; returns chunks stored (0 if unsupported/unchanged).

        With force=False, a file whose modification time is unchanged since the
        last ingest is skipped (used by startup auto-ingest).
        """
        if not self.enabled:
            return 0
        from qdrant_client.models import PointStruct

        source = os.path.basename(path)
        mtime = os.path.getmtime(path)
        if not force and self._manifest.get(source) == mtime:
            return 0

        text = read_document(path)
        chunks = chunk_text(text, self.cfg.chunk_chars, self.cfg.chunk_overlap)
        if not chunks:
            return 0
        points = [
            PointStruct(
                id=str(uuid.uuid5(_NS, f"{source}:{i}")),
                vector=self._vector(chunk),
                payload={"text": chunk, "source": source, "chunk": i},
            )
            for i, chunk in enumerate(chunks)
        ]
        self._client.upsert(self.cfg.collection, points)
        self._manifest[source] = mtime
        self._save_manifest()
        return len(points)

    def ingest_dir(self, directory: str, force: bool = True) -> Tuple[int, int]:
        """Index supported files under a directory. Returns (files, chunks).

        With force=False, unchanged files are skipped (cheap startup re-index).
        """
        files = chunks = 0
        for path in sorted(glob.glob(os.path.join(directory, "**", "*"), recursive=True)):
            if not os.path.isfile(path):
                continue
            if os.path.splitext(path)[1].lower() not in (_TEXT_EXTS | _PDF_EXTS):
                continue
            n = self.ingest_file(path, force=force)
            if n:
                files += 1
                chunks += n
                print(f"  indexed {os.path.basename(path)} ({n} chunks)")
        return files, chunks

    def search(self, query: str, k: Optional[int] = None) -> List[Tuple[str, str]]:
        """Return up to k (text, source) chunks most relevant to the query."""
        if not self.enabled or not query.strip():
            return []
        try:
            hits = self._client.query_points(
                self.cfg.collection,
                query=self._vector(query),
                limit=k or self.cfg.top_k,
                score_threshold=self.cfg.score_threshold,
            ).points
            return [(h.payload["text"], h.payload.get("source", "?")) for h in hits]
        except Exception as e:
            print(f"[rag] search failed ({e})")
            return []

    def count(self) -> int:
        if not self.enabled:
            return 0
        try:
            return self._client.count(self.cfg.collection).count
        except Exception:
            return 0

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass


if __name__ == "__main__":
    store = DocStore(RAGConfig())
    print("enabled:", store.enabled, "| chunks:", store.count(),
          "| reason:", store.disabled_reason or "-")
    if store.enabled:
        for text, source in store.search("test query"):
            print(f"  [{source}] {text[:80]}")
    store.close()
