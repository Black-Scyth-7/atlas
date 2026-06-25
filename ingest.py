"""Index your documents for RAG.

Reads txt/md/pdf files and stores their embedded chunks in Atlas's document
store (a separate Qdrant collection). Run this while main.py is NOT running (the
embedded store is single-process).

    python ingest.py                 # index everything under docs/ (config default)
    python ingest.py notes.md a.pdf  # index specific files
    python ingest.py ~/Documents     # index a folder recursively

Re-running is safe: a file's chunks are overwritten, not duplicated.
"""

import os
import sys

from config import RAGConfig, CacheConfig
from cache import Cache
from rag import DocStore


def main(argv: list[str]) -> None:
    cfg = RAGConfig()
    cache = Cache(CacheConfig())  # reuse the embedding cache if Redis is up
    store = DocStore(cfg, cache=cache)
    if not store.enabled:
        raise SystemExit(f"Document store unavailable: {store.disabled_reason}")

    targets = argv or [cfg.docs_dir]
    total_files = total_chunks = 0
    print(f"Indexing into collection '{cfg.collection}' ({store.count()} chunks already present)...")
    for target in targets:
        if os.path.isdir(target):
            files, chunks = store.ingest_dir(target)
        elif os.path.isfile(target):
            chunks = store.ingest_file(target)
            files = 1 if chunks else 0
            if chunks:
                print(f"  indexed {os.path.basename(target)} ({chunks} chunks)")
        else:
            print(f"  skip (not found): {target}")
            continue
        total_files += files
        total_chunks += chunks

    if total_files == 0:
        print("No supported documents found (txt, md, pdf). "
              f"Put files in '{cfg.docs_dir}/' or pass paths.")
    else:
        print(f"Done: {total_files} file(s), {total_chunks} chunk(s). "
              f"Total in store: {store.count()}.")
    store.close()
    cache.close()


if __name__ == "__main__":
    main(sys.argv[1:])
