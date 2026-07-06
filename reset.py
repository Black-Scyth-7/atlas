"""Manually reset Atlas's state — a standalone alternative to the in-app
"reset everything" voice command, for when you want to wipe things from a
terminal (e.g. you're locked out, or want a clean slate before a demo).

Stop Atlas first (its embedded memory/DB may hold file locks while running).

Usage
-----
    python reset.py                 # full reset (asks you to confirm)
    python reset.py --yes           # full reset, no prompt
    python reset.py --dry-run       # show what WOULD be removed, delete nothing
    python reset.py --identity      # only face / voice / password / users
    python reset.py --memory --cache

Categories (pick any; none given = all of them):
    --identity   enrolled faces, voiceprints, startup password, user registry
    --memory     long-term semantic memory (Qdrant)
    --history    saved conversation history (Postgres, if configured)
    --cache      web-search / embedding cache (Redis, if configured)
    --outputs    generated files: crew_output/, meetings/, photos/, logs
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

# Import config for the canonical paths; fall back to defaults if it can't load.
try:
    from config import (AuthConfig, CacheConfig, Config, FaceConfig,
                        MemoryConfig, StateConfig, VaultConfig)
    _cfg, _auth, _face = Config(), AuthConfig(), FaceConfig()
    _mem, _state, _cache = MemoryConfig(), StateConfig(), CacheConfig()
    VOICEPRINT = _cfg.voiceprint_path
    PASSWORD = _auth.password_path
    USERS = _auth.users_path
    FACES = getattr(_face, "db_path", "faces.npz")
    QDRANT = getattr(_mem, "qdrant_path", "qdrant_data")
    VAULT = VaultConfig().vault_path
except Exception as e:                                    # pragma: no cover
    print(f"(couldn't import config: {e}; using defaults)")
    _state = _cache = None
    VOICEPRINT, PASSWORD, USERS, FACES, QDRANT = (
        "voiceprint.npy", "auth_secret.dat", "users.json", "faces.npz", "qdrant_data")
    VAULT = "vault.dat"

VOICEPRINTS_DIR = "voiceprints"
OUTPUT_DIRS = ["crew_output", "meetings", "photos"]
OUTPUT_FILES = ["face_window.log"]

ALL_CATEGORIES = ["identity", "memory", "history", "cache", "outputs"]


def _p(path: str) -> str:
    """Absolute path anchored at the repo, so reset works from any cwd."""
    return path if os.path.isabs(path) else os.path.join(_HERE, path)


def _rm(path: str, removed: list, dry: bool) -> None:
    """Delete a file or directory if it exists; record what happened."""
    full = _p(path)
    if not os.path.exists(full):
        return
    kind = "dir " if os.path.isdir(full) else "file"
    if dry:
        removed.append(f"  would remove {kind}: {path}")
        return
    try:
        if os.path.isdir(full):
            shutil.rmtree(full)
        else:
            os.remove(full)
        removed.append(f"  removed {kind}: {path}")
    except Exception as e:
        removed.append(f"  COULD NOT remove {path}: {e} "
                       "(is Atlas still running?)")


def reset_identity(removed: list, dry: bool) -> None:
    for path in (VOICEPRINT, VOICEPRINTS_DIR, FACES, PASSWORD, USERS, VAULT):
        _rm(path, removed, dry)


def reset_memory(removed: list, dry: bool) -> None:
    _rm(QDRANT, removed, dry)


def reset_outputs(removed: list, dry: bool) -> None:
    for path in OUTPUT_DIRS + OUTPUT_FILES:
        _rm(path, removed, dry)


def reset_history(removed: list, dry: bool) -> None:
    """Conversation history lives in Postgres (if configured); use its own reset
    so we don't need psql. No-op if the DB isn't reachable."""
    if _state is None:
        return
    if dry:
        removed.append("  would clear conversation history (Postgres, if reachable)")
        return
    try:
        from state import Store
        store = Store(_state)
        if getattr(store, "enabled", False):
            removed.append("  conversation history cleared" if store.reset()
                           else "  history could not be cleared")
        else:
            removed.append(f"  history store not active ({getattr(store, 'disabled_reason', 'n/a')})")
        for close in (getattr(store, "close", None),):
            if close:
                try:
                    close()
                except Exception:
                    pass
    except Exception as e:
        removed.append(f"  history skipped: {e}")


def reset_cache(removed: list, dry: bool) -> None:
    """Web/embedding cache in Redis (if configured). No-op if unreachable."""
    if _cache is None:
        return
    if dry:
        removed.append("  would clear cache (Redis, if reachable)")
        return
    try:
        from cache import Cache
        cache = Cache(_cache)
        if getattr(cache, "enabled", False):
            removed.append(f"  cache cleared ({cache.reset()} keys)")
        else:
            removed.append("  cache not active (Redis not connected)")
    except Exception as e:
        removed.append(f"  cache skipped: {e}")


_RUNNERS = {
    "identity": reset_identity,
    "memory": reset_memory,
    "history": reset_history,
    "cache": reset_cache,
    "outputs": reset_outputs,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Manually reset Atlas's state.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="With no category flags, EVERYTHING is reset.")
    for cat in ALL_CATEGORIES:
        ap.add_argument(f"--{cat}", action="store_true", help=f"reset {cat}")
    ap.add_argument("-y", "--yes", action="store_true", help="skip confirmation")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be removed, delete nothing")
    args = ap.parse_args(argv)

    chosen = [c for c in ALL_CATEGORIES if getattr(args, c)]
    if not chosen:
        chosen = list(ALL_CATEGORIES)

    print("Atlas manual reset")
    print("  target(s): " + ", ".join(chosen))
    if args.dry_run:
        print("  (dry run — nothing will be deleted)\n")
    elif not args.yes:
        print("\nThis is IRREVERSIBLE. Atlas should be stopped first.")
        if chosen == ALL_CATEGORIES or "identity" in chosen:
            print("Removing identity means the NEXT start will re-run first-time "
                  "registration (face / voice / password).")
        try:
            ans = input("Type 'reset' to confirm: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return 1
        if ans != "reset":
            print("Aborted.")
            return 1

    removed: list[str] = []
    for cat in chosen:
        _RUNNERS[cat](removed, args.dry_run)

    print()
    if removed:
        print("\n".join(removed))
    else:
        print("  Nothing to remove — already clean.")
    print("\n" + ("Dry run complete." if args.dry_run else "Reset complete."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
