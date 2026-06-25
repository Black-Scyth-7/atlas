"""Stage 9: durable agent state in PostgreSQL.

Persists the ordered conversation log and a small user-profile key/value store,
so Atlas keeps continuity across restarts. This is distinct from memory.py:
  - memory.py (Qdrant) = semantic recall — "find relevant past exchanges".
  - state.py  (Postgres) = the durable transcript + profile, and reloading the
    last few turns verbatim into working context on startup.

Best-effort: if Postgres is unreachable, every method no-ops / returns defaults
so the assistant still runs (in-memory only).

Connection string from ATLAS_PG_DSN in .env, e.g.
    postgresql://postgres:atlas@localhost:5432/atlas

Run directly for a standalone test:
    python state.py
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from config import StateConfig

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          BIGSERIAL PRIMARY KEY,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS messages (
    id          BIGSERIAL PRIMARY KEY,
    session_id  BIGINT REFERENCES sessions(id),
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    ts          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS messages_id_idx ON messages(id);
CREATE TABLE IF NOT EXISTS profile (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


class Store:
    def __init__(self, cfg: StateConfig):
        self.cfg = cfg
        self.enabled = False
        self.disabled_reason = ""
        self._conn = None
        self.session_id: Optional[int] = None
        if not cfg.enable_state:
            self.disabled_reason = "disabled in config"
            return
        if not cfg.dsn:
            self.disabled_reason = "no ATLAS_PG_DSN set"
            return
        try:
            import psycopg

            self._conn = psycopg.connect(cfg.dsn, autocommit=True, connect_timeout=5)
            with self._conn.cursor() as cur:
                cur.execute(_SCHEMA)
                cur.execute("INSERT INTO sessions DEFAULT VALUES RETURNING id")
                self.session_id = cur.fetchone()[0]
            self.enabled = True
        except Exception as e:
            self.disabled_reason = str(e).strip().splitlines()[0] if str(e) else "connect failed"

    def add_message(self, role: str, content: str) -> None:
        """Append a message to the durable log for this session."""
        if not self.enabled or not content.strip():
            return
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO messages (session_id, role, content) VALUES (%s, %s, %s)",
                    (self.session_id, role, content),
                )
        except Exception as e:
            print(f"[state] write failed ({e})")

    def recent_messages(self, limit: int) -> List[Tuple[str, str]]:
        """Last `limit` (role, content) messages across all sessions, oldest first."""
        if not self.enabled:
            return []
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT role, content FROM messages ORDER BY id DESC LIMIT %s",
                    (limit,),
                )
                rows = cur.fetchall()
            return [(role, content) for role, content in reversed(rows)]
        except Exception as e:
            print(f"[state] read failed ({e})")
            return []

    def set_profile(self, key: str, value: str) -> None:
        """Upsert a durable user-profile fact/setting."""
        if not self.enabled:
            return
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO profile (key, value) VALUES (%s, %s) "
                    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, "
                    "updated_at = now()",
                    (key, value),
                )
        except Exception as e:
            print(f"[state] profile write failed ({e})")

    def get_profile(self, key: str) -> Optional[str]:
        if not self.enabled:
            return None
        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT value FROM profile WHERE key = %s", (key,))
                row = cur.fetchone()
            return row[0] if row else None
        except Exception as e:
            print(f"[state] profile read failed ({e})")
            return None

    def message_count(self) -> int:
        if not self.enabled:
            return 0
        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM messages")
                return cur.fetchone()[0]
        except Exception:
            return 0

    def reset(self) -> bool:
        """Wipe the whole transcript + profile, then start a fresh session."""
        if not self.enabled:
            return False
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "TRUNCATE messages, sessions, profile RESTART IDENTITY CASCADE")
                cur.execute("INSERT INTO sessions DEFAULT VALUES RETURNING id")
                self.session_id = cur.fetchone()[0]
            return True
        except Exception as e:
            print(f"[state] reset failed ({e})")
            return False

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    store = Store(StateConfig())
    print("enabled:", store.enabled, "| reason:", store.disabled_reason or "-")
    if store.enabled:
        print("session:", store.session_id, "| total messages:", store.message_count())
        store.add_message("user", "Testing state persistence.")
        store.add_message("assistant", "Stored to PostgreSQL.")
        store.set_profile("name", "test-user")
        print("recent:", store.recent_messages(4))
        print("profile name:", store.get_profile("name"))
    store.close()
