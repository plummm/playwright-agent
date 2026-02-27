"""SQLite-backed browser activity logging scaffold."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


class BrowserEventLogger:
    """Persist browser activity events into SQLite."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()

    def start(self) -> None:
        with self._lock:
            if self._conn is not None:
                return
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            self._conn = conn
            self._init_schema()

    def close(self) -> None:
        with self._lock:
            if self._conn is None:
                return
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def init_run(self, run_id: str, request_id: str, agent_name: str) -> None:
        payload = {
            "run_id": str(run_id or ""),
            "request_id": str(request_id or ""),
            "agent_name": str(agent_name or ""),
            "started_at": float(time.time()),
        }
        self._safe_execute(
            """
            INSERT OR REPLACE INTO browser_runs (
                run_id, request_id, agent_name, started_at, status, error
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                payload["run_id"],
                payload["request_id"],
                payload["agent_name"],
                payload["started_at"],
                "running",
                None,
            ),
        )

    def complete_run(self, run_id: str, status: str, error: Optional[str] = None) -> None:
        self._safe_execute(
            """
            UPDATE browser_runs
               SET ended_at = ?, status = ?, error = ?
             WHERE run_id = ?
            """,
            (
                float(time.time()),
                str(status or "unknown"),
                str(error) if error else None,
                str(run_id or ""),
            ),
        )

    def log_network_event(self, payload: Dict[str, Any]) -> None:
        self._insert_event("network_events", payload)

    def log_console_event(self, payload: Dict[str, Any]) -> None:
        self._insert_event("console_events", payload)

    def log_download_event(self, payload: Dict[str, Any]) -> None:
        self._insert_event("download_events", payload)

    def log_action_event(self, payload: Dict[str, Any]) -> None:
        self._insert_event("action_events", payload)

    def _init_schema(self) -> None:
        self._safe_execute(
            """
            CREATE TABLE IF NOT EXISTS browser_runs (
                run_id TEXT PRIMARY KEY,
                request_id TEXT,
                agent_name TEXT,
                started_at REAL,
                ended_at REAL,
                status TEXT,
                error TEXT
            )
            """
        )

        for table in ("network_events", "console_events", "download_events", "action_events"):
            self._safe_execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    request_id TEXT,
                    ts REAL,
                    event_type TEXT,
                    payload_json TEXT
                )
                """
            )
            self._safe_execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_run_id ON {table}(run_id)"
            )
            self._safe_execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_ts ON {table}(ts)"
            )
            self._safe_execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_event_type ON {table}(event_type)"
            )

    def _insert_event(self, table: str, payload: Dict[str, Any]) -> None:
        p = payload if isinstance(payload, dict) else {}
        run_id = str(p.get("run_id") or "")
        request_id = str(p.get("request_id") or "")
        event_type = str(p.get("event_type") or "")
        ts = p.get("ts")
        try:
            ts_val = float(ts if ts is not None else time.time())
        except Exception:
            ts_val = float(time.time())
        try:
            payload_json = json.dumps(p, ensure_ascii=False, default=str)
        except Exception:
            payload_json = json.dumps({"_error": "payload_not_serializable"}, ensure_ascii=False)
        self._safe_execute(
            f"""
            INSERT INTO {table} (run_id, request_id, ts, event_type, payload_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, request_id, ts_val, event_type, payload_json),
        )

    def _safe_execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        with self._lock:
            try:
                if self._conn is None:
                    self.start()
                if self._conn is None:
                    return
                self._conn.execute(sql, params)
                self._conn.commit()
            except Exception:
                # Best-effort logger: never raise back into automation flow.
                return
