"""SQLite-backed browser activity logging scaffold."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


class BrowserEventLogger:
    """Persist browser activity events into SQLite."""

    def __init__(self, db_path: Path):
        pass

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def init_run(self, run_id: str, request_id: str, agent_name: str) -> None:
        pass

    def complete_run(self, run_id: str, status: str, error: Optional[str] = None) -> None:
        pass

    def log_network_event(self, payload: Dict[str, Any]) -> None:
        pass

    def log_console_event(self, payload: Dict[str, Any]) -> None:
        pass

    def log_download_event(self, payload: Dict[str, Any]) -> None:
        pass

    def log_action_event(self, payload: Dict[str, Any]) -> None:
        pass
