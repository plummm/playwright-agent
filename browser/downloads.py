"""Download handling feature scaffold."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .event_logger import BrowserEventLogger
from .models import RunState
from .session import BrowserSessionManager


class DownloadsFeature:
    """Handle browser downloads and local file lifecycle."""

    def __init__(
        self,
        session_manager: BrowserSessionManager,
        event_logger: BrowserEventLogger,
        base_tmp_dir: str = "/tmp/playwright_agent",
    ):
        pass

    def get_tools(self) -> List[Any]:
        pass

    async def attach_listener(self, run_state: RunState) -> None:
        pass

    async def detach_listener(self, run_state: RunState) -> None:
        pass

    async def list_downloads(self, run_state: RunState, limit: int = 100) -> Dict[str, Any]:
        pass

    async def get_download(self, run_state: RunState, download_id: str) -> Dict[str, Any]:
        pass

    async def save_download(
        self,
        run_state: RunState,
        download_id: str,
        destination_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        pass

    async def cleanup_tmp(self, run_state: Optional[RunState] = None, ttl_seconds: int = 86400) -> Dict[str, Any]:
        pass
