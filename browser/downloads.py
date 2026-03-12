"""Download handling for the Playwright agent."""

from __future__ import annotations

import asyncio
import inspect
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool

from .event_logger import BrowserEventLogger
from .models import RunState
from .session import BrowserSessionManager
from .tooling import mcp_tool


class DownloadsFeature:
    """Handle browser downloads and local file lifecycle."""

    def __init__(
        self,
        session_manager: BrowserSessionManager,
        event_logger: BrowserEventLogger,
        base_tmp_dir: str = "/tmp/playwright_agent",
    ):
        self.session_manager = session_manager
        self.event_logger = event_logger
        self.base_tmp_dir = str(base_tmp_dir or "/tmp/playwright_agent")
        self._run_state: Optional[RunState] = None
        self._tools: List[Any] = []
        self._download_records_by_run: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._handlers_by_run: Dict[str, Dict[str, Any]] = {}

    def set_run_state(self, run_state: Optional[RunState]) -> None:
        self._run_state = run_state

    def _require_run_state(self) -> RunState:
        if self._run_state is None:
            raise RuntimeError("DownloadsFeature run_state is not set")
        return self._run_state

    def get_tools(self) -> List[Any]:
        if self._tools:
            return self._tools
        tools: List[Any] = []
        for method_name in dir(self):
            method = getattr(self, method_name, None)
            if not callable(method):
                continue
            if not bool(getattr(method, "_is_mcp_tool", False)):
                continue
            tool_name = str(getattr(method, "_mcp_name", method.__name__) or method.__name__)
            doc = inspect.getdoc(method) or f"MCP tool: {tool_name}"
            examples = list(getattr(method, "_mcp_examples", []) or [])
            if examples:
                doc = f"{doc}\n\nExamples:\n" + "\n".join(f"- {x}" for x in examples)
            tools.append(
                StructuredTool.from_function(
                    name=tool_name,
                    description=doc,
                    coroutine=method,
                )
            )
        self._tools = tools
        return self._tools

    def _ensure_run_storage(self, run_state: RunState) -> Dict[str, Dict[str, Any]]:
        run_id = str(run_state.run_id)
        return self._download_records_by_run.setdefault(run_id, {})

    def _log(self, run_state: RunState, event_type: str, payload: Dict[str, Any]) -> None:
        try:
            self.event_logger.log_download_event(
                {
                    "run_id": run_state.run_id,
                    "request_id": run_state.request_id,
                    "event_type": event_type,
                    "ts": time.time(),
                    "payload": payload,
                }
            )
        except Exception:
            pass

    def _normalize_download_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "download_id": str(record.get("download_id") or ""),
            "url": str(record.get("url") or ""),
            "suggested_filename": str(record.get("suggested_filename") or ""),
            "status": str(record.get("status") or "pending"),
            "saved_path": record.get("saved_path"),
            "failure": record.get("failure"),
            "created_at": record.get("created_at"),
            "page_id": record.get("page_id"),
        }

    async def _handle_download(self, run_state: RunState, download: Any) -> None:
        store = self._ensure_run_storage(run_state)
        download_id = self.session_manager.next_download_id(run_state)
        page = None
        try:
            page = download.page
        except Exception:
            page = None
        page_id = None
        if page is not None:
            self.session_manager.sync_pages(run_state, preferred_page=page)
            page_id = str(run_state.active_page_id or "")

        record = {
            "download_id": download_id,
            "url": str(getattr(download, "url", "") or ""),
            "suggested_filename": str(getattr(download, "suggested_filename", "") or ""),
            "status": "pending",
            "saved_path": None,
            "failure": None,
            "created_at": time.time(),
            "page_id": page_id,
            "download": download,
        }
        store[download_id] = record
        self._log(run_state, "download_detected", self._normalize_download_record(record))

    def _attach_page_download_listener(self, run_state: RunState, page: Any) -> None:
        run_id = str(run_state.run_id)
        state = self._handlers_by_run.setdefault(run_id, {"page_handlers": {}, "context_page": None})
        page_handlers = state.setdefault("page_handlers", {})
        page_key = str(id(page))
        if page_key in page_handlers:
            return

        def _on_download(download: Any) -> None:
            asyncio.create_task(self._handle_download(run_state, download))

        page.on("download", _on_download)
        page_handlers[page_key] = {"page": page, "download": _on_download}

    async def attach_listener(self, run_state: RunState) -> None:
        context = getattr(run_state, "browser_context", None)
        if context is None:
            raise RuntimeError("Browser context is not initialized")

        self._ensure_run_storage(run_state)
        self.session_manager.sync_pages(run_state)
        for page in list(getattr(context, "pages", []) or []):
            self._attach_page_download_listener(run_state, page)

        run_id = str(run_state.run_id)
        state = self._handlers_by_run.setdefault(run_id, {"page_handlers": {}, "context_page": None})
        if state.get("context_page") is None:
            def _on_page(page: Any) -> None:
                self._attach_page_download_listener(run_state, page)

            context.on("page", _on_page)
            state["context_page"] = _on_page

    async def detach_listener(self, run_state: RunState) -> None:
        run_id = str(run_state.run_id)
        state = self._handlers_by_run.get(run_id) or {}
        context = getattr(run_state, "browser_context", None)
        context_page_handler = state.get("context_page")
        if context is not None and context_page_handler is not None:
            try:
                context.remove_listener("page", context_page_handler)
            except Exception:
                pass

        for handler_info in (state.get("page_handlers") or {}).values():
            page = handler_info.get("page")
            handler = handler_info.get("download")
            if page is None or handler is None:
                continue
            try:
                page.remove_listener("download", handler)
            except Exception:
                pass
        self._handlers_by_run.pop(run_id, None)

    async def list_downloads(self, run_state: RunState, limit: int = 100) -> Dict[str, Any]:
        items = list(self._ensure_run_storage(run_state).values())
        items.sort(key=lambda item: float(item.get("created_at") or 0.0), reverse=True)
        normalized = [self._normalize_download_record(item) for item in items[: max(1, int(limit or 100))]]
        return {"ok": True, "count": len(normalized), "downloads": normalized}

    async def get_download(self, run_state: RunState, download_id: str) -> Dict[str, Any]:
        record = self._ensure_run_storage(run_state).get(str(download_id))
        if record is None:
            raise ValueError(f"Unknown download_id: {download_id}")
        return {"ok": True, "download": self._normalize_download_record(record)}

    async def save_download(
        self,
        run_state: RunState,
        download_id: str,
        destination_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        record = self._ensure_run_storage(run_state).get(str(download_id))
        if record is None:
            raise ValueError(f"Unknown download_id: {download_id}")
        download = record.get("download")
        if download is None:
            raise RuntimeError(f"Download object is unavailable for {download_id}")

        base_dir = Path((run_state.metadata or {}).get("downloads_dir") or self.base_tmp_dir)
        target_path = Path(destination_path).expanduser() if destination_path else base_dir / (
            record.get("suggested_filename") or f"{download_id}.bin"
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        await download.save_as(str(target_path))
        record["saved_path"] = str(target_path)
        record["status"] = "saved"
        self._log(run_state, "download_saved", self._normalize_download_record(record))
        return {
            "ok": True,
            "download_id": str(download_id),
            "saved_path": str(target_path),
            "suggested_filename": record.get("suggested_filename"),
        }

    async def wait_for_download(
        self,
        run_state: RunState,
        timeout_ms: int = 15000,
    ) -> Dict[str, Any]:
        deadline = time.time() + (max(1, int(timeout_ms or 15000)) / 1000.0)
        seen = set(self._ensure_run_storage(run_state).keys())
        while time.time() < deadline:
            store = self._ensure_run_storage(run_state)
            new_ids = [item for item in store.keys() if item not in seen]
            if new_ids:
                record = store[new_ids[-1]]
                return {"ok": True, "download": self._normalize_download_record(record)}
            await asyncio.sleep(0.2)
        raise TimeoutError("Timed out waiting for a browser download")

    async def cleanup_tmp(self, run_state: Optional[RunState] = None, ttl_seconds: int = 86400) -> Dict[str, Any]:
        _ = ttl_seconds
        current = run_state or self._run_state
        if current is None:
            return {"ok": True, "removed": []}
        downloads_dir = Path((current.metadata or {}).get("downloads_dir") or self.base_tmp_dir)
        removed: List[str] = []
        if downloads_dir.exists():
            for child in downloads_dir.iterdir():
                if child.is_file():
                    child.unlink(missing_ok=True)
                    removed.append(str(child))
                elif child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                    removed.append(str(child))
        return {"ok": True, "removed": removed}

    @mcp_tool(
        name="browser_list_downloads",
        examples=["browser_list_downloads(limit=10)"],
    )
    async def mcp_browser_list_downloads(self, limit: int = 100) -> Dict[str, Any]:
        """
        List downloads detected during the current browser run.

        Use case
        - You want to see which files have been offered or saved by the site before deciding whether to inspect or persist one of them.

        Parameters
        - limit: Maximum number of downloads to return, ordered from newest to oldest.

        Return value
        - Dict with download inventory:
          - ok: Whether the listing succeeded.
          - count: Number of download records returned.
          - downloads: Array of normalized download records. Records may include `download_id`, `url`, `suggested_filename`, `status`, `mime_type`, `saved_path`, and timestamps.

        Concrete example (code)

        ```python
        browser_list_downloads(limit=10)
        ```
        """
        return await self.list_downloads(self._require_run_state(), limit=limit)

    @mcp_tool(
        name="browser_get_download",
        examples=["browser_get_download(download_id='download_1')"],
    )
    async def mcp_browser_get_download(self, download_id: str) -> Dict[str, Any]:
        """
        Retrieve metadata for one detected download.

        Use case
        - You already know the download id and want detailed metadata before deciding whether to save or ignore the file.

        Parameters
        - download_id: Download identifier returned by `browser_list_downloads` or `browser_wait_for_download`.

        Return value
        - Dict with one download record:
          - ok: Whether the lookup succeeded.
          - download: Normalized download metadata such as `download_id`, `url`, `suggested_filename`, `status`, `mime_type`, `saved_path`, and timestamps.

        Concrete example (code)

        ```python
        browser_get_download(download_id="download_1")
        ```
        """
        return await self.get_download(self._require_run_state(), download_id=download_id)

    @mcp_tool(
        name="browser_save_download",
        examples=[
            "browser_save_download(download_id='download_1')",
            "browser_save_download(download_id='download_1', destination_path='/tmp/file.pdf')",
        ],
    )
    async def mcp_browser_save_download(
        self,
        download_id: str,
        destination_path: str = "",
    ) -> Dict[str, Any]:
        """
        Save a detected browser download to disk.

        Use case
        - The page triggered a download and you want the file persisted to a known location for later inspection or handoff.

        Parameters
        - download_id: Download identifier returned by the downloads tools.
        - destination_path: Optional full destination path. If omitted, the tool saves into the run downloads directory using the suggested filename.

        Return value
        - Dict describing the saved file:
          - ok: Whether the save succeeded.
          - download_id: Download identifier that was saved.
          - saved_path: Final on-disk path of the saved file.
          - suggested_filename: Browser-provided filename hint, when available.

        Concrete example (code)

        ```python
        browser_save_download(download_id="download_1")
        browser_save_download(download_id="download_1", destination_path="/tmp/file.pdf")
        ```
        """
        return await self.save_download(
            self._require_run_state(),
            download_id=download_id,
            destination_path=destination_path or None,
        )

    @mcp_tool(
        name="browser_wait_for_download",
        examples=["browser_wait_for_download(timeout_ms=15000)"],
    )
    async def mcp_browser_wait_for_download(self, timeout_ms: int = 15000) -> Dict[str, Any]:
        """
        Wait until a new browser download is detected.

        Use case
        - You just clicked a download link or submit action and need to pause until the browser reports a new download.

        Parameters
        - timeout_ms: Maximum wait time in milliseconds before raising a timeout.

        Return value
        - Dict with the first newly detected download:
          - ok: Whether a new download was detected before timeout.
          - download: Normalized download metadata such as `download_id`, `url`, `suggested_filename`, `status`, `mime_type`, `saved_path`, and timestamps.

        Concrete example (code)

        ```python
        browser_wait_for_download(timeout_ms=15000)
        ```
        """
        return await self.wait_for_download(self._require_run_state(), timeout_ms=timeout_ms)
