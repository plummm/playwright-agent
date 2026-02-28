"""Network and console inspection feature scaffold."""

from __future__ import annotations

import asyncio
import base64
import inspect
import re
import time
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool

from .event_logger import BrowserEventLogger
from .models import RunState
from .page_actions import mcp_tool
from .session import BrowserSessionManager


class NetworkInspectorFeature:
    """Capture and inspect network + console activity."""

    def __init__(
        self,
        session_manager: BrowserSessionManager,
        event_logger: BrowserEventLogger,
    ):
        self.session_manager = session_manager
        self.event_logger = event_logger
        self._run_state: Optional[RunState] = None
        self._tools: List[Any] = []
        self._max_body_bytes = 1024 * 1024
        self._capture_enabled_by_run: Dict[str, bool] = {}
        self._attached_by_run: Dict[str, bool] = {}
        self._handlers_by_run: Dict[str, Dict[str, Any]] = {}
        self._tasks_by_run: Dict[str, set[asyncio.Task[Any]]] = {}
        self._history_by_run: Dict[str, Dict[str, Any]] = {}
        self._request_index_by_run: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._request_key_map_by_run: Dict[str, Dict[int, str]] = {}

    def set_run_state(self, run_state: Optional[RunState]) -> None:
        """Bind active run-state for MCP tool wrappers."""
        self._run_state = run_state

    def _require_run_state(self) -> RunState:
        if self._run_state is None:
            raise RuntimeError("NetworkInspectorFeature run_state is not set")
        return self._run_state

    def get_tools(self) -> List[Any]:
        if self._tools:
            return self._tools
        tools: List[Any] = []
        disabled_tool_names = {"browser_list_requests", "browser_list_console"}
        for method_name in dir(self):
            method = getattr(self, method_name, None)
            if not callable(method):
                continue
            if not bool(getattr(method, "_is_mcp_tool", False)):
                continue
            tool_name = str(getattr(method, "_mcp_name", method.__name__) or method.__name__)
            if tool_name in disabled_tool_names:
                continue
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

    async def attach_listeners(self, run_state: RunState) -> None:
        run_id = str(run_state.run_id)
        if self._attached_by_run.get(run_id):
            return
        page = self.session_manager.get_active_page(run_state)
        if page is None:
            page = await self.session_manager.new_page(run_state)
        if page is None:
            raise RuntimeError("No active browser page for listener attachment")

        self._ensure_run_storage(run_state)
        self._tasks_by_run.setdefault(run_id, set())

        def _on_request(request: Any) -> None:
            self._spawn(run_id, self._handle_request(run_state, request))

        def _on_response(response: Any) -> None:
            self._spawn(run_id, self._handle_response(run_state, response))

        def _on_request_failed(request: Any) -> None:
            self._spawn(run_id, self._handle_request_failed(run_state, request))

        def _on_console(msg: Any) -> None:
            self._spawn(run_id, self._handle_console(run_state, msg))

        def _on_page_error(err: Any) -> None:
            self._spawn(run_id, self._handle_page_error(run_state, err))

        def _on_websocket(ws: Any) -> None:
            self._spawn(run_id, self._handle_websocket_open(run_state, ws))

        def _on_download(download: Any) -> None:
            self._spawn(run_id, self._handle_download(run_state, download))

        page.on("request", _on_request)
        page.on("response", _on_response)
        page.on("requestfailed", _on_request_failed)
        page.on("console", _on_console)
        page.on("pageerror", _on_page_error)
        page.on("websocket", _on_websocket)
        page.on("download", _on_download)

        self._handlers_by_run[run_id] = {
            "page": page,
            "request": _on_request,
            "response": _on_response,
            "requestfailed": _on_request_failed,
            "console": _on_console,
            "pageerror": _on_page_error,
            "websocket": _on_websocket,
            "download": _on_download,
        }
        self._attached_by_run[run_id] = True

    async def detach_listeners(self, run_state: RunState) -> None:
        run_id = str(run_state.run_id)
        ctx = self._handlers_by_run.get(run_id) or {}
        page = ctx.get("page")
        if page is not None:
            for evt in (
                "request",
                "response",
                "requestfailed",
                "console",
                "pageerror",
                "websocket",
                "download",
            ):
                handler = ctx.get(evt)
                if handler is None:
                    continue
                try:
                    page.remove_listener(evt, handler)
                except Exception:
                    pass
        self._handlers_by_run.pop(run_id, None)
        self._attached_by_run[run_id] = False
        tasks = list(self._tasks_by_run.get(run_id, set()))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks_by_run.pop(run_id, None)

    async def start_capture(self, run_state: RunState) -> Dict[str, Any]:
        run_id = str(run_state.run_id)
        self._ensure_run_storage(run_state)
        self._capture_enabled_by_run[run_id] = True
        payload = {
            "run_id": run_id,
            "request_id": str(run_state.request_id or ""),
            "event_type": "capture_started",
            "ts": time.time(),
            "payload": {"capture_enabled": True},
        }
        try:
            self.event_logger.log_action_event(payload)
        except Exception:
            pass
        return {"ok": True, "run_id": run_id, "capture_enabled": True}

    async def stop_capture(self, run_state: RunState) -> Dict[str, Any]:
        run_id = str(run_state.run_id)
        self._capture_enabled_by_run[run_id] = False
        payload = {
            "run_id": run_id,
            "request_id": str(run_state.request_id or ""),
            "event_type": "capture_stopped",
            "ts": time.time(),
            "payload": {"capture_enabled": False},
        }
        try:
            self.event_logger.log_action_event(payload)
        except Exception:
            pass
        history = self._history_by_run.get(run_id) or {}
        return {
            "ok": True,
            "run_id": run_id,
            "capture_enabled": False,
            "counts": {
                "requests": len(history.get("requests", [])),
                "console": len(history.get("console", [])),
                "websocket": len(history.get("websocket", [])),
                "errors": len(history.get("errors", [])),
            },
        }

    async def list_requests(
        self,
        run_state: RunState,
        limit: int = 100,
        offset: int = 0,
        resource_type: str = "",
        mime_type: str = "",
        status_min: Optional[int] = None,
        status_max: Optional[int] = None,
        url_contains: str = "",
        url_regex: str = "",
    ) -> Dict[str, Any]:
        run_id = str(run_state.run_id)
        items = list((self._history_by_run.get(run_id) or {}).get("requests", []))
        items = self._filter_requests(
            items=items,
            resource_type=resource_type,
            mime_type=mime_type,
            status_min=status_min,
            status_max=status_max,
            url_contains=url_contains,
            url_regex=url_regex,
        )
        return self._paginate(items, limit=limit, offset=offset)

    async def list_console(
        self,
        run_state: RunState,
        limit: int = 100,
        offset: int = 0,
        level: str = "",
        text_contains: str = "",
    ) -> Dict[str, Any]:
        run_id = str(run_state.run_id)
        entries = list((self._history_by_run.get(run_id) or {}).get("console", []))
        level_norm = str(level or "").strip().lower()
        text_norm = str(text_contains or "").strip().lower()
        filtered: List[Dict[str, Any]] = []
        for entry in entries:
            if level_norm and str(entry.get("level") or "").lower() != level_norm:
                continue
            if text_norm and text_norm not in str(entry.get("text") or "").lower():
                continue
            filtered.append(entry)
        return self._paginate(filtered, limit=limit, offset=offset)

    async def list_websocket_frames(
        self,
        run_state: RunState,
        limit: int = 100,
        offset: int = 0,
        event_type: str = "",
    ) -> Dict[str, Any]:
        run_id = str(run_state.run_id)
        entries = list((self._history_by_run.get(run_id) or {}).get("websocket", []))
        et = str(event_type or "").strip().lower()
        if et:
            entries = [x for x in entries if str(x.get("event_type") or "").lower() == et]
        return self._paginate(entries, limit=limit, offset=offset)

    async def get_resource_source(
        self,
        run_state: RunState,
        url: str = "",
        request_id: str = "",
        encoding: Optional[str] = None,
    ) -> Dict[str, Any]:
        run_id = str(run_state.run_id)
        index = self._request_index_by_run.get(run_id) or {}
        item: Optional[Dict[str, Any]] = None
        rid = str(request_id or "").strip()
        if rid:
            item = index.get(rid)
        if item is None:
            target_url = str(url or "").strip()
            if target_url:
                for row in reversed((self._history_by_run.get(run_id) or {}).get("requests", [])):
                    if str(row.get("url") or "") == target_url:
                        item = row
                        break
        if item is None:
            return {"ok": False, "error": "resource_not_found"}

        body_kind = str(item.get("response_body_kind") or "")
        body_value = item.get("response_body")
        requested_encoding = str(encoding or "").strip().lower()
        if requested_encoding == "base64" and body_kind == "text" and isinstance(body_value, str):
            body_value = base64.b64encode(body_value.encode("utf-8", errors="replace")).decode("ascii")
            body_kind = "base64"
        elif requested_encoding == "text" and body_kind == "base64" and isinstance(body_value, str):
            try:
                body_value = base64.b64decode(body_value.encode("ascii")).decode("utf-8", errors="replace")
                body_kind = "text"
            except Exception:
                pass

        return {
            "ok": True,
            "request_id": item.get("request_id"),
            "url": item.get("url"),
            "mime_type": item.get("mime_type"),
            "status_code": item.get("status_code"),
            "response_headers": item.get("response_headers"),
            "response_body_kind": body_kind,
            "response_body": body_value,
            "response_body_size": item.get("response_body_size"),
            "response_body_truncated": bool(item.get("response_body_truncated")),
            "is_download": bool(item.get("is_download")),
        }

    async def fetch_url_content(
        self,
        run_state: RunState,
        url: str,
        *,
        timeout_ms: int = 10000,
    ) -> Dict[str, Any]:
        """
        Deterministically fetch URL content via Playwright context request.

        This helper is intended for agent-internal fallback when captured response
        body is missing from network logs.
        """
        target = str(url or "").strip()
        if not target:
            return {"ok": False, "error": "url_required"}

        page = self.session_manager.get_active_page(run_state)
        if page is None:
            page = await self.session_manager.new_page(run_state)
        if page is None:
            return {"ok": False, "error": "no_active_page"}

        request_ctx = getattr(getattr(page, "context", None), "request", None)
        if request_ctx is None:
            return {"ok": False, "error": "request_context_unavailable"}

        try:
            response = await request_ctx.get(target, timeout=int(timeout_ms))
        except Exception as e:
            return {"ok": False, "error": f"fetch_failed:{type(e).__name__}:{e}"}

        try:
            status_code = int(getattr(response, "status", 0) or 0)
        except Exception:
            status_code = 0
        headers = {}
        try:
            headers = dict(await response.all_headers())
        except Exception:
            try:
                headers = dict(getattr(response, "headers", {}) or {})
            except Exception:
                headers = {}
        mime_type = str(headers.get("content-type") or "")
        try:
            body_bytes = await response.body()
        except Exception:
            body_bytes = b""
        body_meta = self._serialize_body(body=bytes(body_bytes or b""), mime_type=mime_type)
        return {
            "ok": True,
            "url": target,
            "status_code": status_code,
            "headers": headers,
            "mime_type": mime_type,
            "body_kind": body_meta.get("kind"),
            "body": body_meta.get("body"),
            "body_size": body_meta.get("size"),
            "body_truncated": body_meta.get("truncated"),
        }

    def get_delta_cursor(self, run_state: RunState) -> Dict[str, int]:
        history = self._history_by_run.get(str(run_state.run_id)) or {}
        return {
            "requests": len(history.get("requests", [])),
            "console": len(history.get("console", [])),
            "websocket": len(history.get("websocket", [])),
            "errors": len(history.get("errors", [])),
        }

    def get_delta_since(self, run_state: RunState, cursor: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        run_id = str(run_state.run_id)
        history = self._history_by_run.get(run_id) or {}
        c = cursor if isinstance(cursor, dict) else {}
        req_i = max(0, int(c.get("requests", 0) or 0))
        con_i = max(0, int(c.get("console", 0) or 0))
        ws_i = max(0, int(c.get("websocket", 0) or 0))
        err_i = max(0, int(c.get("errors", 0) or 0))
        requests = list(history.get("requests", []))[req_i:]
        console = list(history.get("console", []))[con_i:]
        websocket = list(history.get("websocket", []))[ws_i:]
        errors = list(history.get("errors", []))[err_i:]
        return {
            "run_id": run_id,
            "requests": requests,
            "console": console,
            "websocket": websocket,
            "errors": errors,
            "cursor": self.get_delta_cursor(run_state),
        }

    def _ensure_run_storage(self, run_state: RunState) -> None:
        run_id = str(run_state.run_id)
        if run_id not in self._history_by_run:
            self._history_by_run[run_id] = {
                "requests": [],
                "console": [],
                "websocket": [],
                "errors": [],
            }
        self._request_index_by_run.setdefault(run_id, {})
        self._request_key_map_by_run.setdefault(run_id, {})

    def _capture_enabled(self, run_state: RunState) -> bool:
        run_id = str(run_state.run_id)
        return bool(self._capture_enabled_by_run.get(run_id, False))

    def _capture_network_enabled(self, run_state: RunState) -> bool:
        return bool((run_state.metadata or {}).get("capture_network", True))

    def _capture_console_enabled(self, run_state: RunState) -> bool:
        return bool((run_state.metadata or {}).get("capture_console", True))

    def _spawn(self, run_id: str, coro: Any) -> None:
        task = asyncio.create_task(coro)
        bucket = self._tasks_by_run.setdefault(run_id, set())
        bucket.add(task)
        task.add_done_callback(lambda t: bucket.discard(t))

    def _next_request_id(self) -> str:
        return str(uuid.uuid4())

    async def _handle_request(self, run_state: RunState, request: Any) -> None:
        if not self._capture_enabled(run_state) or not self._capture_network_enabled(run_state):
            return
        self._ensure_run_storage(run_state)
        run_id = str(run_state.run_id)
        req_id = self._next_request_id()
        request_key = id(request)
        self._request_key_map_by_run[run_id][request_key] = req_id

        post_data = None
        try:
            post_data = request.post_data
        except Exception:
            post_data = None
        payload: Dict[str, Any] = {
            "run_id": run_id,
            "request_id": req_id,
            "event_type": "request",
            "ts": time.time(),
            "method": str(getattr(request, "method", "") or ""),
            "url": str(getattr(request, "url", "") or ""),
            "resource_type": str(getattr(request, "resource_type", "") or ""),
            "request_headers": dict(getattr(request, "headers", {}) or {}),
            "post_data": post_data,
            "page_url": str(getattr(getattr(request, "frame", None), "url", "") or ""),
            "status_code": None,
            "response_headers": {},
            "mime_type": "",
            "response_body": None,
            "response_body_kind": None,
            "response_body_size": 0,
            "response_body_truncated": False,
            "error_text": None,
            "is_download": False,
            "download_reason": None,
        }
        self._history_by_run[run_id]["requests"].append(payload)
        self._request_index_by_run[run_id][req_id] = payload
        try:
            self.event_logger.log_network_event(
                {
                    "run_id": run_id,
                    "request_id": str(run_state.request_id or ""),
                    "event_type": "request",
                    "ts": payload["ts"],
                    "payload": payload,
                }
            )
        except Exception:
            pass

    async def _handle_response(self, run_state: RunState, response: Any) -> None:
        if not self._capture_enabled(run_state) or not self._capture_network_enabled(run_state):
            return
        self._ensure_run_storage(run_state)
        run_id = str(run_state.run_id)
        request = getattr(response, "request", None)
        request_key = id(request) if request is not None else None
        req_id = self._request_key_map_by_run.get(run_id, {}).get(request_key) if request_key is not None else None
        if not req_id:
            # Fallback for out-of-order events.
            req_id = self._next_request_id()
            self._request_key_map_by_run[run_id][request_key or 0] = req_id
            fallback = {
                "run_id": run_id,
                "request_id": req_id,
                "event_type": "request",
                "ts": time.time(),
                "method": str(getattr(request, "method", "") or ""),
                "url": str(getattr(request, "url", "") or ""),
                "resource_type": str(getattr(request, "resource_type", "") or ""),
                "request_headers": dict(getattr(request, "headers", {}) or {}),
                "post_data": getattr(request, "post_data", None),
                "page_url": "",
                "status_code": None,
                "response_headers": {},
                "mime_type": "",
                "response_body": None,
                "response_body_kind": None,
                "response_body_size": 0,
                "response_body_truncated": False,
                "error_text": None,
                "is_download": False,
                "download_reason": None,
            }
            self._history_by_run[run_id]["requests"].append(fallback)
            self._request_index_by_run[run_id][req_id] = fallback

        record = self._request_index_by_run[run_id].get(req_id)
        if record is None:
            return
        headers = {}
        try:
            headers = dict(getattr(response, "headers", {}) or {})
        except Exception:
            headers = {}
        record["event_type"] = "response"
        record["status_code"] = int(getattr(response, "status", 0) or 0)
        record["response_headers"] = headers
        record["mime_type"] = str(headers.get("content-type") or "")

        capture_bodies = bool((run_state.metadata or {}).get("capture_response_bodies", True))
        if capture_bodies:
            try:
                body_bytes = await response.body()
            except Exception:
                body_bytes = None
            if isinstance(body_bytes, (bytes, bytearray)):
                body_meta = self._serialize_body(
                    body=bytes(body_bytes),
                    mime_type=str(record.get("mime_type") or ""),
                )
                record["response_body"] = body_meta.get("body")
                record["response_body_kind"] = body_meta.get("kind")
                record["response_body_size"] = body_meta.get("size")
                record["response_body_truncated"] = body_meta.get("truncated")

        is_download, reason = self._detect_download(
            url=str(record.get("url") or ""),
            headers=headers,
            mime_type=str(record.get("mime_type") or ""),
            resource_type=str(record.get("resource_type") or ""),
        )
        record["is_download"] = is_download
        record["download_reason"] = reason

        try:
            self.event_logger.log_network_event(
                {
                    "run_id": run_id,
                    "request_id": str(run_state.request_id or ""),
                    "event_type": "response",
                    "ts": time.time(),
                    "payload": record,
                }
            )
        except Exception:
            pass

    async def _handle_request_failed(self, run_state: RunState, request: Any) -> None:
        if not self._capture_enabled(run_state) or not self._capture_network_enabled(run_state):
            return
        self._ensure_run_storage(run_state)
        run_id = str(run_state.run_id)
        request_key = id(request)
        req_id = self._request_key_map_by_run.get(run_id, {}).get(request_key)
        if not req_id:
            req_id = self._next_request_id()
            self._request_key_map_by_run[run_id][request_key] = req_id
            self._request_index_by_run[run_id][req_id] = {
                "run_id": run_id,
                "request_id": req_id,
                "event_type": "request_failed",
                "ts": time.time(),
                "method": str(getattr(request, "method", "") or ""),
                "url": str(getattr(request, "url", "") or ""),
                "resource_type": str(getattr(request, "resource_type", "") or ""),
                "request_headers": dict(getattr(request, "headers", {}) or {}),
                "post_data": getattr(request, "post_data", None),
                "response_headers": {},
                "status_code": None,
                "mime_type": "",
                "response_body": None,
                "response_body_kind": None,
                "response_body_size": 0,
                "response_body_truncated": False,
                "error_text": None,
                "is_download": False,
                "download_reason": None,
            }
            self._history_by_run[run_id]["requests"].append(self._request_index_by_run[run_id][req_id])

        record = self._request_index_by_run[run_id].get(req_id)
        if record is None:
            return
        record["event_type"] = "request_failed"
        failure = getattr(request, "failure", None)
        error_text = None
        if callable(failure):
            try:
                failure_obj = failure()
                if isinstance(failure_obj, dict):
                    error_text = str(failure_obj.get("errorText") or "")
                else:
                    error_text = str(failure_obj or "")
            except Exception:
                error_text = None
        record["error_text"] = error_text
        try:
            self.event_logger.log_network_event(
                {
                    "run_id": run_id,
                    "request_id": str(run_state.request_id or ""),
                    "event_type": "request_failed",
                    "ts": time.time(),
                    "payload": record,
                }
            )
        except Exception:
            pass

    async def _handle_console(self, run_state: RunState, msg: Any) -> None:
        if not self._capture_enabled(run_state) or not self._capture_console_enabled(run_state):
            return
        run_id = str(run_state.run_id)
        level = str(getattr(msg, "type", "") or "log")
        text = str(getattr(msg, "text", "") or "")
        location = {}
        try:
            location = dict(getattr(msg, "location", {}) or {})
        except Exception:
            location = {}
        args_json: List[Any] = []
        for arg in list(getattr(msg, "args", []) or []):
            try:
                args_json.append(await arg.json_value())
            except Exception:
                args_json.append(str(arg))
        entry = {
            "run_id": run_id,
            "ts": time.time(),
            "level": level,
            "text": text,
            "location": location,
            "args": args_json,
            "page_url": str(getattr(getattr(msg, "page", None), "url", "") or ""),
        }
        self._history_by_run[run_id]["console"].append(entry)
        try:
            self.event_logger.log_console_event(
                {
                    "run_id": run_id,
                    "request_id": str(run_state.request_id or ""),
                    "event_type": "console",
                    "ts": entry["ts"],
                    "payload": entry,
                }
            )
        except Exception:
            pass

    async def _handle_page_error(self, run_state: RunState, err: Any) -> None:
        if not self._capture_enabled(run_state) or not self._capture_console_enabled(run_state):
            return
        run_id = str(run_state.run_id)
        entry = {
            "run_id": run_id,
            "ts": time.time(),
            "level": "pageerror",
            "text": str(err or ""),
        }
        self._history_by_run[run_id]["errors"].append(entry)
        self._history_by_run[run_id]["console"].append(entry)
        try:
            self.event_logger.log_console_event(
                {
                    "run_id": run_id,
                    "request_id": str(run_state.request_id or ""),
                    "event_type": "pageerror",
                    "ts": entry["ts"],
                    "payload": entry,
                }
            )
        except Exception:
            pass

    async def _handle_websocket_open(self, run_state: RunState, ws: Any) -> None:
        if not self._capture_enabled(run_state) or not self._capture_network_enabled(run_state):
            return
        run_id = str(run_state.run_id)
        ws_id = str(uuid.uuid4())
        open_evt = {
            "run_id": run_id,
            "ts": time.time(),
            "event_type": "websocket_open",
            "websocket_id": ws_id,
            "url": str(getattr(ws, "url", "") or ""),
        }
        self._history_by_run[run_id]["websocket"].append(open_evt)
        try:
            self.event_logger.log_network_event(
                {
                    "run_id": run_id,
                    "request_id": str(run_state.request_id or ""),
                    "event_type": "websocket_open",
                    "ts": open_evt["ts"],
                    "payload": open_evt,
                }
            )
        except Exception:
            pass

        # Frame events are best-effort and version-sensitive.
        def _on_frame_sent(payload: Any) -> None:
            self._spawn(run_id, self._handle_websocket_frame(run_state, ws_id, "sent", payload))

        def _on_frame_received(payload: Any) -> None:
            self._spawn(run_id, self._handle_websocket_frame(run_state, ws_id, "received", payload))

        def _on_close() -> None:
            self._spawn(run_id, self._handle_websocket_close(run_state, ws_id))

        try:
            ws.on("framesent", _on_frame_sent)
            ws.on("framereceived", _on_frame_received)
            ws.on("close", _on_close)
        except Exception:
            return

    async def _handle_websocket_frame(
        self,
        run_state: RunState,
        websocket_id: str,
        direction: str,
        payload: Any,
    ) -> None:
        if not self._capture_enabled(run_state) or not self._capture_network_enabled(run_state):
            return
        run_id = str(run_state.run_id)
        text_payload: Optional[str] = None
        binary_payload: Optional[str] = None
        raw = payload
        if isinstance(raw, str):
            text_payload = raw
        elif isinstance(raw, (bytes, bytearray)):
            b = bytes(raw)
            if len(b) > self._max_body_bytes:
                b = b[: self._max_body_bytes]
            binary_payload = base64.b64encode(b).decode("ascii")
        else:
            text_payload = str(raw)
        evt = {
            "run_id": run_id,
            "ts": time.time(),
            "event_type": f"websocket_frame_{direction}",
            "websocket_id": websocket_id,
            "text": text_payload,
            "base64": binary_payload,
            "truncated": isinstance(payload, (bytes, bytearray)) and len(payload) > self._max_body_bytes,
        }
        self._history_by_run[run_id]["websocket"].append(evt)
        try:
            self.event_logger.log_network_event(
                {
                    "run_id": run_id,
                    "request_id": str(run_state.request_id or ""),
                    "event_type": evt["event_type"],
                    "ts": evt["ts"],
                    "payload": evt,
                }
            )
        except Exception:
            pass

    async def _handle_websocket_close(self, run_state: RunState, websocket_id: str) -> None:
        if not self._capture_enabled(run_state) or not self._capture_network_enabled(run_state):
            return
        run_id = str(run_state.run_id)
        evt = {
            "run_id": run_id,
            "ts": time.time(),
            "event_type": "websocket_close",
            "websocket_id": websocket_id,
        }
        self._history_by_run[run_id]["websocket"].append(evt)
        try:
            self.event_logger.log_network_event(
                {
                    "run_id": run_id,
                    "request_id": str(run_state.request_id or ""),
                    "event_type": "websocket_close",
                    "ts": evt["ts"],
                    "payload": evt,
                }
            )
        except Exception:
            pass

    async def _handle_download(self, run_state: RunState, download: Any) -> None:
        if not self._capture_enabled(run_state) or not self._capture_network_enabled(run_state):
            return
        run_id = str(run_state.run_id)
        download_url = str(getattr(download, "url", "") or "")
        for row in reversed((self._history_by_run.get(run_id) or {}).get("requests", [])):
            if str(row.get("url") or "") == download_url:
                row["is_download"] = True
                row["download_reason"] = "download_event"
                break
        evt = {
            "run_id": run_id,
            "request_id": str(run_state.request_id or ""),
            "event_type": "download",
            "ts": time.time(),
            "payload": {
                "url": download_url,
                "suggested_filename": str(getattr(download, "suggested_filename", "") or ""),
            },
        }
        try:
            self.event_logger.log_download_event(evt)
        except Exception:
            pass

    def _serialize_body(self, *, body: bytes, mime_type: str) -> Dict[str, Any]:
        size = len(body)
        truncated = size > self._max_body_bytes
        clipped = body[: self._max_body_bytes] if truncated else body
        mime = str(mime_type or "").lower()
        likely_text = (
            mime.startswith("text/")
            or "json" in mime
            or "xml" in mime
            or "javascript" in mime
            or "svg" in mime
        )
        if likely_text:
            text = clipped.decode("utf-8", errors="replace")
            return {"kind": "text", "body": text, "size": size, "truncated": truncated}
        return {
            "kind": "base64",
            "body": base64.b64encode(clipped).decode("ascii"),
            "size": size,
            "truncated": truncated,
        }

    def _detect_download(
        self,
        *,
        url: str,
        headers: Dict[str, Any],
        mime_type: str,
        resource_type: str,
    ) -> tuple[bool, Optional[str]]:
        h = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
        disposition = h.get("content-disposition", "").lower()
        if "attachment" in disposition or "filename=" in disposition:
            return True, "content_disposition"
        mime = str(mime_type or "").lower()
        downloadable_mimes = (
            "application/octet-stream",
            "application/zip",
            "application/pdf",
            "application/vnd",
            "application/x-",
            "image/",
            "audio/",
            "video/",
        )
        if any(mime.startswith(prefix) for prefix in downloadable_mimes):
            return True, "mime_type"
        lower_url = str(url or "").lower()
        download_ext = (
            ".zip",
            ".pdf",
            ".dmg",
            ".pkg",
            ".exe",
            ".tar",
            ".gz",
            ".7z",
            ".mp4",
            ".mp3",
            ".png",
            ".jpg",
            ".jpeg",
            ".csv",
        )
        if any(lower_url.endswith(ext) for ext in download_ext):
            return True, "url_extension"
        if str(resource_type or "").lower() == "media":
            return True, "resource_type"
        return False, None

    def _filter_requests(
        self,
        *,
        items: List[Dict[str, Any]],
        resource_type: str,
        mime_type: str,
        status_min: Optional[int],
        status_max: Optional[int],
        url_contains: str,
        url_regex: str,
    ) -> List[Dict[str, Any]]:
        resource_norm = str(resource_type or "").strip().lower()
        mime_norm = str(mime_type or "").strip().lower()
        contains_norm = str(url_contains or "").strip().lower()
        regex_obj = None
        if str(url_regex or "").strip():
            try:
                regex_obj = re.compile(str(url_regex))
            except Exception:
                regex_obj = None
        out: List[Dict[str, Any]] = []
        for item in items:
            if resource_norm and str(item.get("resource_type") or "").lower() != resource_norm:
                continue
            if mime_norm and mime_norm not in str(item.get("mime_type") or "").lower():
                continue
            status = item.get("status_code")
            if status_min is not None:
                try:
                    if int(status or 0) < int(status_min):
                        continue
                except Exception:
                    continue
            if status_max is not None:
                try:
                    if int(status or 0) > int(status_max):
                        continue
                except Exception:
                    continue
            url = str(item.get("url") or "")
            if contains_norm and contains_norm not in url.lower():
                continue
            if regex_obj is not None and not regex_obj.search(url):
                continue
            out.append(item)
        return out

    def _paginate(self, items: List[Dict[str, Any]], *, limit: int, offset: int) -> Dict[str, Any]:
        safe_limit = max(1, min(500, int(limit or 100)))
        safe_offset = max(0, int(offset or 0))
        total = len(items)
        page = items[safe_offset : safe_offset + safe_limit]
        has_more = (safe_offset + safe_limit) < total
        return {
            "ok": True,
            "total": total,
            "offset": safe_offset,
            "limit": safe_limit,
            "has_more": has_more,
            "items": page,
        }

    @mcp_tool(
        name="browser_list_requests",
        examples=[
            "browser_list_requests(limit=50, offset=0)",
            "browser_list_requests(mime_type='application/json', url_contains='api')",
        ],
    )
    async def mcp_browser_list_requests(
        self,
        limit: int = 100,
        offset: int = 0,
        resource_type: str = "",
        mime_type: str = "",
        status_min: Optional[int] = None,
        status_max: Optional[int] = None,
        url_contains: str = "",
        url_regex: str = "",
    ) -> Dict[str, Any]:
        """
        List captured network request/response records with filtering and pagination.

        Args:
            limit (optional): Max records to return in this page.
                Effective range: [1..500]. Default: `100`.
            offset (optional): Zero-based index of first record. Must be integer >= 0.
                Default: `0`.
            resource_type (optional): Exact Playwright resource type filter.
                Common values: [`document`, `stylesheet`, `script`, `image`, `media`,
                `font`, `xhr`, `fetch`, `eventsource`, `websocket`, `manifest`, `other`].
                Empty means no resource-type filter.
            mime_type (optional): Case-insensitive MIME substring filter
                (e.g. `application/json`, `text/html`, `javascript`).
            status_min (optional): Minimum HTTP status code (inclusive), e.g. `400`.
            status_max (optional): Maximum HTTP status code (inclusive), e.g. `599`.
            url_contains (optional): Case-insensitive URL substring filter.
            url_regex (optional): URL regex filter (Python regex syntax).

        Returns:
            Dict pagination envelope:
            - ok (bool): Whether query succeeded.
            - total (int): Total records after filters.
            - offset (int): Applied offset.
            - limit (int): Applied limit.
            - has_more (bool): Whether more pages exist.
            - items (list[dict]): Full raw request/response records.

        Examples:
            await mcp_browser_list_requests(limit=50, offset=0)
            await mcp_browser_list_requests(mime_type="application/json", url_contains="/api/")
            await mcp_browser_list_requests(status_min=400, status_max=599)
        """
        return await self.list_requests(
            self._require_run_state(),
            limit=limit,
            offset=offset,
            resource_type=resource_type,
            mime_type=mime_type,
            status_min=status_min,
            status_max=status_max,
            url_contains=url_contains,
            url_regex=url_regex,
        )

    @mcp_tool(
        name="browser_list_console",
        examples=[
            "browser_list_console(limit=100, offset=0)",
            "browser_list_console(level='error', text_contains='failed')",
        ],
    )
    async def mcp_browser_list_console(
        self,
        limit: int = 100,
        offset: int = 0,
        level: str = "",
        text_contains: str = "",
    ) -> Dict[str, Any]:
        """
        List captured browser console/pageerror entries with filtering and pagination.

        Args:
            limit (optional): Max records to return in this page.
                Effective range: [1..500]. Default: `100`.
            offset (optional): Zero-based index of first record. Must be integer >= 0.
                Default: `0`.
            level (optional): Exact console level filter.
                Common values: [`log`, `info`, `warn`, `error`, `debug`, `pageerror`].
                Empty means no level filter.
            text_contains (optional): Case-insensitive text substring filter.

        Returns:
            Dict pagination envelope:
            - ok (bool): Whether query succeeded.
            - total (int): Total records after filters.
            - offset (int): Applied offset.
            - limit (int): Applied limit.
            - has_more (bool): Whether more pages exist.
            - items (list[dict]): Raw console/error entries.

        Examples:
            await mcp_browser_list_console(limit=100, offset=0)
            await mcp_browser_list_console(level="error", text_contains="failed")
        """
        return await self.list_console(
            self._require_run_state(),
            limit=limit,
            offset=offset,
            level=level,
            text_contains=text_contains,
        )

    @mcp_tool(
        name="browser_get_resource_source",
        examples=[
            "browser_get_resource_source(request_id='...')",
            "browser_get_resource_source(url='https://example.com/app.js', encoding='text')",
        ],
    )
    async def mcp_browser_get_resource_source(
        self,
        url: str = "",
        request_id: str = "",
        encoding: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fetch stored raw response payload for a captured resource by `request_id` or URL.

        Args:
            url (optional): Exact resource URL to resolve (used when `request_id` is omitted).
            request_id (optional): Stable captured request ID (preferred lookup key).
            encoding (optional): Output conversion hint.
                - `text`: decode base64 payload to UTF-8 when possible
                - `base64`: base64-encode text payload
                - None: keep stored representation
                Allowed values: [`text`, `base64`, null/omitted].

        Returns:
            Dict with payload details:
            - ok (bool): Whether resource lookup succeeded.
            - request_id (str): Captured request identifier.
            - url (str): Resource URL.
            - mime_type (str): Response MIME type.
            - status_code (int | null): HTTP status code.
            - response_headers (dict): Raw response headers.
            - response_body_kind (str): `text` or `base64`.
            - response_body (str | null): Raw response payload.
            - response_body_size (int): Original body size in bytes.
            - response_body_truncated (bool): Whether payload exceeded capture cap.
            - is_download (bool): Download heuristic outcome.

        Examples:
            await mcp_browser_get_resource_source(request_id="0f8fad5b-d9cb-469f-a165-70867728950e")
            await mcp_browser_get_resource_source(url="https://example.com/app.js", encoding="text")
        """
        return await self.get_resource_source(
            self._require_run_state(),
            url=url,
            request_id=request_id,
            encoding=encoding,
        )

    @mcp_tool(
        name="browser_list_websocket_frames",
        examples=[
            "browser_list_websocket_frames(limit=100, offset=0)",
            "browser_list_websocket_frames(event_type='websocket_frame_received')",
        ],
    )
    async def mcp_browser_list_websocket_frames(
        self,
        limit: int = 100,
        offset: int = 0,
        event_type: str = "",
    ) -> Dict[str, Any]:
        """
        List captured websocket events/frames with pagination.

        Args:
            limit (optional): Max websocket events to return.
                Effective range: [1..500]. Default: `100`.
            offset (optional): Zero-based index of first websocket event.
                Must be integer >= 0. Default: `0`.
            event_type (optional): Exact websocket event filter.
                Allowed values: [`websocket_open`, `websocket_frame_received`,
                `websocket_frame_sent`, `websocket_close`].
                Empty means no event-type filter.

        Returns:
            Dict pagination envelope:
            - ok (bool): Whether query succeeded.
            - total (int): Total websocket records after filter.
            - offset (int): Applied offset.
            - limit (int): Applied limit.
            - has_more (bool): Whether more pages exist.
            - items (list[dict]): Raw websocket event/frame records.

        Examples:
            await mcp_browser_list_websocket_frames(limit=100, offset=0)
            await mcp_browser_list_websocket_frames(event_type="websocket_frame_received")
        """
        return await self.list_websocket_frames(
            self._require_run_state(),
            limit=limit,
            offset=offset,
            event_type=event_type,
        )
