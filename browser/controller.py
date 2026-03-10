"""High-level browser controller for the Playwright agent."""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .event_logger import BrowserEventLogger
from .models import RunState
from .session import BrowserSessionManager


class BrowserController:
    """Coordinate browser state mutations for a single loop-driven run."""

    def __init__(
        self,
        session_manager: BrowserSessionManager,
        event_logger: Optional[BrowserEventLogger] = None,
        network_inspector: Optional[Any] = None,
    ):
        self.session_manager = session_manager
        self.event_logger = event_logger
        self.network_inspector = network_inspector
        self._run_state: Optional[RunState] = None

    def set_run_state(self, run_state: Optional[RunState]) -> None:
        self._run_state = run_state

    def set_network_inspector(self, network_inspector: Optional[Any]) -> None:
        self.network_inspector = network_inspector

    def _require_run_state(self) -> RunState:
        if self._run_state is None:
            raise RuntimeError("BrowserController run_state is not set")
        return self._run_state

    def _timeout_ms(self, run_state: RunState) -> int:
        try:
            return int((run_state.metadata or {}).get("timeout_ms", 30000) or 30000)
        except Exception:
            return 30000

    def _snapshot_followup(self, reason: str) -> Dict[str, Any]:
        message = str(reason or "").strip() or "Verify the updated browser state."
        return {
            "snapshot_stale": True,
            "suggested_next_tool": "browser_snapshot",
            "suggested_next_tool_reason": message,
        }

    def _log_action(self, run_state: RunState, *, action: str, payload: Dict[str, Any]) -> None:
        if self.event_logger is None:
            return
        try:
            self.event_logger.log_action_event(
                {
                    "run_id": run_state.run_id,
                    "request_id": run_state.request_id,
                    "event_type": action,
                    "ts": time.time(),
                    "payload": payload,
                }
            )
        except Exception:
            pass

    def _network_cursor(self, run_state: RunState) -> Optional[Dict[str, int]]:
        inspector = self.network_inspector
        if inspector is None:
            return None
        get_cursor = getattr(inspector, "get_delta_cursor", None)
        if not callable(get_cursor):
            return None
        try:
            return get_cursor(run_state)
        except Exception:
            return None

    def _network_delta(self, run_state: RunState, cursor: Optional[Dict[str, int]]) -> Optional[Dict[str, Any]]:
        inspector = self.network_inspector
        if inspector is None or cursor is None:
            return None
        get_delta = getattr(inspector, "get_delta_since", None)
        if not callable(get_delta):
            return None
        try:
            return get_delta(run_state, cursor)
        except Exception:
            return None

    async def ensure_page(self, run_state: Optional[RunState] = None):
        current = run_state or self._require_run_state()
        page = self.session_manager.get_active_page(current)
        if page is None and getattr(current, "browser_context", None) is not None:
            page = await self.session_manager.new_page(current)
        if page is None:
            raise RuntimeError("No active browser page. Start a browser session first.")
        return page

    async def describe_state(self, run_state: Optional[RunState] = None) -> Dict[str, Any]:
        current = run_state or self._require_run_state()
        page = await self.ensure_page(current)
        title = ""
        try:
            title = str(await page.title() or "")
        except Exception:
            title = ""
        tabs = await self.session_manager.describe_tabs(current)
        return {
            "active_tab_id": current.active_page_id,
            "url": str(getattr(page, "url", "") or current.current_url or ""),
            "title": title,
            "tabs": tabs,
            "snapshot_id": getattr(current.snapshot, "snapshot_id", None),
        }

    async def navigate(self, run_state: RunState, url: str, wait_until: str = "load") -> Dict[str, Any]:
        target = str(url or "").strip()
        if not target:
            raise ValueError("URL is required")

        page = await self.ensure_page(run_state)
        valid_wait_until = {"load", "domcontentloaded", "networkidle", "commit"}
        wait_mode = wait_until if wait_until in valid_wait_until else "load"
        cursor = self._network_cursor(run_state)
        response = await page.goto(target, wait_until=wait_mode, timeout=self._timeout_ms(run_state))
        self.session_manager.sync_pages(run_state, preferred_page=page)
        run_state.current_url = str(getattr(page, "url", "") or target)
        self.session_manager.invalidate_snapshot(run_state, reason="navigate")

        title = ""
        try:
            title = str(await page.title() or "")
        except Exception:
            title = ""

        status_code = None
        response_url = None
        if response is not None:
            try:
                status_code = response.status
            except Exception:
                status_code = None
            try:
                response_url = response.url
            except Exception:
                response_url = None

        result = {
            "ok": True,
            "requested_url": target,
            "final_url": run_state.current_url,
            "response_url": response_url,
            "status_code": status_code,
            "title": title,
            "wait_until": wait_mode,
            "active_tab_id": run_state.active_page_id,
            "tabs": await self.session_manager.describe_tabs(run_state),
            "capture_delta": self._network_delta(run_state, cursor),
            **self._snapshot_followup("Verify the loaded page before any further action."),
        }
        self._log_action(run_state, action="navigate", payload=result)
        return result

    async def wait(
        self,
        run_state: RunState,
        *,
        timeout_ms: Optional[int] = None,
        load_state: str = "",
        text: str = "",
        url_contains: str = "",
    ) -> Dict[str, Any]:
        page = await self.ensure_page(run_state)
        resolved_timeout = int(timeout_ms or 0) or self._timeout_ms(run_state)
        wait_target = str(load_state or "").strip().lower()
        if wait_target:
            allowed = {"load", "domcontentloaded", "networkidle"}
            if wait_target not in allowed:
                raise ValueError(f"Unsupported load_state: {wait_target}")
            await page.wait_for_load_state(wait_target, timeout=resolved_timeout)

        if str(url_contains or "").strip():
            await page.wait_for_function(
                "(expected) => window.location.href.includes(expected)",
                arg=str(url_contains),
                timeout=resolved_timeout,
            )

        if str(text or "").strip():
            await page.get_by_text(str(text), exact=False).first.wait_for(
                state="visible",
                timeout=resolved_timeout,
            )

        self.session_manager.sync_pages(run_state, preferred_page=page)
        result = {
            "ok": True,
            "url": str(getattr(page, "url", "") or run_state.current_url or ""),
            "active_tab_id": run_state.active_page_id,
            "load_state": wait_target or None,
            "text": str(text or "").strip() or None,
            "url_contains": str(url_contains or "").strip() or None,
            **self._snapshot_followup("Confirm the awaited page state before continuing."),
        }
        self._log_action(run_state, action="wait", payload=result)
        return result

    async def list_tabs(self, run_state: RunState) -> Dict[str, Any]:
        tabs = await self.session_manager.describe_tabs(run_state)
        result = {"ok": True, "active_tab_id": run_state.active_page_id, "tabs": tabs}
        self._log_action(run_state, action="list_tabs", payload=result)
        return result

    async def select_tab(self, run_state: RunState, page_id: str) -> Dict[str, Any]:
        page = await self.session_manager.set_active_page(run_state, page_id)
        title = ""
        try:
            title = str(await page.title() or "")
        except Exception:
            title = ""
        result = {
            "ok": True,
            "active_tab_id": run_state.active_page_id,
            "url": str(getattr(page, "url", "") or run_state.current_url or ""),
            "title": title,
            "tabs": await self.session_manager.describe_tabs(run_state),
            **self._snapshot_followup("Inspect the selected tab before taking another action."),
        }
        self._log_action(run_state, action="select_tab", payload=result)
        return result

    async def close_tab(self, run_state: RunState, page_id: str) -> Dict[str, Any]:
        result = await self.session_manager.close_page(run_state, page_id)
        current = await self.describe_state(run_state)
        payload = {
            **result,
            "active_tab_id": current.get("active_tab_id"),
            "url": current.get("url"),
            "tabs": current.get("tabs"),
            **self._snapshot_followup("Inspect the remaining active tab after closing the previous tab."),
        }
        self._log_action(run_state, action="close_tab", payload=payload)
        return payload

    def _resolve_screenshot_path(self, run_state: RunState, path: Optional[str]) -> Path:
        if path:
            return Path(path).expanduser()
        root = Path((run_state.metadata or {}).get("screenshots_dir", "/tmp/playwright_agent"))
        ts = int(time.time() * 1000)
        return root / f"{run_state.run_id}_{ts}.png"

    async def screenshot(
        self,
        run_state: RunState,
        *,
        path: Optional[str] = None,
        full_page: bool = True,
    ) -> Dict[str, Any]:
        page = await self.ensure_page(run_state)
        screenshot_path = self._resolve_screenshot_path(run_state, path)
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        screenshot_bytes = await page.screenshot(path=str(screenshot_path), full_page=bool(full_page))
        encoded = base64.b64encode(bytes(screenshot_bytes or b"")).decode("ascii")
        title = ""
        try:
            title = str(await page.title() or "")
        except Exception:
            title = ""
        result = {
            "ok": True,
            "path": str(screenshot_path),
            "url": str(getattr(page, "url", "") or run_state.current_url or ""),
            "title": title,
            "full_page": bool(full_page),
            "active_tab_id": run_state.active_page_id,
            "mime_type": "image/png",
            "base64": encoded,
        }
        self._log_action(
            run_state,
            action="screenshot",
            payload={
                **result,
                "base64": "<omitted>",
                "base64_length": len(encoded),
            },
        )
        return result

    async def finalize_after_dom_mutation(
        self,
        run_state: RunState,
        *,
        previous_tab_ids: Sequence[str],
        action: str,
        capture_delta_cursor: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        page = await self.ensure_page(run_state)
        new_page_ids = self.session_manager.sync_pages(run_state, preferred_page=page)
        current_tabs = await self.session_manager.describe_tabs(run_state)

        opened_tab_ids = [
            page_id
            for page_id in new_page_ids
            if page_id not in set(str(item) for item in previous_tab_ids)
        ]
        if opened_tab_ids:
            await self.session_manager.set_active_page(run_state, opened_tab_ids[-1])
            page = await self.ensure_page(run_state)
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=self._timeout_ms(run_state))
            except Exception:
                pass

        run_state.current_url = str(getattr(page, "url", "") or run_state.current_url or "")
        self.session_manager.invalidate_snapshot(run_state, reason=action)
        result = {
            "url": run_state.current_url,
            "active_tab_id": run_state.active_page_id,
            "tabs": current_tabs,
            "opened_tab_ids": opened_tab_ids,
            "capture_delta": self._network_delta(run_state, capture_delta_cursor),
            **self._snapshot_followup(f"Verify the page state after browser_{action}."),
        }
        return result
