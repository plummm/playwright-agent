"""Page action feature scaffold."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .event_logger import BrowserEventLogger
from .models import RunState
from .session import BrowserSessionManager


class PageActionsFeature:
    """Navigate and interact with pages."""

    def __init__(
        self,
        session_manager: BrowserSessionManager,
        event_logger: BrowserEventLogger,
    ):
        self.session_manager = session_manager
        self.event_logger = event_logger

    def get_tools(self) -> List[Any]:
        # Tools are wired by the browser automation class once run-state tool wrappers are finalized.
        return []

    async def navigate(self, run_state: RunState, url: str, wait_until: str = "load") -> Dict[str, Any]:
        target = str(url or "").strip()
        if not target:
            raise ValueError("URL is required")

        page = await self._get_or_create_page(run_state)
        timeout_ms = self._timeout_ms(run_state)
        valid_wait_until = {"load", "domcontentloaded", "networkidle", "commit"}
        wait_mode = wait_until if wait_until in valid_wait_until else "load"

        response = await page.goto(target, wait_until=wait_mode, timeout=timeout_ms)
        run_state.current_url = page.url or target

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

        try:
            title = await page.title()
        except Exception:
            title = ""

        result = {
            "ok": True,
            "requested_url": target,
            "final_url": run_state.current_url,
            "response_url": response_url,
            "status_code": status_code,
            "title": title,
            "wait_until": wait_mode,
        }
        self._log_action(
            run_state,
            action="navigate",
            payload=result,
        )
        return result

    async def screenshot(
        self,
        run_state: RunState,
        path: Optional[str] = None,
        full_page: bool = True,
    ) -> Dict[str, Any]:
        page = await self._get_or_create_page(run_state)
        screenshot_path = self._resolve_screenshot_path(run_state, path)
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)

        image = await page.screenshot(path=str(screenshot_path), full_page=bool(full_page))
        result = {
            "ok": True,
            "path": str(screenshot_path),
            "bytes": len(image) if isinstance(image, (bytes, bytearray)) else 0,
            "full_page": bool(full_page),
            "url": page.url,
        }
        self._log_action(run_state, action="screenshot", payload=result)
        return result

    async def click(
        self,
        run_state: RunState,
        selector: str,
        timeout_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        page = await self._get_or_create_page(run_state)
        target = str(selector or "").strip()
        if not target:
            raise ValueError("Selector is required")

        resolved_timeout = int(timeout_ms or self._timeout_ms(run_state))
        locator = page.locator(target).first
        await locator.click(timeout=resolved_timeout)
        run_state.current_url = page.url or run_state.current_url

        result = {
            "ok": True,
            "selector": target,
            "timeout_ms": resolved_timeout,
            "url": run_state.current_url,
        }
        self._log_action(run_state, action="click", payload=result)
        return result

    async def type_text(
        self,
        run_state: RunState,
        selector: str,
        text: str,
        delay_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        page = await self._get_or_create_page(run_state)
        target = str(selector or "").strip()
        if not target:
            raise ValueError("Selector is required")

        value = str(text or "")
        locator = page.locator(target).first
        timeout_ms = self._timeout_ms(run_state)

        # Prefer fill() for deterministic replacement; use type() only when delay is requested.
        if delay_ms is None or int(delay_ms) <= 0:
            await locator.fill(value, timeout=timeout_ms)
        else:
            await locator.fill("", timeout=timeout_ms)
            await locator.type(value, timeout=timeout_ms, delay=int(delay_ms))

        result = {
            "ok": True,
            "selector": target,
            "chars": len(value),
            "delay_ms": int(delay_ms) if delay_ms else 0,
            "url": page.url,
        }
        self._log_action(run_state, action="type_text", payload=result)
        return result

    async def press_key(
        self,
        run_state: RunState,
        key: str,
        selector: Optional[str] = None,
    ) -> Dict[str, Any]:
        page = await self._get_or_create_page(run_state)
        press_key = str(key or "").strip()
        if not press_key:
            raise ValueError("Key is required")

        timeout_ms = self._timeout_ms(run_state)
        target = str(selector or "").strip()
        if target:
            await page.locator(target).first.press(press_key, timeout=timeout_ms)
        else:
            await page.keyboard.press(press_key)

        result = {
            "ok": True,
            "key": press_key,
            "selector": target or None,
            "url": page.url,
        }
        self._log_action(run_state, action="press_key", payload=result)
        return result

    def _timeout_ms(self, run_state: RunState) -> int:
        """Read effective timeout from run-state metadata."""
        try:
            return int((run_state.metadata or {}).get("timeout_ms", 30000))
        except Exception:
            return 30000

    async def _get_or_create_page(self, run_state: RunState):
        """Return an active page, creating one if needed."""
        page = self.session_manager.get_active_page(run_state)
        if page is None and getattr(run_state, "browser_context", None) is not None:
            page = await self.session_manager.new_page(run_state)
        if page is None:
            raise RuntimeError("No active browser page. Start a browser session first.")
        return page

    def _resolve_screenshot_path(self, run_state: RunState, path: Optional[str]) -> Path:
        """Resolve screenshot output path and ensure deterministic defaults."""
        if path:
            return Path(path).expanduser()

        root = Path((run_state.metadata or {}).get("screenshots_dir", "/tmp/playwright_agent"))
        ts = int(time.time() * 1000)
        return root / f"{run_state.run_id}_{ts}.png"

    def _log_action(self, run_state: RunState, *, action: str, payload: Dict[str, Any]) -> None:
        """Best-effort action event logging."""
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
