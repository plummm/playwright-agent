"""Page action feature scaffold."""

from __future__ import annotations

import asyncio
import base64
import inspect
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.tools import StructuredTool

from .event_logger import BrowserEventLogger
from .models import RunState
from .session import BrowserSessionManager


def mcp_tool(
    _func: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
    examples: Optional[List[str]] = None,
) -> Any:
    """Decorator to mark a method as an MCP-exposed tool."""

    def _decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        setattr(func, "_is_mcp_tool", True)
        setattr(func, "_mcp_name", name or func.__name__)
        setattr(func, "_mcp_examples", examples or [])
        return func

    if _func is None:
        return _decorate
    return _decorate(_func)


class PageActionsFeature:
    """Navigate and interact with pages."""

    def __init__(
        self,
        session_manager: BrowserSessionManager,
        event_logger: BrowserEventLogger,
        network_inspector: Optional[Any] = None,
    ):
        self.session_manager = session_manager
        self.event_logger = event_logger
        self.network_inspector = network_inspector
        self._run_state: Optional[RunState] = None
        self._tools: List[Any] = []

    def get_tools(self) -> List[Any]:
        """Export page-action tools for LLM tool calling."""
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

    def set_run_state(self, run_state: Optional[RunState]) -> None:
        """Bind the active run-state used by tool wrappers."""
        self._run_state = run_state

    def set_network_inspector(self, network_inspector: Optional[Any]) -> None:
        """Bind network inspector dependency for navigation delta snapshots."""
        self.network_inspector = network_inspector

    async def navigate(self, run_state: RunState, url: str, wait_until: str = "load") -> Dict[str, Any]:
        target = str(url or "").strip()
        if not target:
            raise ValueError("URL is required")

        page = await self._get_or_create_page(run_state)
        timeout_ms = self._timeout_ms(run_state)
        valid_wait_until = {"load", "domcontentloaded", "networkidle", "commit"}
        wait_mode = wait_until if wait_until in valid_wait_until else "load"

        capture_cursor = None
        if self.network_inspector is not None:
            try:
                get_cursor = getattr(self.network_inspector, "get_delta_cursor", None)
                if callable(get_cursor):
                    capture_cursor = get_cursor(run_state)
            except Exception:
                capture_cursor = None

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

        # Attach a post-navigation screenshot snapshot so downstream consumers can
        # reason over the current visual state without making another tool call.
        screenshot_payload = await self.screenshot(run_state, path=None, full_page=True)

        result = {
            "ok": True,
            "requested_url": target,
            "final_url": run_state.current_url,
            "response_url": response_url,
            "status_code": status_code,
            "title": title,
            "wait_until": wait_mode,
            "screenshot": screenshot_payload,
        }
        if self.network_inspector is not None:
            try:
                get_delta = getattr(self.network_inspector, "get_delta_since", None)
                if callable(get_delta):
                    result["capture_delta"] = get_delta(run_state, capture_cursor)
            except Exception:
                pass
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
        image_base64 = ""
        if isinstance(image, (bytes, bytearray)):
            image_base64 = base64.b64encode(bytes(image)).decode("ascii")
        result = {
            "ok": True,
            "image_base64": image_base64,
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

    async def move_cursor(
        self,
        run_state: RunState,
        selector: str,
        *,
        position: str = "center",
        steps: int = 20,
        timeout_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Move mouse cursor to a target element point.

        Coordinate strategy:
        - Use Playwright locator auto-wait/scroll behavior.
        - Resolve element viewport bounding box.
        - Compute recommended point from box (default center).
        """
        page = await self._get_or_create_page(run_state)
        target = str(selector or "").strip()
        if not target:
            raise ValueError("Selector is required")
        x, y = await self._resolve_target_point(
            run_state=run_state,
            page=page,
            selector=target,
            position=position,
            timeout_ms=timeout_ms,
        )
        move_steps = max(1, int(steps or 1))
        await page.mouse.move(x, y, steps=move_steps)
        result = {
            "ok": True,
            "selector": target,
            "position": str(position or "center"),
            "x": x,
            "y": y,
            "steps": move_steps,
            "url": page.url,
        }
        self._log_action(run_state, action="move_cursor", payload=result)
        return result

    async def mouse_click(
        self,
        run_state: RunState,
        selector: str,
        *,
        position: str = "center",
        button: str = "left",
        click_count: int = 1,
        delay_ms: int = 0,
        timeout_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Move cursor to target point and click via page.mouse.click."""
        page = await self._get_or_create_page(run_state)
        target = str(selector or "").strip()
        if not target:
            raise ValueError("Selector is required")
        x, y = await self._resolve_target_point(
            run_state=run_state,
            page=page,
            selector=target,
            position=position,
            timeout_ms=timeout_ms,
        )
        normalized_button = str(button or "left").strip().lower()
        if normalized_button not in {"left", "middle", "right"}:
            normalized_button = "left"
        clicks = max(1, int(click_count or 1))
        delay = max(0, int(delay_ms or 0))
        await page.mouse.click(
            x,
            y,
            button=normalized_button,
            click_count=clicks,
            delay=delay,
        )
        run_state.current_url = page.url or run_state.current_url
        result = {
            "ok": True,
            "selector": target,
            "position": str(position or "center"),
            "x": x,
            "y": y,
            "button": normalized_button,
            "click_count": clicks,
            "delay_ms": delay,
            "url": run_state.current_url,
        }
        self._log_action(run_state, action="mouse_click", payload=result)
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

    async def _resolve_target_point(
        self,
        *,
        run_state: RunState,
        page: Any,
        selector: str,
        position: str = "center",
        timeout_ms: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        Resolve deterministic element coordinates for mouse actions.

        Recommended Playwright approach:
        1) wait for element visibility
        2) scroll into view
        3) read bounding box
        4) compute point from box (center by default)
        """
        resolved_timeout = int(timeout_ms or 0) or self._timeout_ms(run_state)
        locator = page.locator(selector).first
        await locator.wait_for(state="visible", timeout=resolved_timeout)
        await locator.scroll_into_view_if_needed(timeout=resolved_timeout)
        box = await locator.bounding_box()
        if not box:
            raise RuntimeError(f"Could not resolve bounding box for selector: {selector}")
        x = float(box["x"])
        y = float(box["y"])
        w = float(box["width"])
        h = float(box["height"])
        pos = str(position or "center").strip().lower()
        if pos == "top_left":
            return x + 1.0, y + 1.0
        if pos == "top_right":
            return x + max(1.0, w - 1.0), y + 1.0
        if pos == "bottom_left":
            return x + 1.0, y + max(1.0, h - 1.0)
        if pos == "bottom_right":
            return x + max(1.0, w - 1.0), y + max(1.0, h - 1.0)
        return x + (w / 2.0), y + (h / 2.0)

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

    def _require_run_state(self) -> RunState:
        """Return currently bound run-state for tool wrappers."""
        if self._run_state is None:
            raise RuntimeError("PageActionsFeature run_state is not set")
        return self._run_state

    def _make_sync_adapter(self, async_method: Callable[..., Any]) -> Callable[..., Any]:
        """Create sync adapter for an async MCP tool method."""
        def _adapter(**kwargs: Any) -> Any:
            return self._run_async(async_method(**kwargs))

        return _adapter

    @mcp_tool(
        name="browser_navigate",
        examples=[
            "browser_navigate(url='https://example.com')",
            "browser_navigate(url='https://news.ycombinator.com', wait_until='domcontentloaded')",
        ],
    )
    async def mcp_browser_navigate(self, url: str, wait_until: str = "load") -> Dict[str, Any]:
        """
        Navigate the active page to a target URL and capture a post-navigation screenshot.

        Args:
            url: Absolute URL to open in the current page.
            wait_until (optional): Page readiness condition.
                Allowed values: [`load`, `domcontentloaded`, `networkidle`, `commit`].
                Any other value falls back to `load`.

        Returns:
            Dict with navigation metadata:
            - ok (bool): Whether navigation completed.
            - requested_url (str): URL requested by the caller.
            - final_url (str): Final URL after redirects/navigation.
            - response_url (str | null): URL from the navigation response object.
            - status_code (int | null): HTTP status when available.
            - title (str): Current page title after navigation.
            - wait_until (str): Effective wait mode used.
            - screenshot (dict): Result from `browser_screenshot`.
            - capture_delta (dict, optional): Newly captured network/console records
              since this navigate call started (when network inspector is enabled).

        Examples:
            await mcp_browser_navigate(url="https://example.com", wait_until="load")
            await mcp_browser_navigate(url="https://news.ycombinator.com", wait_until="domcontentloaded")
        """
        return await self.navigate(self._require_run_state(), url=url, wait_until=wait_until)

    @mcp_tool(
        name="browser_screenshot",
        examples=[
            "browser_screenshot()",
            "browser_screenshot(path='/tmp/playwright_agent/shot.png', full_page=False)",
        ],
    )
    async def mcp_browser_screenshot(self, path: str = "", full_page: bool = True) -> Dict[str, Any]:
        """
        Capture a screenshot of the active browser page.

        Args:
            path (optional): Filesystem path to save screenshot. If empty/omitted,
                a default run-scoped path is used.
            full_page (optional): Whether to capture full scrollable page.
                Allowed values: [`true`, `false`]. Default: `true`.

        Returns:
            Dict with screenshot payload:
            - ok (bool): Whether screenshot capture succeeded.
            - image_base64 (str): PNG bytes encoded as base64.
            - full_page (bool): Whether full-page mode was used.
            - url (str): Current page URL at capture time.

        Examples:
            await mcp_browser_screenshot(full_page=True)
            await mcp_browser_screenshot(path="/tmp/playwright_agent/shot.png", full_page=False)
        """
        normalized_path = path.strip() or None
        return await self.screenshot(
            self._require_run_state(),
            path=normalized_path,
            full_page=bool(full_page),
        )

    @mcp_tool(
        name="browser_click",
        examples=[
            "browser_click(selector='button[type=submit]')",
            "browser_click(selector='text=Sign in', timeout_ms=10000)",
        ],
    )
    async def mcp_browser_click(self, selector: str, timeout_ms: Optional[int] = None) -> Dict[str, Any]:
        """
        Click a page element identified by selector.

        Args:
            selector: Playwright selector expression for the target element.
            timeout_ms (optional): Click timeout in milliseconds.
                If omitted, run default timeout is used.

        Returns:
            Dict with click execution metadata:
            - ok (bool): Whether click succeeded.
            - selector (str): Selector used for the click.
            - timeout_ms (int): Effective timeout used.
            - url (str): Current page URL after the click.

        Examples:
            await mcp_browser_click(selector="#login-button", timeout_ms=5000)
            await mcp_browser_click(selector="text=Sign in")
        """
        return await self.click(
            self._require_run_state(),
            selector=selector,
            timeout_ms=timeout_ms,
        )

    @mcp_tool(
        name="browser_type_text",
        examples=[
            "browser_type_text(selector='input[name=q]', text='playwright docs')",
            "browser_type_text(selector='#email', text='user@example.com', delay_ms=40)",
        ],
    )
    async def mcp_browser_type_text(
        self,
        selector: str,
        text: str,
        delay_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Type text into a form field or editable element.

        Args:
            selector: Playwright selector expression for the input target.
            text: Text value to enter.
            delay_ms (optional): Per-keystroke delay in milliseconds.
                If `delay_ms <= 0` or omitted, the tool uses fast deterministic `fill()`.
                If `delay_ms > 0`, the tool uses `type()` with the specified delay.

        Returns:
            Dict with typing metadata:
            - ok (bool): Whether typing succeeded.
            - selector (str): Selector used for input.
            - chars (int): Number of characters sent.
            - delay_ms (int): Effective key delay used (0 when using fill mode).
            - url (str): Current page URL after typing.

        Examples:
            await mcp_browser_type_text(selector='input[name=search]', text='moose framework')
            await mcp_browser_type_text(selector='#email', text='user@example.com', delay_ms=40)
        """
        return await self.type_text(
            self._require_run_state(),
            selector=selector,
            text=text,
            delay_ms=delay_ms,
        )

    @mcp_tool(
        name="browser_press_key",
        examples=[
            "browser_press_key(key='Enter')",
            "browser_press_key(key='Escape', selector='input[name=q]')",
        ],
    )
    async def mcp_browser_press_key(self, key: str, selector: Optional[str] = None) -> Dict[str, Any]:
        """
        Press a keyboard key globally or on a selected element.

        Args:
            key: Playwright key name (for example `Enter`, `Escape`, `Tab`).
            selector (optional): Selector to target a specific element before key press.
                If omitted, key is pressed at page keyboard level.

        Returns:
            Dict with key-press metadata:
            - ok (bool): Whether key press succeeded.
            - key (str): Key that was pressed.
            - selector (str | null): Selector targeted (null when global keyboard press).
            - url (str): Current page URL after the key press.

        Examples:
            await mcp_browser_press_key(key="Enter")
            await mcp_browser_press_key(key="Escape", selector="input[name=q]")
        """
        return await self.press_key(
            self._require_run_state(),
            key=key,
            selector=selector,
        )

    @mcp_tool(
        name="browser_move_cursor",
        examples=[
            "browser_move_cursor(selector='button[type=submit]')",
            "browser_move_cursor(selector='#chart', position='top_left', steps=30)",
        ],
    )
    async def mcp_browser_move_cursor(
        self,
        selector: str,
        position: str = "center",
        steps: int = 20,
        timeout_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Move cursor to an element using coordinates derived from element bounding box.

        Args:
            selector: Playwright selector for the target element.
            position (optional): Target point on the element.
                Allowed values: [`center`, `top_left`, `top_right`, `bottom_left`, `bottom_right`].
                Any other value falls back to `center`.
            steps (optional): Interpolation steps for mouse movement animation.
                Must be an integer >= 1. Default: `20`.
            timeout_ms (optional): Timeout for locating/scrolling target element in ms.
                If omitted, run default timeout is used.

        Returns:
            Dict with mouse-move metadata:
            - ok (bool): Whether move succeeded.
            - selector (str): Selector used.
            - position (str): Position mode used to compute x/y.
            - x (float): Final x coordinate used.
            - y (float): Final y coordinate used.
            - steps (int): Effective movement steps.
            - url (str): Current page URL.

        Examples:
            await mcp_browser_move_cursor(selector="button[type=submit]")
            await mcp_browser_move_cursor(selector="#chart", position="top_left", steps=30)
        """
        return await self.move_cursor(
            self._require_run_state(),
            selector=selector,
            position=position,
            steps=steps,
            timeout_ms=timeout_ms,
        )

    @mcp_tool(
        name="browser_mouse_click",
        examples=[
            "browser_mouse_click(selector='button[type=submit]')",
            "browser_mouse_click(selector='#canvas', position='top_left', button='left', click_count=1)",
        ],
    )
    async def mcp_browser_mouse_click(
        self,
        selector: str,
        position: str = "center",
        button: str = "left",
        click_count: int = 1,
        delay_ms: int = 0,
        timeout_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Click an element using explicit mouse coordinates (computed from element geometry).

        Args:
            selector: Playwright selector for the target element.
            position (optional): Target point on the element.
                Allowed values: [`center`, `top_left`, `top_right`, `bottom_left`, `bottom_right`].
                Any other value falls back to `center`.
            button (optional): Mouse button.
                Allowed values: [`left`, `middle`, `right`].
                Any other value falls back to `left`.
            click_count (optional): Number of clicks to issue. Must be integer >= 1.
                Default: `1`.
            delay_ms (optional): Delay between mouse down/up in milliseconds.
                Must be integer >= 0. Default: `0`.
            timeout_ms (optional): Timeout for locating/scrolling target element in ms.
                If omitted, run default timeout is used.

        Returns:
            Dict with mouse-click metadata:
            - ok (bool): Whether click succeeded.
            - selector (str): Selector used.
            - position (str): Position mode used to compute x/y.
            - x (float): Click x coordinate.
            - y (float): Click y coordinate.
            - button (str): Effective button used.
            - click_count (int): Effective click count.
            - delay_ms (int): Effective click delay.
            - url (str): Current page URL after click.

        Examples:
            await mcp_browser_mouse_click(selector="button[type=submit]")
            await mcp_browser_mouse_click(selector="#canvas", position="top_left", button="left", click_count=1)
        """
        return await self.mouse_click(
            self._require_run_state(),
            selector=selector,
            position=position,
            button=button,
            click_count=click_count,
            delay_ms=delay_ms,
            timeout_ms=timeout_ms,
        )

    @staticmethod
    def _run_async(coro: Any) -> Any:
        """Run async tool coroutine from sync tool wrapper."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import threading

                holder: Dict[str, Any] = {}

                def _runner() -> None:
                    try:
                        holder["result"] = asyncio.run(coro)
                    except Exception as e:
                        holder["error"] = e

                t = threading.Thread(target=_runner, daemon=True)
                t.start()
                t.join()
                if "error" in holder:
                    raise holder["error"]
                return holder.get("result")
            return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)
