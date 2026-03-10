"""Ref-based page actions for the Playwright browser agent."""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool

from .controller import BrowserController
from .event_logger import BrowserEventLogger
from .models import RunState
from .semantic_snapshot import SemanticSnapshotManager
from .session import BrowserSessionManager
from .tooling import mcp_tool

_FOCUS_AND_CENTER_JS = """
function() {
  const el = this;
  el.scrollIntoView({ block: 'center', inline: 'center' });
  if (typeof el.focus === 'function') {
    el.focus();
  }
  const rect = el.getBoundingClientRect();
  return {
    x: rect.x + (rect.width / 2),
    y: rect.y + (rect.height / 2),
    tag_name: (el.tagName || '').toLowerCase(),
    href: typeof el.href === 'string' ? el.href : null,
    content_editable: !!el.isContentEditable
  };
}
""".strip()

_CLEAR_FIELD_JS = """
function() {
  const el = this;
  el.scrollIntoView({ block: 'center', inline: 'center' });
  if (typeof el.focus === 'function') {
    el.focus();
  }
  if ('value' in el) {
    el.value = '';
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  } else if (el.isContentEditable) {
    el.textContent = '';
    if (typeof InputEvent === 'function') {
      el.dispatchEvent(new InputEvent('input', { bubbles: true, data: '' }));
    }
  }
  const rect = el.getBoundingClientRect();
  return {
    x: rect.x + (rect.width / 2),
    y: rect.y + (rect.height / 2),
    tag_name: (el.tagName || '').toLowerCase(),
    content_editable: !!el.isContentEditable
  };
}
""".strip()


class PageActionsFeature:
    """Navigate and interact with pages using semantic refs."""

    def __init__(
        self,
        session_manager: BrowserSessionManager,
        event_logger: BrowserEventLogger,
        network_inspector: Optional[Any] = None,
    ):
        self.session_manager = session_manager
        self.event_logger = event_logger
        self.network_inspector = network_inspector
        self.controller = BrowserController(
            session_manager=session_manager,
            event_logger=event_logger,
            network_inspector=network_inspector,
        )
        self.snapshot_manager = SemanticSnapshotManager(
            session_manager=session_manager,
            event_logger=event_logger,
        )
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
        self._run_state = run_state
        self.controller.set_run_state(run_state)
        self.snapshot_manager.set_run_state(run_state)

    def set_network_inspector(self, network_inspector: Optional[Any]) -> None:
        self.network_inspector = network_inspector
        self.controller.set_network_inspector(network_inspector)

    def _require_run_state(self) -> RunState:
        if self._run_state is None:
            raise RuntimeError("PageActionsFeature run_state is not set")
        return self._run_state

    def _network_cursor(self, run_state: RunState) -> Optional[Dict[str, int]]:
        inspector = self.network_inspector
        get_cursor = getattr(inspector, "get_delta_cursor", None) if inspector is not None else None
        if not callable(get_cursor):
            return None
        try:
            return get_cursor(run_state)
        except Exception:
            return None

    async def navigate(self, run_state: RunState, url: str, wait_until: str = "load") -> Dict[str, Any]:
        return await self.controller.navigate(run_state, url=url, wait_until=wait_until)

    async def capture_snapshot(
        self,
        run_state: RunState,
        *,
        max_refs: int = 80,
        include_page_text: bool = True,
    ) -> Dict[str, Any]:
        return await self.snapshot_manager.capture(
            run_state,
            max_refs=max(1, int(max_refs or 80)),
            include_page_text=bool(include_page_text),
        )

    async def click_ref(
        self,
        run_state: RunState,
        *,
        ref: str,
        button: str = "left",
        click_count: int = 1,
        delay_ms: int = 0,
    ) -> Dict[str, Any]:
        before_tabs = await self.session_manager.describe_tabs(run_state)
        delta_cursor = self._network_cursor(run_state)
        current, _session, semantic_ref, _object_id = await self.snapshot_manager.resolve_ref_object(ref, run_state)
        page = await self.controller.ensure_page(current)
        target = await self.snapshot_manager.call_function_on_ref(
            ref,
            function_declaration=_FOCUS_AND_CENTER_JS,
            run_state=current,
        )
        if not isinstance(target, dict):
            raise RuntimeError(f"Failed to compute click target for ref: {ref}")

        button_name = str(button or "left").strip().lower()
        if button_name not in {"left", "middle", "right"}:
            button_name = "left"

        await page.mouse.click(
            float(target.get("x") or 0.0),
            float(target.get("y") or 0.0),
            button=button_name,
            click_count=max(1, int(click_count or 1)),
            delay=max(0, int(delay_ms or 0)),
        )
        post = await self.controller.finalize_after_dom_mutation(
            current,
            previous_tab_ids=[tab["page_id"] for tab in before_tabs],
            action="click",
            capture_delta_cursor=delta_cursor,
        )
        return {
            "ok": True,
            "ref": semantic_ref.ref,
            "description": semantic_ref.description,
            "button": button_name,
            "click_count": max(1, int(click_count or 1)),
            **post,
        }

    async def type_into_ref(
        self,
        run_state: RunState,
        *,
        ref: str,
        text: str,
        submit: bool = False,
        delay_ms: int = 0,
        clear_first: bool = True,
    ) -> Dict[str, Any]:
        before_tabs = await self.session_manager.describe_tabs(run_state)
        delta_cursor = self._network_cursor(run_state)
        current, _session, semantic_ref, _object_id = await self.snapshot_manager.resolve_ref_object(ref, run_state)
        page = await self.controller.ensure_page(current)
        function_declaration = _CLEAR_FIELD_JS if clear_first else _FOCUS_AND_CENTER_JS
        target = await self.snapshot_manager.call_function_on_ref(
            ref,
            function_declaration=function_declaration,
            run_state=current,
        )
        if isinstance(target, dict) and target.get("x") is not None and target.get("y") is not None:
            await page.mouse.click(float(target["x"]), float(target["y"]))

        await page.keyboard.type(str(text or ""), delay=max(0, int(delay_ms or 0)))
        if submit:
            await page.keyboard.press("Enter")

        post = await self.controller.finalize_after_dom_mutation(
            current,
            previous_tab_ids=[tab["page_id"] for tab in before_tabs],
            action="type",
            capture_delta_cursor=delta_cursor,
        )
        return {
            "ok": True,
            "ref": semantic_ref.ref,
            "description": semantic_ref.description,
            "chars": len(str(text or "")),
            "submit": bool(submit),
            **post,
        }

    async def press_key(
        self,
        run_state: RunState,
        *,
        key: str,
        ref: str = "",
    ) -> Dict[str, Any]:
        before_tabs = await self.session_manager.describe_tabs(run_state)
        delta_cursor = self._network_cursor(run_state)
        page = await self.controller.ensure_page(run_state)
        if str(ref or "").strip():
            await self.snapshot_manager.call_function_on_ref(
                ref,
                function_declaration=_FOCUS_AND_CENTER_JS,
                run_state=run_state,
            )
        press_key = str(key or "").strip()
        if not press_key:
            raise ValueError("key is required")
        await page.keyboard.press(press_key)
        post = await self.controller.finalize_after_dom_mutation(
            run_state,
            previous_tab_ids=[tab["page_id"] for tab in before_tabs],
            action="press_key",
            capture_delta_cursor=delta_cursor,
        )
        return {
            "ok": True,
            "key": press_key,
            "ref": str(ref or "").strip() or None,
            **post,
        }

    async def scroll(
        self,
        run_state: RunState,
        *,
        ref: str = "",
        delta_y: int = 800,
    ) -> Dict[str, Any]:
        page = await self.controller.ensure_page(run_state)
        if str(ref or "").strip():
            await self.snapshot_manager.call_function_on_ref(
                ref,
                function_declaration="""
function() {
  this.scrollIntoView({ block: 'center', inline: 'center' });
  return true;
}
""".strip(),
                run_state=run_state,
            )
        else:
            await page.mouse.wheel(0, int(delta_y or 0))
        self.session_manager.invalidate_snapshot(run_state, reason="scroll")
        return {
            "ok": True,
            "ref": str(ref or "").strip() or None,
            "delta_y": int(delta_y or 0),
            **self.controller._snapshot_followup("Inspect the updated viewport before continuing."),
        }

    async def wait(
        self,
        run_state: RunState,
        *,
        timeout_ms: Optional[int] = None,
        load_state: str = "",
        text: str = "",
        url_contains: str = "",
    ) -> Dict[str, Any]:
        return await self.controller.wait(
            run_state,
            timeout_ms=timeout_ms,
            load_state=load_state,
            text=text,
            url_contains=url_contains,
        )

    async def new_tab(self, run_state: RunState, *, url: str = "") -> Dict[str, Any]:
        page = await self.session_manager.new_page(run_state, make_active=True)
        if str(url or "").strip():
            await self.controller.navigate(run_state, url=str(url), wait_until="domcontentloaded")
        title = ""
        try:
            title = str(await page.title() or "")
        except Exception:
            title = ""
        return {
            "ok": True,
            "active_tab_id": run_state.active_page_id,
            "url": str(getattr(page, "url", "") or ""),
            "title": title,
            "tabs": await self.session_manager.describe_tabs(run_state),
            **self.controller._snapshot_followup("Inspect the new tab before continuing."),
        }

    @mcp_tool(name="browser_navigate", examples=["browser_navigate(url='https://example.com')"])
    async def mcp_browser_navigate(self, url: str, wait_until: str = "load") -> Dict[str, Any]:
        """Navigate the active tab to a URL, then call browser_snapshot to verify the new page state."""
        return await self.navigate(self._require_run_state(), url=url, wait_until=wait_until)

    @mcp_tool(name="browser_snapshot", examples=["browser_snapshot()", "browser_snapshot(max_refs=40)"])
    async def mcp_browser_snapshot(
        self,
        max_refs: int = 80,
        include_page_text: bool = True,
    ) -> Dict[str, Any]:
        """Capture a fresh semantic snapshot and return ref-based page context."""
        return await self.capture_snapshot(
            self._require_run_state(),
            max_refs=max_refs,
            include_page_text=include_page_text,
        )

    @mcp_tool(name="browser_click", examples=["browser_click(ref='r3')"])
    async def mcp_browser_click(
        self,
        ref: str,
        button: str = "left",
        click_count: int = 1,
        delay_ms: int = 0,
    ) -> Dict[str, Any]:
        """Click a semantic ref from the latest browser snapshot, then call browser_snapshot to verify the result."""
        return await self.click_ref(
            self._require_run_state(),
            ref=ref,
            button=button,
            click_count=click_count,
            delay_ms=delay_ms,
        )

    @mcp_tool(name="browser_type", examples=["browser_type(ref='r2', text='hello world')"])
    async def mcp_browser_type(
        self,
        ref: str,
        text: str,
        submit: bool = False,
        delay_ms: int = 0,
        clear_first: bool = True,
    ) -> Dict[str, Any]:
        """Focus a semantic ref, type text, and optionally submit with Enter, then call browser_snapshot to verify the page state."""
        return await self.type_into_ref(
            self._require_run_state(),
            ref=ref,
            text=text,
            submit=submit,
            delay_ms=delay_ms,
            clear_first=clear_first,
        )

    @mcp_tool(name="browser_press", examples=["browser_press(key='Enter')"])
    async def mcp_browser_press(self, key: str, ref: str = "") -> Dict[str, Any]:
        """Press a keyboard key globally or after focusing a semantic ref, then call browser_snapshot to verify the outcome."""
        return await self.press_key(self._require_run_state(), key=key, ref=ref)

    @mcp_tool(name="browser_scroll", examples=["browser_scroll(delta_y=900)", "browser_scroll(ref='r12')"])
    async def mcp_browser_scroll(self, ref: str = "", delta_y: int = 800) -> Dict[str, Any]:
        """Scroll the viewport or bring a semantic ref into view, then call browser_snapshot to inspect the updated viewport."""
        return await self.scroll(self._require_run_state(), ref=ref, delta_y=delta_y)

    @mcp_tool(name="browser_wait", examples=["browser_wait(load_state='networkidle', timeout_ms=8000)"])
    async def mcp_browser_wait(
        self,
        timeout_ms: int = 15000,
        load_state: str = "",
        text: str = "",
        url_contains: str = "",
    ) -> Dict[str, Any]:
        """Wait for page readiness, visible text, or a URL change, then call browser_snapshot to verify the awaited state."""
        return await self.wait(
            self._require_run_state(),
            timeout_ms=timeout_ms,
            load_state=load_state,
            text=text,
            url_contains=url_contains,
        )

    @mcp_tool(name="browser_screenshot", examples=["browser_screenshot()", "browser_screenshot(full_page=False)"])
    async def mcp_browser_screenshot(self, path: str = "", full_page: bool = True) -> Dict[str, Any]:
        """Capture a browser screenshot and return the saved path."""
        return await self.controller.screenshot(
            self._require_run_state(),
            path=path or None,
            full_page=full_page,
        )

    @mcp_tool(name="browser_list_tabs", examples=["browser_list_tabs()"])
    async def mcp_browser_list_tabs(self) -> Dict[str, Any]:
        """List open browser tabs and identify the active one."""
        return await self.controller.list_tabs(self._require_run_state())

    @mcp_tool(name="browser_select_tab", examples=["browser_select_tab(page_id='tab_2')"])
    async def mcp_browser_select_tab(self, page_id: str) -> Dict[str, Any]:
        """Switch the active tab, then call browser_snapshot to inspect the selected tab."""
        return await self.controller.select_tab(self._require_run_state(), page_id=page_id)

    @mcp_tool(name="browser_close_tab", examples=["browser_close_tab(page_id='tab_2')"])
    async def mcp_browser_close_tab(self, page_id: str) -> Dict[str, Any]:
        """Close a tab and return the new active tab state, then call browser_snapshot to verify what remains."""
        return await self.controller.close_tab(self._require_run_state(), page_id=page_id)

    @mcp_tool(name="browser_new_tab", examples=["browser_new_tab()", "browser_new_tab(url='https://example.com')"])
    async def mcp_browser_new_tab(self, url: str = "") -> Dict[str, Any]:
        """Create a new tab and optionally navigate it to a URL, then call browser_snapshot to inspect it."""
        return await self.new_tab(self._require_run_state(), url=url)
