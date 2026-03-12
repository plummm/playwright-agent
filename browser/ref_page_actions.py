"""Ref-based page actions for the Playwright browser agent."""

from __future__ import annotations

import base64
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from moose.framework.llm_core import LLMClient, Message, MessageRole, create_llm_client_from_config
from moose.framework.llm_core.tool_runtime import ToolRuntime

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


def _llm_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
                continue
            if isinstance(item, dict):
                text_value = item.get("text")
                if text_value is not None and str(text_value).strip():
                    parts.append(str(text_value).strip())
        return "\n".join(parts).strip()
    if isinstance(content, dict):
        text_value = content.get("text")
        if text_value is not None:
            return str(text_value).strip()
    return str(content or "").strip()


class PageActionsFeature:
    """Navigate and interact with pages using semantic refs."""

    def __init__(
        self,
        session_manager: BrowserSessionManager,
        event_logger: BrowserEventLogger,
        network_inspector: Optional[Any] = None,
        llm_settings: Optional[Dict[str, Any]] = None,
        screenshot_analyzer_llm_settings: Optional[Dict[str, Any]] = None,
        screenshot_analyzer_system_prompt: str = "",
    ):
        self.session_manager = session_manager
        self.event_logger = event_logger
        self.network_inspector = network_inspector
        self.llm_settings = dict(llm_settings or {})
        self.screenshot_analyzer_llm_settings = dict(screenshot_analyzer_llm_settings or {})
        self.screenshot_analyzer_system_prompt = str(screenshot_analyzer_system_prompt or "").strip()
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

    def set_llm_settings(self, llm_settings: Optional[Dict[str, Any]]) -> None:
        self.llm_settings = dict(llm_settings or {})

    def _require_run_state(self) -> RunState:
        if self._run_state is None:
            raise RuntimeError("PageActionsFeature run_state is not set")
        return self._run_state

    def _create_screenshot_llm_client(self, run_state: RunState) -> LLMClient:
        cfg = dict(self.screenshot_analyzer_llm_settings or {})
        cfg_kwargs = dict(cfg.get("kwargs") or {}) if isinstance(cfg.get("kwargs"), dict) else {}
        runtime_overrides: Dict[str, Any] = {
            "enable_web_search": False,
            "use_responses_api": True,
        }
        if "timeout" not in cfg and "timeout" not in cfg_kwargs:
            try:
                timeout_ms = int((run_state.metadata or {}).get("timeout_ms", 0) or 0)
            except Exception:
                timeout_ms = 0
            if timeout_ms > 0:
                runtime_overrides["timeout"] = timeout_ms / 1000.0
        return create_llm_client_from_config(
            cfg,
            agent_name="playwright_agent",
            runtime_overrides=runtime_overrides,
        )

    async def analyze_screenshot(
        self,
        run_state: RunState,
        instruction: str,
        *,
        full_page: bool = True,
    ) -> str:
        prompt = str(instruction or "").strip()
        if not prompt:
            return "Screenshot analysis instruction is required."

        capture = await self.controller.screenshot(run_state, full_page=full_page)
        screenshot_path = str(capture.get("path") or "").strip()
        if not screenshot_path:
            return "I could not capture a screenshot to analyze."

        image_bytes = Path(screenshot_path).read_bytes()
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_uri = f"data:image/png;base64,{b64}"

        llm = self._create_screenshot_llm_client(run_state)
        response = await llm.send_message(
            Message(
                role=MessageRole.USER,
                content=[
                    {
                        "type": "input_text",
                        "text": (
                            f"Instruction: {prompt}\n"
                            f"Page URL: {capture.get('url') or ''}\n"
                            f"Page title: {capture.get('title') or ''}\n\n"
                            "Answer with only the requested conclusion in natural language."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": data_uri,
                    },
                ],
            ),
            system_message=self.screenshot_analyzer_system_prompt or None,
        )
        runtime = ToolRuntime.current()
        if runtime is not None:
            try:
                runtime.add_external_llm_usage(
                    usage=getattr(response, "usage", None),
                    cost=getattr(response, "cost", None),
                )
            except Exception:
                pass
        return _llm_content_to_text(getattr(response, "content", "")).strip() or (
            "I could not determine a reliable conclusion from the screenshot."
        )

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
        """
        Navigate the active browser tab to a URL and return the resulting page metadata.

        Use case
        - You want to open a new page, follow a direct link, or reload the active tab to a known destination before inspecting it.

        Parameters
        - url: Target URL to open in the active tab.
        - wait_until: Navigation readiness mode. Common values: `load`, `domcontentloaded`, `networkidle`, `commit`.

        Return value
        - Dict with navigation result details:
          - ok: Whether navigation succeeded.
          - requested_url: URL requested by the tool.
          - final_url: Final page URL after redirects.
          - response_url: URL associated with the Playwright response, when available.
          - status_code: Top-level HTTP status code, when available.
          - title: Page title after navigation.
          - wait_until: Applied wait mode.
          - active_tab_id: Active tab after navigation.
          - tabs: Current tab list.
          - capture_delta: Newly captured network/console activity counts since navigation started.
          - snapshot_stale / suggested_next_tool / suggested_next_tool_reason: Guidance to call `browser_snapshot` next.

        Concrete example (code)

        ```python
        browser_navigate(url="https://example.com", wait_until="domcontentloaded")
        ```
        """
        return await self.navigate(self._require_run_state(), url=url, wait_until=wait_until)

    @mcp_tool(name="browser_snapshot", examples=["browser_snapshot()", "browser_snapshot(max_refs=40)"])
    async def mcp_browser_snapshot(
        self,
        max_refs: int = 80,
        include_page_text: bool = True,
    ) -> Dict[str, Any]:
        """
        Capture a fresh semantic snapshot of the active page and expose interactive refs.

        Use case
        - You need a compact, token-efficient representation of the current page before choosing the next browser action.

        Parameters
        - max_refs: Maximum number of interactive refs to include in the snapshot.
        - include_page_text: Whether to include short page text snippets in the rendered snapshot.

        Return value
        - Dict with snapshot data:
          - ok: Whether capture succeeded.
          - snapshot_id: Snapshot identifier for the captured page state.
          - active_tab_id: Tab the snapshot belongs to.
          - url: Current page URL.
          - title: Current page title.
          - ref_count: Number of refs included.
          - snapshot_text: Rendered semantic snapshot text for the model.
          - refs: Map from ref id (for example `r3`) to public metadata such as description, role, name, tag, href, and page id.

        Concrete example (code)

        ```python
        browser_snapshot(max_refs=40, include_page_text=True)
        ```
        """
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
        """
        Click a semantic ref from the latest snapshot.

        Use case
        - You have already identified an interactive ref with `browser_snapshot` and want to activate that element.

        Parameters
        - ref: Snapshot ref id to click (for example `r3`).
        - button: Mouse button to use: `left`, `middle`, or `right`.
        - click_count: Number of clicks to perform.
        - delay_ms: Delay in milliseconds between mouse down and mouse up.

        Return value
        - Dict describing the action result:
          - ok: Whether the click succeeded.
          - ref: Clicked ref id.
          - description: Human-readable ref description from the snapshot.
          - button: Applied mouse button.
          - click_count: Applied click count.
          - active_tab_id / url / title / tabs: Updated browser state.
          - capture_delta: Newly captured network/console activity after the click, when available.
          - snapshot_stale / suggested_next_tool / suggested_next_tool_reason: Guidance to refresh with `browser_snapshot`.

        Concrete example (code)

        ```python
        browser_click(ref="r3", button="left", click_count=1)
        ```
        """
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
        """
        Focus a semantic ref, type text into it, and optionally submit with Enter.

        Use case
        - You want to fill a search box, text field, or form control identified by `browser_snapshot`.

        Parameters
        - ref: Snapshot ref id to focus and type into.
        - text: Text content to enter.
        - submit: Whether to press Enter after typing.
        - delay_ms: Delay in milliseconds between typed characters.
        - clear_first: Whether to clear the field before typing.

        Return value
        - Dict describing the typing result:
          - ok: Whether typing succeeded.
          - ref: Target ref id.
          - description: Human-readable ref description.
          - chars: Number of characters typed.
          - submit: Whether Enter was pressed after typing.
          - active_tab_id / url / title / tabs: Updated browser state.
          - capture_delta: Newly captured network/console activity after typing, when available.
          - snapshot_stale / suggested_next_tool / suggested_next_tool_reason: Guidance to refresh with `browser_snapshot`.

        Concrete example (code)

        ```python
        browser_type(ref="r2", text="hello world", submit=True, clear_first=True)
        ```
        """
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
        """
        Press a keyboard key on the current page, optionally after focusing a ref.

        Use case
        - You need to trigger keyboard-driven interactions such as Enter, Escape, Tab, or arrow keys.

        Parameters
        - key: Keyboard key name to press (for example `Enter`, `Escape`, `Tab`, `ArrowDown`).
        - ref: Optional snapshot ref id to focus before pressing the key.

        Return value
        - Dict describing the key press result:
          - ok: Whether the key press succeeded.
          - key: Key that was pressed.
          - ref: Focused ref id, if one was provided.
          - active_tab_id / url / title / tabs: Updated browser state.
          - capture_delta: Newly captured network/console activity after the key press, when available.
          - snapshot_stale / suggested_next_tool / suggested_next_tool_reason: Guidance to refresh with `browser_snapshot`.

        Concrete example (code)

        ```python
        browser_press(key="Enter", ref="r5")
        ```
        """
        return await self.press_key(self._require_run_state(), key=key, ref=ref)

    @mcp_tool(name="browser_scroll", examples=["browser_scroll(delta_y=900)", "browser_scroll(ref='r12')"])
    async def mcp_browser_scroll(self, ref: str = "", delta_y: int = 800) -> Dict[str, Any]:
        """
        Scroll the page viewport or bring a specific semantic ref into view.

        Use case
        - You want to reveal more content, move down the page, or center a known element before inspecting it again.

        Parameters
        - ref: Optional snapshot ref id to scroll into view. If provided, `delta_y` is ignored.
        - delta_y: Vertical wheel delta to apply when scrolling the viewport directly.

        Return value
        - Dict describing the scroll result:
          - ok: Whether scrolling succeeded.
          - ref: Ref scrolled into view, if one was provided.
          - delta_y: Applied viewport wheel delta.
          - snapshot_stale / suggested_next_tool / suggested_next_tool_reason: Guidance to refresh with `browser_snapshot`.

        Concrete example (code)

        ```python
        browser_scroll(delta_y=900)
        browser_scroll(ref="r12")
        ```
        """
        return await self.scroll(self._require_run_state(), ref=ref, delta_y=delta_y)

    @mcp_tool(name="browser_wait", examples=["browser_wait(load_state='networkidle', timeout_ms=8000)"])
    async def mcp_browser_wait(
        self,
        timeout_ms: int = 15000,
        load_state: str = "",
        text: str = "",
        url_contains: str = "",
    ) -> Dict[str, Any]:
        """
        Wait for page readiness, visible text, or a URL condition.

        Use case
        - You have already triggered navigation or DOM changes and want to pause until the expected page state is visible.

        Parameters
        - timeout_ms: Maximum time to wait in milliseconds.
        - load_state: Optional Playwright load state to wait for: `load`, `domcontentloaded`, or `networkidle`.
        - text: Optional visible text substring that must appear on the page.
        - url_contains: Optional substring that must appear in the current URL.

        Return value
        - Dict describing the satisfied wait condition:
          - ok: Whether the wait succeeded.
          - url: Current page URL after waiting.
          - active_tab_id: Active tab after waiting.
          - load_state: Applied load-state condition, if any.
          - text: Applied text condition, if any.
          - url_contains: Applied URL condition, if any.
          - snapshot_stale / suggested_next_tool / suggested_next_tool_reason: Guidance to refresh with `browser_snapshot`.

        Concrete example (code)

        ```python
        browser_wait(load_state="networkidle", timeout_ms=8000)
        browser_wait(text="Sign in", timeout_ms=5000)
        ```
        """
        return await self.wait(
            self._require_run_state(),
            timeout_ms=timeout_ms,
            load_state=load_state,
            text=text,
            url_contains=url_contains,
        )

    @mcp_tool(
        name="browser_screenshot",
        examples=[
            "browser_screenshot(instruction='Is the login button visible?')",
            "browser_screenshot(instruction='Summarize the chart trend in one sentence.', full_page=False)",
        ],
    )
    async def mcp_browser_screenshot(self, instruction: str, full_page: bool = True) -> str:
        """
        Capture a screenshot, analyze it with a separate LLM call, and return only the requested conclusion.

        Use case
        - Whenever you need a visual check on the webpage

        Parameters
        - instruction: What the screenshot-analysis LLM should determine from the image, please be concise and comprehensive, including objective and expected return.
        - full_page: Whether to capture the full scrollable page (`True`) or only the viewport (`False`).

        Return value
        - Natural-language conclusion string produced from the screenshot.
        - The screenshot file path is managed internally and not exposed as a tool argument.

        Concrete example (code)

        ```python
        browser_screenshot(instruction="Is there an error banner on the page?")
        browser_screenshot(instruction="Does the webpage look like a malicious website, such phishing, techscam, malvertising, unwanted download, etc.?", full_page=False)
        ```
        """
        return await self.analyze_screenshot(
            self._require_run_state(),
            instruction=instruction,
            full_page=full_page,
        )

    @mcp_tool(name="browser_list_tabs", examples=["browser_list_tabs()"])
    async def mcp_browser_list_tabs(self) -> Dict[str, Any]:
        """
        List the currently open browser tabs and identify the active tab.

        Use case
        - You want to inspect multi-tab state before switching, closing, or deciding where the next action should happen.

        Parameters
        - This tool takes no arguments.

        Return value
        - Dict with tab state:
          - ok: Whether listing succeeded.
          - active_tab_id: Active tab id.
          - tabs: Array of tab records with ids, URLs, titles, and active-state flags.

        Concrete example (code)

        ```python
        browser_list_tabs()
        ```
        """
        return await self.controller.list_tabs(self._require_run_state())

    @mcp_tool(name="browser_select_tab", examples=["browser_select_tab(page_id='tab_2')"])
    async def mcp_browser_select_tab(self, page_id: str) -> Dict[str, Any]:
        """
        Switch the active browser tab.

        Use case
        - You know which tab should become active and want subsequent browser actions to operate there.

        Parameters
        - page_id: Target tab id from `browser_list_tabs`.

        Return value
        - Dict describing the selected tab:
          - ok: Whether the tab switch succeeded.
          - active_tab_id: New active tab id.
          - url: URL of the selected tab.
          - title: Title of the selected tab.
          - tabs: Current tab list after switching.
          - snapshot_stale / suggested_next_tool / suggested_next_tool_reason: Guidance to refresh with `browser_snapshot`.

        Concrete example (code)

        ```python
        browser_select_tab(page_id="tab_2")
        ```
        """
        return await self.controller.select_tab(self._require_run_state(), page_id=page_id)

    @mcp_tool(name="browser_close_tab", examples=["browser_close_tab(page_id='tab_2')"])
    async def mcp_browser_close_tab(self, page_id: str) -> Dict[str, Any]:
        """
        Close one browser tab and return the remaining active tab state.

        Use case
        - You want to dismiss an extra tab or popup tab and continue work in the remaining browsing context.

        Parameters
        - page_id: Tab id to close.

        Return value
        - Dict describing the updated tab state:
          - ok: Whether the close action succeeded.
          - closed_tab_id: Closed tab id, when available from the session manager.
          - active_tab_id: Active tab after closing.
          - url: URL of the remaining active tab.
          - tabs: Current tab list after the close.
          - snapshot_stale / suggested_next_tool / suggested_next_tool_reason: Guidance to refresh with `browser_snapshot`.

        Concrete example (code)

        ```python
        browser_close_tab(page_id="tab_2")
        ```
        """
        return await self.controller.close_tab(self._require_run_state(), page_id=page_id)

    @mcp_tool(name="browser_new_tab", examples=["browser_new_tab()", "browser_new_tab(url='https://example.com')"])
    async def mcp_browser_new_tab(self, url: str = "") -> Dict[str, Any]:
        """
        Create a new browser tab and optionally navigate it immediately.

        Use case
        - You want to keep the current page open while exploring another URL in parallel.

        Parameters
        - url: Optional URL to open in the new tab right after creation.

        Return value
        - Dict describing the new tab:
          - ok: Whether the tab was created successfully.
          - active_tab_id: New active tab id.
          - url: URL currently loaded in the new tab.
          - title: Title of the new tab, when available.
          - tabs: Current tab list after creation.
          - snapshot_stale / suggested_next_tool / suggested_next_tool_reason: Guidance to inspect the new tab with `browser_snapshot`.

        Concrete example (code)

        ```python
        browser_new_tab()
        browser_new_tab(url="https://example.com")
        ```
        """
        return await self.new_tab(self._require_run_state(), url=url)
