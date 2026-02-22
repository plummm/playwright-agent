"""PlaywrightAgent scaffold."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import inspect
import json
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool

from moose.framework.agent_core.prompt_loader import (
    load_prompt_text,
    load_system_prompt,
    render_prompt_template,
)
from moose.framework import BaseAgent
from moose.framework.llm_core import LLMClient

from .browser.downloads import DownloadsFeature
from .browser.event_logger import BrowserEventLogger
from .browser.models import RunConfig, RunState
from .browser.network_inspector import NetworkInspectorFeature
from .browser.page_actions import PageActionsFeature
from .browser.session import BrowserSessionManager


class PlaywrightAgent(BaseAgent):
    """Agent scaffold for browser automation with Playwright."""

    name = "playwright_agent"
    description = "Browser automation agent using Playwright"

    def __init__(self, config_path: Optional[str] = None, debug: bool = False):
        """Initialize the agent and feature placeholders."""
        super().__init__(config_path, debug=debug)
        custom = self.config.get("custom", {}) if isinstance(self.config.get("custom"), dict) else {}
        llm_cfg = custom.get("llm", {}) if isinstance(custom.get("llm"), dict) else {}
        url_cfg = custom.get("url_extractor", {}) if isinstance(custom.get("url_extractor"), dict) else {}

        self.agent_system_prompt = (
            load_system_prompt(
                system_prompt_path=str(llm_cfg.get("system_prompt_path") or ""),
                skills_dir=str(llm_cfg.get("skills_dir") or ""),
                logger=self.logger,
                label="playwright_agent.custom.llm.system_prompt_path",
                required=True,
            )
        )
        self.url_extractor_system_prompt = (
            load_prompt_text(
                path=str(url_cfg.get("system_prompt_path") or ""),
                logger=self.logger,
                label="playwright_agent.custom.url_extractor.system_prompt_path",
                required=True,
            )
        )
        self.url_extractor_user_prompt_template = (
            load_prompt_text(
                path=str(url_cfg.get("user_prompt_path") or ""),
                logger=self.logger,
                label="playwright_agent.custom.url_extractor.user_prompt_path",
                required=True,
            )
        )

        self.url_extractor_llm: Optional[LLMClient] = None
        self.llm_cfg: Dict[str, Any] = llm_cfg
        self.url_extractor_cfg: Dict[str, Any] = url_cfg
        self.browser_automation: Any = None
        self.session_manager: Optional[BrowserSessionManager] = None
        self.event_logger: Optional[BrowserEventLogger] = None
        self.run_state: Optional[RunState] = None
        self.page_actions: Optional[PageActionsFeature] = None
        self.network_inspector: Optional[NetworkInspectorFeature] = None
        self.downloads: Optional[DownloadsFeature] = None
        self.url_extractor_tool: Any = None

    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Process a browser automation request.

        Input expects `user_prompt` and optional `system_prompt`.
        This method builds run configuration, delegates execution to the
        browser automation class, and returns a standardized wrapper.
        """
        try:
            payload: Dict[str, Any]
            if isinstance(input_data, dict):
                payload = input_data.get("input", input_data) if isinstance(input_data.get("input"), dict) else input_data
            else:
                payload = {"user_prompt": str(input_data)}

            user_prompt = str(payload.get("user_prompt") or payload.get("prompt") or "").strip()
            if not user_prompt:
                return {
                    "status": "error",
                    "result": {},
                    "error": "Missing required field: user_prompt",
                }

            normalized_payload = dict(payload)
            normalized_payload["user_prompt"] = user_prompt
            base_run_config = self._build_run_config(normalized_payload)
            self._ensure_runtime(base_run_config)

            extraction = self._extract_actionable_urls(user_prompt)
            actionable_urls = [
                str(u).strip()
                for u in extraction.get("actionable_urls", [])
                if str(u).strip()
            ]
            if not actionable_urls and base_run_config.start_url:
                actionable_urls = [str(base_run_config.start_url)]
            actionable_urls = list(dict.fromkeys(actionable_urls))

            if not actionable_urls:
                clarification_question = extraction.get("clarification_question")
                if clarification_question:
                    return {
                        "status": "error",
                        "result": {},
                        "error": str(clarification_question),
                    }
                return {
                    "status": "error",
                    "result": {},
                    "error": "No actionable URLs identified from user prompt",
                }

            browser_tools = self._build_browser_action_tools()
            merged_system_prompt = self._compose_system_prompt(
                str(normalized_payload.get("system_prompt") or "").strip() or None
            )
            result_by_url: Dict[str, Any] = {}
            for target_url in actionable_urls:
                try:
                    url_result = self._run_url_workflow_sync(
                        target_url=target_url,
                        user_prompt=user_prompt,
                        merged_system_prompt=merged_system_prompt,
                        base_run_config=base_run_config,
                        browser_tools=browser_tools,
                    )
                    result_by_url[target_url] = url_result
                except Exception as url_error:
                    result_by_url[target_url] = {
                        "status": "error",
                        "error": f"{type(url_error).__name__}: {url_error}",
                    }

            return {"status": "success", "result": result_by_url}
        except Exception as e:
            self.logger.error("PlaywrightAgent process error: %s", e)
            return {"status": "error", "result": {}, "error": str(e)}

    async def analyze(self, data: Any) -> Dict[str, Any]:
        """Async compatibility wrapper that delegates to `process`."""
        return self.process(data)

    def health(self, _data: Dict[str, Any]) -> Dict[str, Any]:
        """Return health and runtime wiring status."""
        return {
            "status": "ok",
            "agent": self.name,
            "automation_ready": self.browser_automation is not None,
        }

    async def mcp_extract_actionable_urls(self, user_prompt: str) -> str:
        """
        MCP tool: classify user-mentioned URLs by execution intent.

        Args:
            user_prompt: Raw user prompt from the automation request.

        Returns:
            JSON string with actionable/reference/excluded URL groups.
        """
        llm = self._get_or_create_url_extractor_llm()
        tool_prompt = render_prompt_template(
            self.url_extractor_user_prompt_template,
            replacements={"user_prompt": str(user_prompt or "").strip()},
        )
        response = await llm.send_message(
            message=tool_prompt,
            system_message=self.url_extractor_system_prompt,
        )
        return self._normalize_url_extractor_output(str(response.content or ""))

    def _build_tools(self) -> List[Any]:
        """Collect tools exposed by configured feature modules."""
        tools: List[Any] = []
        extractor_tool = self._get_url_extractor_tool()
        if extractor_tool is not None:
            tools.append(extractor_tool)

        tools.extend(self._build_browser_action_tools())
        return tools

    def _build_browser_action_tools(self) -> List[Any]:
        """Collect browser action tools exposed by feature modules."""
        tools: List[Any] = []
        for feature in (self.page_actions, self.network_inspector, self.downloads):
            if feature is None:
                continue
            get_tools = getattr(feature, "get_tools", None)
            if callable(get_tools):
                feature_tools = get_tools()
                if isinstance(feature_tools, list):
                    tools.extend(feature_tools)
        return tools

    def _compose_system_prompt(self, user_system_prompt: Optional[str]) -> str:
        """Merge built-in guardrail prompt with user-provided system prompt."""
        parts = [self.agent_system_prompt]
        extra = (user_system_prompt or "").strip()
        if extra:
            parts.append(extra)
        return "\n\n".join(parts)

    def _build_run_config(self, payload: Dict[str, Any]) -> RunConfig:
        """Normalize request payload into immutable run configuration."""
        request_id = str(payload.get("request_id") or uuid.uuid4())
        run_id = str(payload.get("run_id") or uuid.uuid4())
        user_prompt = str(payload.get("user_prompt") or "").strip()
        user_system_prompt = str(payload.get("system_prompt") or "").strip()
        merged_system_prompt = self._compose_system_prompt(user_system_prompt or None)

        return RunConfig(
            run_id=run_id,
            request_id=request_id,
            start_url=(str(payload.get("start_url")).strip() or None) if payload.get("start_url") else None,
            system_prompt=merged_system_prompt,
            user_prompt=user_prompt,
            headless=bool(payload.get("headless", True)),
            extension_path=(str(payload.get("extension_path")).strip() or None)
            if payload.get("extension_path")
            else None,
            timeout_ms=int(payload.get("timeout_ms", 30000)),
            capture_network=bool(payload.get("capture_network", True)),
            capture_console=bool(payload.get("capture_console", True)),
            capture_response_bodies=bool(payload.get("capture_response_bodies", True)),
            accept_downloads=bool(payload.get("accept_downloads", True)),
        )

    def _ensure_runtime(self, run_config: RunConfig) -> None:
        """
        Validate and initialize the browser automation runtime.

        If the automation class exposes a `start` method, it is invoked once per run.
        """
        if self.browser_automation is None:
            raise NotImplementedError(
                "Browser automation class is not wired yet. "
                "Set `self.browser_automation` to an implementation."
            )

        start_method = getattr(self.browser_automation, "start", None)
        if callable(start_method):
            start_result = start_method(run_config=run_config, tools=self._build_browser_action_tools())
            if inspect.isawaitable(start_result):
                self._run_async_blocking(start_result)

    def _get_url_extractor_tool(self) -> Any:
        """Build or return cached StructuredTool for URL extraction."""
        if self.url_extractor_tool is not None:
            return self.url_extractor_tool

        self.url_extractor_tool = StructuredTool.from_function(
            func=self._mcp_extract_actionable_urls_sync,
            coroutine=self.mcp_extract_actionable_urls,
            name="mcp_extract_actionable_urls",
            description=(
                "Extract and classify URLs from the user prompt into actionable browsing URLs "
                "versus reference-only URLs."
            ),
        )
        return self.url_extractor_tool

    def _get_or_create_url_extractor_llm(self) -> LLMClient:
        """Return the shared secondary LLM client used by URL extraction tool calls."""
        if self.url_extractor_llm is not None:
            return self.url_extractor_llm

        llm_cfg = self.llm_cfg
        extractor_cfg = self.url_extractor_cfg

        model = str(extractor_cfg.get("model") or llm_cfg.get("model") or "gpt-4o-mini")
        temperature = float(extractor_cfg.get("temperature", 0.0))
        max_output_tokens = extractor_cfg.get("max_output_tokens", 600)
        enable_web_search = bool(extractor_cfg.get("enable_web_search", False))

        kwargs = extractor_cfg.get("kwargs", {})
        if not isinstance(kwargs, dict):
            kwargs = {}

        self.url_extractor_llm = LLMClient(
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            enable_web_search=enable_web_search,
            **kwargs,
        )
        return self.url_extractor_llm

    def _mcp_extract_actionable_urls_sync(self, user_prompt: str) -> str:
        """Sync wrapper for MCP URL extraction tool execution."""
        coro = self.mcp_extract_actionable_urls(user_prompt=user_prompt)
        return self._run_async_blocking(coro)

    def _extract_actionable_urls(self, user_prompt: str) -> Dict[str, Any]:
        """Run URL extraction tool and return normalized parsed dict."""
        raw = self._mcp_extract_actionable_urls_sync(user_prompt=user_prompt)
        parsed = self._extract_json_obj(raw)
        return {
            "actionable_urls": parsed.get("actionable_urls", []) if isinstance(parsed, dict) else [],
            "reference_urls": parsed.get("reference_urls", []) if isinstance(parsed, dict) else [],
            "excluded_urls": parsed.get("excluded_urls", []) if isinstance(parsed, dict) else [],
            "rationale_by_url": parsed.get("rationale_by_url", {}) if isinstance(parsed, dict) else {},
            "needs_clarification": bool(parsed.get("needs_clarification", False)) if isinstance(parsed, dict) else False,
            "clarification_question": parsed.get("clarification_question") if isinstance(parsed, dict) else None,
        }

    def _run_url_workflow_sync(
        self,
        *,
        target_url: str,
        user_prompt: str,
        merged_system_prompt: str,
        base_run_config: RunConfig,
        browser_tools: List[Any],
    ) -> Dict[str, Any]:
        """Run the per-URL LangGraph workflow in sync mode."""
        coro = self._run_url_workflow(
            target_url=target_url,
            user_prompt=user_prompt,
            merged_system_prompt=merged_system_prompt,
            base_run_config=base_run_config,
            browser_tools=browser_tools,
        )
        out = self._run_async_blocking(coro)
        return out if isinstance(out, dict) else {"output": out}

    async def _run_url_workflow(
        self,
        *,
        target_url: str,
        user_prompt: str,
        merged_system_prompt: str,
        base_run_config: RunConfig,
        browser_tools: List[Any],
    ) -> Dict[str, Any]:
        """Run URL workflow graph: task refinement node then browser automation node."""

        async def _identify_user_task(state: Dict[str, Any]) -> Dict[str, Any]:
            refined_prompt = await self._refine_user_task_for_url(
                user_prompt=state["user_prompt"],
                target_url=state["target_url"],
            )
            return {"refined_user_prompt": refined_prompt}

        async def _browser_automation_node(state: Dict[str, Any]) -> Dict[str, Any]:
            worker_system_prompt = self._compose_browser_worker_system_prompt(
                base_system_prompt=merged_system_prompt,
                browser_tools=browser_tools,
                target_url=state["target_url"],
            )
            browser_result = await self._invoke_browser_automation_for_url(
                target_url=state["target_url"],
                refined_user_prompt=state.get("refined_user_prompt") or state["user_prompt"],
                worker_system_prompt=worker_system_prompt,
                base_run_config=base_run_config,
                browser_tools=browser_tools,
            )
            return {"browser_result": browser_result}

        initial_state = {
            "target_url": target_url,
            "user_prompt": user_prompt,
            "refined_user_prompt": "",
            "browser_result": {},
        }

        try:
            from langgraph.graph import END, StateGraph
        except Exception:
            first = await _identify_user_task(initial_state)
            second_input = dict(initial_state)
            second_input.update(first)
            second = await _browser_automation_node(second_input)
            return second.get("browser_result", {})

        graph = StateGraph(dict)
        graph.add_node("identify_user_task", _identify_user_task)
        graph.add_node("browser_automation", _browser_automation_node)
        graph.set_entry_point("identify_user_task")
        graph.add_edge("identify_user_task", "browser_automation")
        graph.add_edge("browser_automation", END)
        app = graph.compile()
        output = await app.ainvoke(initial_state)
        if isinstance(output, dict):
            return output.get("browser_result", {}) if isinstance(output.get("browser_result"), dict) else output
        return {"output": output}

    async def _refine_user_task_for_url(self, *, user_prompt: str, target_url: str) -> str:
        """Use a dedicated LLM call to extract a URL-specific browser objective."""
        llm = self._create_task_refiner_llm_client()
        system_message = (
            "You convert a user request into a concise URL-specific browser automation objective.\n"
            "Given a target URL and original user request, output a clear objective prompt that tells\n"
            "a browser automation model exactly what to verify on that target URL.\n"
            "Focus only on actions and checks relevant to the target URL.\n"
            "Return plain text only."
        )
        user_message = (
            f"Target URL:\n{target_url}\n\n"
            f"Original user request:\n{user_prompt}\n\n"
            "Rewrite as one actionable objective prompt for browser automation on this URL."
        )
        response = await llm.send_message(message=user_message, system_message=system_message)
        objective = str(response.content or "").strip()
        return objective or user_prompt

    def _create_task_refiner_llm_client(self) -> LLMClient:
        """Create a fresh LLM client for the task-refinement node."""
        custom = self.config.get("custom", {}) if isinstance(self.config.get("custom"), dict) else {}
        llm_cfg = custom.get("llm", {}) if isinstance(custom.get("llm"), dict) else {}
        refiner_cfg = custom.get("task_refiner", {}) if isinstance(custom.get("task_refiner"), dict) else {}

        model = str(refiner_cfg.get("model") or llm_cfg.get("model") or "gpt-4o-mini")
        temperature = float(refiner_cfg.get("temperature", 0.1))
        max_output_tokens = refiner_cfg.get("max_output_tokens", 512)
        enable_web_search = bool(refiner_cfg.get("enable_web_search", False))
        kwargs = refiner_cfg.get("kwargs", {})
        if not isinstance(kwargs, dict):
            kwargs = {}
        return LLMClient(
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            enable_web_search=enable_web_search,
            **kwargs,
        )

    async def _invoke_browser_automation_for_url(
        self,
        *,
        target_url: str,
        refined_user_prompt: str,
        worker_system_prompt: str,
        base_run_config: RunConfig,
        browser_tools: List[Any],
    ) -> Dict[str, Any]:
        """Invoke browser automation implementation for one URL."""
        run_method = getattr(self.browser_automation, "run", None)
        if not callable(run_method):
            raise NotImplementedError("Browser automation class must implement a `run` method")

        per_url_run_config = replace(
            base_run_config,
            run_id=str(uuid.uuid4()),
            start_url=target_url,
            user_prompt=refined_user_prompt,
            system_prompt=worker_system_prompt,
        )

        kwargs = self._filter_call_kwargs(
            run_method,
            {
                "url": target_url,
                "target_url": target_url,
                "run_config": per_url_run_config,
                "user_prompt": refined_user_prompt,
                "system_prompt": worker_system_prompt,
                "tools": browser_tools,
            },
        )
        out = None
        last_error: Optional[Exception] = None
        if kwargs:
            try:
                out = run_method(**kwargs)
            except TypeError as e:
                last_error = e
                out = None
        if out is None:
            try:
                out = run_method(target_url)
            except TypeError as e:
                last_error = e
                out = run_method()
        if inspect.isawaitable(out):
            out = await out
        if out is None and last_error is not None:
            raise last_error
        return out if isinstance(out, dict) else {"output": out}

    def _compose_browser_worker_system_prompt(
        self,
        *,
        base_system_prompt: str,
        browser_tools: List[Any],
        target_url: str,
    ) -> str:
        """Compose per-URL worker system prompt with explicit browser tool capabilities."""
        capabilities: List[str] = []
        for tool in browser_tools:
            name = str(getattr(tool, "name", "") or "").strip() or tool.__class__.__name__
            desc = str(getattr(tool, "description", "") or "").strip()
            if desc:
                capabilities.append(f"- {name}: {desc}")
            else:
                capabilities.append(f"- {name}")
        capabilities_text = "\n".join(capabilities) if capabilities else "- No browser tools registered"

        worker_prompt = (
            "You are executing browser automation for one target URL.\n"
            f"Target URL: {target_url}\n\n"
            "Capabilities available as MCP tools:\n"
            f"{capabilities_text}\n\n"
            "Execution policy:\n"
            "1. Use the available browser tools to complete the objective.\n"
            "2. Prefer deterministic sequences (navigate, inspect, interact, verify).\n"
            "3. Keep actions scoped to the target URL task.\n"
            "4. Ground conclusions in tool-observed evidence.\n"
        )
        return f"{base_system_prompt}\n\n{worker_prompt}"

    @staticmethod
    def _filter_call_kwargs(func: Any, candidate_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Filter kwargs by function signature while preserving **kwargs compatibility."""
        try:
            sig = inspect.signature(func)
        except Exception:
            return {}
        params = sig.parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return dict(candidate_kwargs)
        return {k: v for k, v in candidate_kwargs.items() if k in params}

    @staticmethod
    def _run_async_blocking(coro: Any) -> Any:
        """Run awaitable from sync context, handling existing event-loop cases."""
        if not inspect.isawaitable(coro):
            return coro
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is not None and running_loop.is_running():
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

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return asyncio.run_coroutine_threadsafe(coro, loop).result()
            return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    def _normalize_url_extractor_output(self, raw_content: str) -> str:
        """Normalize LLM output into the URL extractor JSON schema."""
        parsed = self._extract_json_obj(raw_content)
        result: Dict[str, Any] = {
            "actionable_urls": parsed.get("actionable_urls", []),
            "reference_urls": parsed.get("reference_urls", []),
            "excluded_urls": parsed.get("excluded_urls", []),
            "rationale_by_url": parsed.get("rationale_by_url", {}),
            "needs_clarification": bool(parsed.get("needs_clarification", False)),
            "clarification_question": parsed.get("clarification_question"),
        }
        result["actionable_urls"] = [
            str(v).strip() for v in result["actionable_urls"] if str(v).strip()
        ]
        result["reference_urls"] = [
            str(v).strip() for v in result["reference_urls"] if str(v).strip()
        ]
        result["excluded_urls"] = [
            str(v).strip() for v in result["excluded_urls"] if str(v).strip()
        ]

        clean_rationale: Dict[str, str] = {}
        for key, val in (result["rationale_by_url"] or {}).items():
            k = str(key).strip()
            if k:
                clean_rationale[k] = str(val).strip()
        result["rationale_by_url"] = clean_rationale

        question = result.get("clarification_question")
        result["clarification_question"] = str(question).strip() if question else None
        if not result["needs_clarification"]:
            result["clarification_question"] = None

        return json.dumps(result, ensure_ascii=False)

    @staticmethod
    def _extract_json_obj(raw_content: str) -> Dict[str, Any]:
        """Best-effort extraction of a JSON object from model output."""
        text = str(raw_content or "").strip()
        if not text:
            return {}

        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass

        if "```" in text:
            for part in text.split("```"):
                candidate = part.strip()
                if candidate.lower().startswith("json"):
                    candidate = candidate[4:].strip()
                if not candidate:
                    continue
                try:
                    obj = json.loads(candidate)
                    return obj if isinstance(obj, dict) else {}
                except Exception:
                    continue

        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(text[start : end + 1])
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}
        return {}
