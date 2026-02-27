"""PlaywrightAgent scaffold."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import uuid
from typing import Any, Dict, List, Optional

from moose.framework.agent_core.prompt_loader import (
    load_prompt_text,
    load_system_prompt,
    render_prompt_template,
)
from moose.framework import BaseAgent
from moose.framework.llm_core import LLMClient

from browser.automation import BrowserAutomation
from browser.downloads import DownloadsFeature
from browser.event_logger import BrowserEventLogger
from browser.models import RunConfig, RunState
from browser.network_inspector import NetworkInspectorFeature
from browser.page_actions import PageActionsFeature
from browser.session import BrowserSessionManager


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
        task_refiner_cfg = custom.get("task_refiner", {}) if isinstance(custom.get("task_refiner"), dict) else {}

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
        self.task_refiner_system_prompt = (
            load_prompt_text(
                path=str(task_refiner_cfg.get("system_prompt_path") or ""),
                logger=self.logger,
                label="playwright_agent.custom.task_refiner.system_prompt_path",
                required=True,
            )
        )
        self.task_refiner_user_prompt_template = (
            load_prompt_text(
                path=str(task_refiner_cfg.get("user_prompt_path") or ""),
                logger=self.logger,
                label="playwright_agent.custom.task_refiner.user_prompt_path",
                required=True,
            )
        )

        self.url_extractor_llm: Optional[LLMClient] = None
        self.llm_cfg: Dict[str, Any] = llm_cfg
        self.url_extractor_cfg: Dict[str, Any] = url_cfg
        self.task_refiner_cfg: Dict[str, Any] = task_refiner_cfg
        runtime_cfg = custom.get("browser_runtime", {}) if isinstance(custom.get("browser_runtime"), dict) else {}
        downloads_dir = str(runtime_cfg.get("downloads_dir") or "/tmp/playwright_agent")
        activity_db_path = str(runtime_cfg.get("activity_db_path") or "/tmp/playwright_agent/activity.db")

        self.session_manager: Optional[BrowserSessionManager] = BrowserSessionManager(downloads_dir=downloads_dir)
        self.event_logger: Optional[BrowserEventLogger] = BrowserEventLogger(db_path=Path(activity_db_path))
        self.run_state: Optional[RunState] = None
        self.page_actions: Optional[PageActionsFeature] = PageActionsFeature(
            session_manager=self.session_manager,
            event_logger=self.event_logger,
        )
        self.network_inspector: Optional[NetworkInspectorFeature] = NetworkInspectorFeature(
            session_manager=self.session_manager,
            event_logger=self.event_logger,
        )
        if self.page_actions is not None:
            try:
                self.page_actions.set_network_inspector(self.network_inspector)
            except Exception:
                pass
        self.downloads: Optional[DownloadsFeature] = DownloadsFeature(
            session_manager=self.session_manager,
            event_logger=self.event_logger,
            base_tmp_dir=downloads_dir,
        )
        self.browser_automation: Any = BrowserAutomation(
            session_manager=self.session_manager,
            page_actions=self.page_actions,
            network_inspector=self.network_inspector,
            downloads=self.downloads,
            llm_settings=self.llm_cfg,
            logger=self.logger,
        )

    async def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Primary async processing endpoint for browser automation.

        Input must be a direct JSON payload (dict) with required keys:
        - user_prompt
        - system_prompt
        """
        if not isinstance(input_data, dict):
            raise TypeError("Payload must be a JSON object with direct keys")

        payload: Dict[str, Any] = dict(input_data)
        user_prompt = str(payload.get("user_prompt") or "").strip()
        if not user_prompt:
            raise ValueError("Missing required field: user_prompt")

        user_system_prompt = str(payload.get("system_prompt") or "").strip()
        if not user_system_prompt:
            raise ValueError("Missing required field: system_prompt")

        normalized_payload = dict(payload)
        normalized_payload["user_prompt"] = user_prompt
        normalized_payload["system_prompt"] = user_system_prompt
        base_run_config = self._build_run_config(normalized_payload)
        self._ensure_automation_ready()

        extraction = await self._extract_actionable_urls(user_prompt)
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
                raise ValueError(str(clarification_question))
            raise ValueError("No actionable URLs identified from user prompt")

        browser_tools = self._build_browser_action_tools()
        merged_system_prompt = self._compose_system_prompt(user_system_prompt)
        result_by_url: Dict[str, Any] = {}
        for target_url in actionable_urls:
            try:
                url_result = await self._run_url_workflow(
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

    async def browse_url(self, input_data: Any) -> Dict[str, Any]:
        """Compatibility endpoint alias that delegates to `process`."""
        return await self.process(input_data)

    async def analyze(self, data: Any) -> Dict[str, Any]:
        """Async compatibility wrapper that delegates to `browse_url`."""
        return await self.process(data)

    def health(self, _data: Dict[str, Any]) -> Dict[str, Any]:
        """Return health and runtime wiring status."""
        return {
            "status": "ok",
            "agent": self.name,
            "automation_ready": self.browser_automation is not None,
        }

    def _build_tools(self) -> List[Any]:
        """Collect tools exposed by configured feature modules."""
        return self._build_browser_action_tools()

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
        run_id = request_id
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

    def _ensure_automation_ready(self) -> None:
        """Validate that the browser automation runtime is wired."""
        if self.browser_automation is None:
            raise NotImplementedError(
                "Browser automation class is not wired yet. "
                "Set `self.browser_automation` to an implementation."
            )

    def _get_or_create_url_extractor_llm(self) -> LLMClient:
        """Return the shared secondary LLM client used by URL extraction tool calls."""
        if self.url_extractor_llm is not None:
            return self.url_extractor_llm

        llm_cfg = self.llm_cfg
        extractor_cfg = self.url_extractor_cfg

        model = str(extractor_cfg.get("model") or llm_cfg.get("model") or "gpt-4o-mini")
        temperature = float(extractor_cfg.get("temperature", 0.0))
        enable_web_search = bool(extractor_cfg.get("enable_web_search", False))

        kwargs = extractor_cfg.get("kwargs", {})
        if not isinstance(kwargs, dict):
            kwargs = {}

        self.url_extractor_llm = LLMClient(
            model=model,
            temperature=temperature,
            enable_web_search=enable_web_search,
            **kwargs,
        )
        return self.url_extractor_llm

    async def _extract_actionable_urls(self, user_prompt: str) -> Dict[str, Any]:
        """Classify actionable URLs from user prompt using a direct LLM call."""
        llm = self._get_or_create_url_extractor_llm()
        prompt = render_prompt_template(
            self.url_extractor_user_prompt_template,
            replacements={"user_prompt": str(user_prompt or "").strip()},
        )
        response = await llm.send_message(
            message=prompt,
            system_message=self.url_extractor_system_prompt,
        )
        parsed = self._extract_json_obj(str(getattr(response, "content", "") or ""))
        return self._normalize_url_extractor_payload(parsed)

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
            return {**state, "refined_user_prompt": refined_prompt}

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
            return {**state, "browser_result": browser_result}

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
        system_message = str(self.task_refiner_system_prompt or "").strip()
        user_message = render_prompt_template(
            str(self.task_refiner_user_prompt_template or ""),
            replacements={
                "target_url": target_url,
                "user_prompt": user_prompt,
            },
        )
        response = await llm.send_message(message=user_message, system_message=system_message)
        objective = str(response.content or "").strip()
        return objective or user_prompt

    def _create_task_refiner_llm_client(self) -> LLMClient:
        """Create a fresh LLM client for the task-refinement node."""
        llm_cfg = self.llm_cfg
        refiner_cfg = self.task_refiner_cfg

        model = str(refiner_cfg.get("model") or llm_cfg.get("model") or "gpt-4o-mini")
        temperature = float(refiner_cfg.get("temperature", 0.1))
        enable_web_search = bool(refiner_cfg.get("enable_web_search", False))
        kwargs = refiner_cfg.get("kwargs", {})
        if not isinstance(kwargs, dict):
            kwargs = {}
        return LLMClient(
            model=model,
            temperature=temperature,
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
        out = await run_method(run_config=per_url_run_config, tools=browser_tools)
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

    def _normalize_url_extractor_payload(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize URL extractor payload into the expected schema."""
        result: Dict[str, Any] = {
            "actionable_urls": (parsed or {}).get("actionable_urls", []),
            "reference_urls": (parsed or {}).get("reference_urls", []),
            "excluded_urls": (parsed or {}).get("excluded_urls", []),
            "rationale_by_url": (parsed or {}).get("rationale_by_url", {}),
            "needs_clarification": bool((parsed or {}).get("needs_clarification", False)),
            "clarification_question": (parsed or {}).get("clarification_question"),
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

        return result

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
