"""PlaywrightAgent scaffold."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import uuid
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from moose.framework.agent_core.prompt_loader import (
    load_prompt_text,
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
from workflow.dispatcher_agent import DispatcherAgentNode
from workflow.network_analyst_agent import NetworkAnalystNode
from workflow.runtime_analyst_agent import RuntimeAnalystNode

from langgraph.graph import StateGraph


def _merge_subagent_outputs(
    left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(left or {})
    merged.update(dict(right or {}))
    return merged


def _sum_llm_metrics(
    left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    base = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "total_cost": 0.0}
    out = dict(base)
    for src in (left or {}, right or {}):
        try:
            out["input_tokens"] += int(src.get("input_tokens", 0) or 0)
        except Exception:
            pass
        try:
            out["output_tokens"] += int(src.get("output_tokens", 0) or 0)
        except Exception:
            pass
        try:
            out["total_tokens"] += int(src.get("total_tokens", 0) or 0)
        except Exception:
            pass
        try:
            out["total_cost"] += float(src.get("total_cost", 0.0) or 0.0)
        except Exception:
            pass
    return out


class WorkflowState(TypedDict, total=False):
    target_url: str
    user_prompt: str
    raw_user_prompt: str
    content_mode: str
    run_config: Any
    run_state: Any
    dispatch_round: int
    max_dispatch_rounds: int
    max_records_per_subagent: int
    dispatcher_mode: str
    dispatch_plan: Dict[str, Any]
    selected_agents: List[str]
    subagent_tasks: Dict[str, Any]
    expected_agents: List[str]
    subagent_outputs: Annotated[Dict[str, Any], _merge_subagent_outputs]
    llm_metrics: Annotated[Dict[str, Any], _sum_llm_metrics]
    merged_subagent_output: Dict[str, Any]
    final_response: Any


class PlaywrightAgent(BaseAgent):
    """Agent scaffold for browser automation with Playwright."""

    name = "playwright_agent"
    description = "Browser automation agent using Playwright"

    def __init__(self, config_path: Optional[str] = None, debug: bool = False):
        """Initialize the agent and feature placeholders."""
        super().__init__(config_path, debug=debug)
        custom = self.config.get("custom", {}) if isinstance(self.config.get("custom"), dict) else {}
        url_cfg = custom.get("url_extractor", {}) if isinstance(custom.get("url_extractor"), dict) else {}
        task_refiner_cfg = custom.get("task_refiner", {}) if isinstance(custom.get("task_refiner"), dict) else {}
        dispatcher_cfg = custom.get("dispatcher", {}) if isinstance(custom.get("dispatcher"), dict) else {}
        network_analyst_cfg = (
            custom.get("network_analyst") if isinstance(custom.get("network_analyst"), dict) else {}
        )
        runtime_analyst_cfg = (
            custom.get("runtime_analyst") if isinstance(custom.get("runtime_analyst"), dict) else {}
        )
        workflow_cfg = custom.get("workflow") if isinstance(custom.get("workflow"), dict) else {}

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
        self.dispatcher_system_prompt = load_prompt_text(
            path=str(dispatcher_cfg.get("system_prompt_path") or ""),
            logger=self.logger,
            label="playwright_agent.custom.dispatcher.system_prompt_path",
            required=True,
        )
        self.network_analyst_system_prompt = (
            load_prompt_text(
                path=str(network_analyst_cfg.get("system_prompt_path") or ""),
                logger=self.logger,
                label="playwright_agent.custom.network_analyst.system_prompt_path",
                required=False,
            )
            or (
                "You are network_analyst. Analyze captured network request/response evidence and return "
                "strict JSON envelope with concise findings and evidence refs. Avoid raw full-body dumps."
            )
        )
        self.runtime_analyst_system_prompt = (
            load_prompt_text(
                path=str(runtime_analyst_cfg.get("system_prompt_path") or ""),
                logger=self.logger,
                label="playwright_agent.custom.runtime_analyst.system_prompt_path",
                required=False,
            )
            or (
                "You are runtime_analyst. Analyze captured console/runtime errors and return strict JSON "
                "envelope with concise findings and evidence refs."
            )
        )

        self.url_extractor_llm: Optional[LLMClient] = None
        self.dispatcher_llm: Optional[LLMClient] = None
        self.url_extractor_cfg: Dict[str, Any] = url_cfg
        self.task_refiner_cfg: Dict[str, Any] = task_refiner_cfg
        self.dispatcher_cfg: Dict[str, Any] = dispatcher_cfg
        self.network_analyst_cfg: Dict[str, Any] = network_analyst_cfg
        self.runtime_analyst_cfg: Dict[str, Any] = runtime_analyst_cfg
        self.default_model = str(dispatcher_cfg.get("model") or "gpt-5.2")
        self.default_temperature = float(dispatcher_cfg.get("temperature", 0.2))
        self.default_enable_web_search = bool(dispatcher_cfg.get("enable_web_search", False))
        try:
            self.max_dispatch_rounds = max(1, int(workflow_cfg.get("max_dispatch_rounds", 3) or 3))
        except Exception:
            self.max_dispatch_rounds = 3
        try:
            self.max_records_per_subagent = max(1, int(workflow_cfg.get("max_records_per_subagent", 200) or 200))
        except Exception:
            self.max_records_per_subagent = 200
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
        self.browser_automation: BrowserAutomation = BrowserAutomation(
            session_manager=self.session_manager,
            page_actions=self.page_actions,
            network_inspector=self.network_inspector,
            downloads=self.downloads,
            llm_settings={
                "model": self.default_model,
                "temperature": self.default_temperature,
                "enable_web_search": self.default_enable_web_search,
            },
            logger=self.logger,
        )
        self.dispatcher_node_impl = DispatcherAgentNode(
            create_llm_client=self._create_llm_client,
            build_tools=self._build_dispatcher_tools,
            system_prompt=self.dispatcher_system_prompt,
            dispatcher_cfg=self.dispatcher_cfg,
            normalize_dispatch_plan=self._normalize_dispatch_plan,
            resolve_content_mode=self._resolve_content_mode,
            max_dispatch_rounds=self.max_dispatch_rounds,
        )
        self.network_analyst_node_impl = NetworkAnalystNode(
            network_inspector=self.network_inspector,
            create_llm_client=self._create_llm_client,
            build_tools=self._build_network_analyst_tools,
            role_cfg=self.network_analyst_cfg,
            system_prompt=self.network_analyst_system_prompt,
            max_records_per_subagent=self.max_records_per_subagent,
        )
        self.runtime_analyst_node_impl = RuntimeAnalystNode(
            network_inspector=self.network_inspector,
            create_llm_client=self._create_llm_client,
            build_tools=self._build_runtime_analyst_tools,
            role_cfg=self.runtime_analyst_cfg,
            system_prompt=self.runtime_analyst_system_prompt,
            max_records_per_subagent=self.max_records_per_subagent,
        )

    async def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Primary async processing endpoint for browser automation.

        Input must be a direct JSON payload (dict) with required keys:
        - user_prompt
        """
        if not isinstance(input_data, dict):
            raise TypeError("Payload must be a JSON object with direct keys")

        payload: Dict[str, Any] = dict(input_data)
        user_prompt = str(payload.get("user_prompt") or "").strip()
        if not user_prompt:
            raise ValueError("Missing required field: user_prompt")

        normalized_payload = dict(payload)
        normalized_payload["user_prompt"] = user_prompt
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

        result_by_url: Dict[str, Any] = {}
        for target_url in actionable_urls:
            try:
                url_result = await self._run_dispatcher_workflow(
                    target_url=target_url,
                    user_prompt=user_prompt,
                    base_run_config=base_run_config,
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

    def _build_dispatcher_tools(self) -> List[Any]:
        """Dispatcher gets page action tools only."""
        if self.page_actions is None:
            return []
        get_tools = getattr(self.page_actions, "get_tools", None)
        if not callable(get_tools):
            return []
        out = get_tools()
        return out if isinstance(out, list) else []

    def _build_network_analyst_tools(self) -> List[Any]:
        """Network analyst gets request/resource tools only."""
        if self.network_inspector is None:
            return []
        get_tools = getattr(self.network_inspector, "get_tools", None)
        if not callable(get_tools):
            return []
        names = {"browser_get_resource_source", "browser_list_websocket_frames"}
        tools: List[Any] = []
        for tool in (get_tools() or []):
            name = str(getattr(tool, "name", "") or "")
            if name in names:
                tools.append(tool)
        return tools

    def _build_runtime_analyst_tools(self) -> List[Any]:
        """Runtime analyst gets console-focused tools only."""
        if self.network_inspector is None:
            return []
        get_tools = getattr(self.network_inspector, "get_tools", None)
        if not callable(get_tools):
            return []
        names = {"browser_list_websocket_frames"}
        tools: List[Any] = []
        for tool in (get_tools() or []):
            name = str(getattr(tool, "name", "") or "")
            if name in names:
                tools.append(tool)
        return tools

    def _create_llm_client(self, role_cfg: Dict[str, Any], tools: Optional[List[Any]] = None) -> LLMClient:
        """Create role-specific LLM client with optional scoped tools."""
        model = str(role_cfg.get("model") or self.default_model)
        temperature = float(role_cfg.get("temperature", self.default_temperature))
        enable_web_search = bool(role_cfg.get("enable_web_search", self.default_enable_web_search))
        kwargs = role_cfg.get("kwargs", {})
        if not isinstance(kwargs, dict):
            kwargs = {}
        return LLMClient(
            model=model,
            temperature=temperature,
            enable_web_search=enable_web_search,
            tools=(tools or []),
            **kwargs,
        )

    async def _run_dispatcher_workflow(
        self,
        *,
        target_url: str,
        user_prompt: str,
        base_run_config: RunConfig,
    ) -> Dict[str, Any]:
        """Run dispatcher-centered LangGraph workflow for one URL."""
        run_config = replace(
            base_run_config,
            run_id=str(uuid.uuid4()),
            start_url=target_url,
            user_prompt=user_prompt,
            system_prompt=self._compose_system_prompt(),
        )
        run_state = await self.session_manager.start(run_config)
        self.run_state = run_state
        if self.event_logger is not None:
            self.event_logger.start()
            self.event_logger.init_run(
                run_id=str(run_state.run_id),
                request_id=str(run_state.request_id),
                agent_name=self.name,
            )
        if self.network_inspector is not None:
            self.network_inspector.set_run_state(run_state)
            await self.network_inspector.attach_listeners(run_state)
            if run_config.capture_network or run_config.capture_console:
                await self.network_inspector.start_capture(run_state)
        if self.page_actions is not None:
            try:
                self.page_actions.set_network_inspector(self.network_inspector)
            except Exception:
                pass

        state: WorkflowState = {
            "target_url": target_url,
            "user_prompt": user_prompt,
            "raw_user_prompt": user_prompt,
            "content_mode": "distilled",
            "run_config": run_config,
            "run_state": run_state,
            "dispatch_round": 0,
            "max_dispatch_rounds": self.max_dispatch_rounds,
            "max_records_per_subagent": self.max_records_per_subagent,
            "dispatcher_mode": "plan",
            "dispatch_plan": {},
            "selected_agents": [],
            "subagent_tasks": {},
            "expected_agents": [],
            "subagent_outputs": {},
            "llm_metrics": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "total_cost": 0.0},
            "merged_subagent_output": {},
            "final_response": None,
        }

        try:
            graph = StateGraph(WorkflowState)
            graph.add_node("task_refiner", self._task_refiner_node)
            graph.add_node("dispatcher", self._dispatcher_node)
            graph.add_node("network_analyst", self._network_analyst_node)
            graph.add_node("runtime_analyst", self._runtime_analyst_node)
            graph.add_node("merge_subagent_outputs", self._merge_subagent_outputs_node)
            graph.set_entry_point("task_refiner")
            graph.add_edge("task_refiner", "dispatcher")
            graph.add_conditional_edges("dispatcher", self._route_from_dispatcher)
            graph.add_edge("network_analyst", "merge_subagent_outputs")
            graph.add_edge("runtime_analyst", "merge_subagent_outputs")
            graph.add_conditional_edges("merge_subagent_outputs", self._route_from_merge)
            app = graph.compile()
            final_state = await app.ainvoke(state)
            if isinstance(final_state, dict) and final_state.get("final_response") is not None:
                return final_state["final_response"]
            return {"status": "error", "error": "workflow_finished_without_final_response"}
        finally:
            if self.network_inspector is not None:
                try:
                    await self.network_inspector.stop_capture(run_state)
                except Exception:
                    pass
                try:
                    await self.network_inspector.detach_listeners(run_state)
                except Exception:
                    pass
                self.network_inspector.set_run_state(None)
            if self.event_logger is not None:
                try:
                    self.event_logger.complete_run(run_id=str(run_state.run_id), status="completed")
                except Exception:
                    pass
            try:
                await self.session_manager.close_context(run_state)
            except Exception:
                pass
            self.run_state = None

    async def _dispatcher_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatcher node delegate."""
        return await self.dispatcher_node_impl.run(state, extract_json_obj=self._extract_json_obj)

    async def _task_refiner_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Refine user task before dispatcher planning."""
        raw_user_prompt = str(state.get("raw_user_prompt") or state.get("user_prompt") or "")
        target_url = str(state.get("target_url") or "").strip()
        refined_prompt, llm_metrics = await self._refine_user_task_for_url(
            user_prompt=raw_user_prompt, target_url=target_url
        )
        updated = dict(state)
        updated["user_prompt"] = refined_prompt or raw_user_prompt
        updated["llm_metrics"] = _sum_llm_metrics(state.get("llm_metrics"), llm_metrics)
        return updated

    async def _dispatcher_finalize_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Compatibility wrapper for finalize path."""
        updated = dict(state)
        updated["dispatcher_mode"] = "final"
        return await self.dispatcher_node_impl.run(updated, extract_json_obj=self._extract_json_obj)

    async def _network_analyst_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Network analyst node delegate."""
        return await self.network_analyst_node_impl.run(
            state,
            normalize_envelope=self._normalize_subagent_envelope,
            extract_json_obj=self._extract_json_obj,
        )

    async def _runtime_analyst_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Runtime analyst node delegate."""
        return await self.runtime_analyst_node_impl.run(
            state,
            normalize_envelope=self._normalize_subagent_envelope,
            extract_json_obj=self._extract_json_obj,
        )

    async def _merge_subagent_outputs_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Fan-in merge delegate."""
        return self.dispatcher_node_impl.merge_subagent_outputs(state)

    def _route_from_dispatcher(self, state: Dict[str, Any]) -> Any:
        """Conditional router delegate for dispatcher fan-out."""
        return self.dispatcher_node_impl.route_from_dispatcher(state)

    def _route_from_merge(self, state: Dict[str, Any]) -> str:
        """Route merge output delegate."""
        return self.dispatcher_node_impl.route_from_merge(state)

    @staticmethod
    def _normalize_dispatch_plan(parsed: Dict[str, Any]) -> Dict[str, Any]:
        allowed = {"network_analyst", "runtime_analyst"}
        selected = []
        for item in (parsed or {}).get("selected_agents", []) or []:
            name = str(item or "").strip()
            if name in allowed and name not in selected:
                selected.append(name)
        execution_mode = str((parsed or {}).get("execution_mode") or "parallel").strip().lower()
        if execution_mode not in {"parallel", "single"}:
            execution_mode = "parallel"
        content_mode = str((parsed or {}).get("content_mode") or "distilled").strip().lower()
        if content_mode not in {"distilled", "full_body"}:
            content_mode = "distilled"
        tasks = (parsed or {}).get("subagent_tasks", {})
        if not isinstance(tasks, dict):
            tasks = {}
        normalized_tasks: Dict[str, Dict[str, Any]] = {}
        for agent_name in ("network_analyst", "runtime_analyst"):
            entry = tasks.get(agent_name)
            if isinstance(entry, dict):
                task_text = str(entry.get("task") or "").strip()
                filter_raw = entry.get("filter")
            else:
                task_text = str(entry or "").strip()
                filter_raw = None
            if agent_name == "network_analyst":
                filter_obj = PlaywrightAgent._normalize_network_filters(filter_raw)
            else:
                filter_obj = PlaywrightAgent._normalize_runtime_filters(filter_raw)
            if task_text or filter_obj:
                normalized_entry: Dict[str, Any] = {}
                if task_text:
                    normalized_entry["task"] = task_text
                if filter_obj:
                    normalized_entry["filter"] = filter_obj
                normalized_tasks[agent_name] = normalized_entry
        return {
            "finish": bool((parsed or {}).get("finish", False)),
            "final_response": str((parsed or {}).get("final_response") or "").strip(),
            "selected_agents": selected,
            "execution_mode": execution_mode,
            "content_mode": content_mode,
            "subagent_tasks": normalized_tasks,
        }

    @staticmethod
    def _normalize_network_filters(raw: Any) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            return {}
        out: Dict[str, Any] = {}
        allowed_resource_types = {
            "document",
            "stylesheet",
            "image",
            "media",
            "font",
            "script",
            "texttrack",
            "xhr",
            "fetch",
            "eventsource",
            "websocket",
            "manifest",
            "other",
        }
        rt_raw = str(raw.get("resource_type") or "").strip().lower()
        rt_parts = [v.strip() for v in rt_raw.split(",") if v.strip()]
        rt_valid = [v for v in rt_parts if v in allowed_resource_types]
        if rt_valid:
            out["resource_type"] = ",".join(rt_valid)
        mime_raw = str(raw.get("mime_type") or "").strip().lower()
        mime_parts = [v.strip()[:128] for v in mime_raw.split(",") if v.strip()]
        if mime_parts:
            out["mime_type"] = ",".join(mime_parts)
        status_min = PlaywrightAgent._safe_status_int(raw.get("status_min"))
        status_max = PlaywrightAgent._safe_status_int(raw.get("status_max"))
        if status_min is not None:
            out["status_min"] = status_min
        if status_max is not None:
            out["status_max"] = status_max
        if status_min is not None and status_max is not None and status_min > status_max:
            out["status_min"], out["status_max"] = out["status_max"], out["status_min"]
        url_contains = str(raw.get("url_contains") or "").strip()
        if url_contains:
            out["url_contains"] = url_contains[:256]
        url_regex = str(raw.get("url_regex") or "").strip()
        if url_regex:
            out["url_regex"] = url_regex[:256]
        return out

    @staticmethod
    def _normalize_runtime_filters(raw: Any) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            return {}
        out: Dict[str, Any] = {}
        allowed_levels = {"error", "warning", "warn", "info", "log", "debug", "trace"}
        level_raw = str(raw.get("level") or "").strip().lower()
        level_parts = [v.strip() for v in level_raw.split(",") if v.strip()]
        level_valid = [v for v in level_parts if v in allowed_levels]
        if level_valid:
            out["level"] = ",".join(level_valid)
        text_contains = str(raw.get("text_contains") or "").strip()
        if text_contains:
            out["text_contains"] = text_contains[:256]
        return out

    @staticmethod
    def _safe_status_int(value: Any) -> Optional[int]:
        try:
            iv = int(value)
        except Exception:
            return None
        if iv < 100 or iv > 599:
            return None
        return iv

    @staticmethod
    def _resolve_content_mode(requested_mode: str, user_prompt: str, task_map: Dict[str, Any]) -> str:
        """Dispatcher-owned content mode resolution with security-aware override."""
        mode = str(requested_mode or "").strip().lower()
        if mode not in {"distilled", "full_body"}:
            mode = "distilled"
        network_task = ""
        runtime_task = ""
        if isinstance(task_map, dict):
            net_entry = task_map.get("network_analyst")
            rt_entry = task_map.get("runtime_analyst")
            network_task = (
                str((net_entry or {}).get("task") or "")
                if isinstance(net_entry, dict)
                else str(net_entry or "")
            )
            runtime_task = (
                str((rt_entry or {}).get("task") or "")
                if isinstance(rt_entry, dict)
                else str(rt_entry or "")
            )
        combined_text = " ".join(
            [
                str(user_prompt or ""),
                network_task,
                runtime_task,
            ]
        ).lower()
        security_markers = (
            "security",
            "vulnerability",
            "threat",
            "xss",
            "sqli",
            "csrf",
            "injection",
            "exploit",
            "audit",
            "pentest",
        )
        if any(marker in combined_text for marker in security_markers):
            return "full_body"
        return mode

    @staticmethod
    def _normalize_subagent_envelope(agent: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
        base = parsed if isinstance(parsed, dict) else {}
        finding_evidence_pairs = base.get("finding_evidence_pairs", [])
        if not isinstance(finding_evidence_pairs, list):
            finding_evidence_pairs = []
        return {
            "agent": agent,
            "task_id": str(base.get("task_id") or f"{agent}-task"),
            "status": str(base.get("status") or "ok"),
            "summary": str(base.get("summary") or ""),
            "finding_evidence_pairs": finding_evidence_pairs,
        }

    def _compose_system_prompt(self) -> str:
        """Return the built-in dispatcher system prompt."""
        return self.dispatcher_system_prompt

    def _build_run_config(self, payload: Dict[str, Any]) -> RunConfig:
        """Normalize request payload into immutable run configuration."""
        request_id = str(payload.get("request_id") or uuid.uuid4())
        run_id = request_id
        user_prompt = str(payload.get("user_prompt") or "").strip()
        merged_system_prompt = self._compose_system_prompt()

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

        extractor_cfg = self.url_extractor_cfg

        model = str(extractor_cfg.get("model") or self.default_model)
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

    async def _refine_user_task_for_url(self, *, user_prompt: str, target_url: str) -> tuple[str, Dict[str, Any]]:
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
        usage = getattr(response, "usage", {}) or {}
        metrics = {
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
            "total_cost": float(getattr(response, "cost", 0.0) or 0.0),
        }
        return objective or user_prompt, metrics

    def _create_task_refiner_llm_client(self) -> LLMClient:
        """Create a fresh LLM client for the task-refinement node."""
        refiner_cfg = self.task_refiner_cfg

        model = str(refiner_cfg.get("model") or self.default_model)
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
