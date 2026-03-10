"""Single-loop Playwright agent powered by llm_core."""

from __future__ import annotations

import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from moose.framework import BaseAgent
    from moose.framework.agent_core.prompt_loader import (
        load_prompt_text,
        load_system_prompt,
        render_prompt_template,
    )
    from moose.framework.llm_core import LLMClient
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from moose.framework import BaseAgent
    from moose.framework.agent_core.prompt_loader import (
        load_prompt_text,
        load_system_prompt,
        render_prompt_template,
    )
    from moose.framework.llm_core import LLMClient

try:
    from .browser.loop_runtime import BrowserAutomation
    from .browser.downloads import DownloadsFeature
    from .browser.event_logger import BrowserEventLogger
    from .browser.models import RunConfig
    from .browser.network_inspector import NetworkInspectorFeature
    from .browser.ref_page_actions import PageActionsFeature
    from .browser.session import BrowserSessionManager
except ImportError:
    from browser.loop_runtime import BrowserAutomation
    from browser.downloads import DownloadsFeature
    from browser.event_logger import BrowserEventLogger
    from browser.models import RunConfig
    from browser.network_inspector import NetworkInspectorFeature
    from browser.ref_page_actions import PageActionsFeature
    from browser.session import BrowserSessionManager

_URL_RE = re.compile(r"https?://[^\s)>\"']+")


class PlaywrightAgent(BaseAgent):
    """Browser automation agent using Playwright + semantic snapshots."""

    name = "playwright_agent"
    description = "Browser automation agent using Playwright"

    def __init__(self, config_path: Optional[str] = None, debug: bool = False):
        super().__init__(config_path, debug=debug)
        custom = self.config.get("custom", {}) if isinstance(self.config.get("custom"), dict) else {}
        browser_agent_cfg = (
            custom.get("browser_agent") if isinstance(custom.get("browser_agent"), dict) else {}
        )
        task_refiner_cfg = (
            custom.get("task_refiner") if isinstance(custom.get("task_refiner"), dict) else {}
        )
        summarizer_cfg = custom.get("summarizer") if isinstance(custom.get("summarizer"), dict) else {}
        runtime_cfg = (
            custom.get("browser_runtime") if isinstance(custom.get("browser_runtime"), dict) else {}
        )

        self.browser_agent_cfg: Dict[str, Any] = browser_agent_cfg
        self.task_refiner_cfg: Dict[str, Any] = task_refiner_cfg
        self.summarizer_cfg: Dict[str, Any] = summarizer_cfg
        self.runtime_cfg: Dict[str, Any] = runtime_cfg
        self.browser_system_prompt = load_system_prompt(
            system_prompt_path=str(browser_agent_cfg.get("system_prompt_path") or ""),
            skills_dir=str(browser_agent_cfg.get("skills_dir") or ""),
            logger=self.logger,
            label="playwright_agent.custom.browser_agent.system_prompt_path",
            required=True,
        )
        self.task_refiner_system_prompt = load_prompt_text(
            path=str(task_refiner_cfg.get("system_prompt_path") or ""),
            logger=self.logger,
            label="playwright_agent.custom.task_refiner.system_prompt_path",
            required=True,
        )
        self.task_refiner_user_prompt_template = load_prompt_text(
            path=str(task_refiner_cfg.get("user_prompt_path") or ""),
            logger=self.logger,
            label="playwright_agent.custom.task_refiner.user_prompt_path",
            required=True,
        )
        self.summarizer_system_prompt = load_prompt_text(
            path=str(summarizer_cfg.get("system_prompt_path") or ""),
            logger=self.logger,
            label="playwright_agent.custom.summarizer.system_prompt_path",
            required=True,
        )
        self.summarizer_user_prompt_template = load_prompt_text(
            path=str(summarizer_cfg.get("user_prompt_path") or ""),
            logger=self.logger,
            label="playwright_agent.custom.summarizer.user_prompt_path",
            required=True,
        )

        downloads_dir = str(runtime_cfg.get("downloads_dir") or "/tmp/playwright_agent")
        activity_db_path = str(runtime_cfg.get("activity_db_path") or "/tmp/playwright_agent/activity.db")

        self.session_manager = BrowserSessionManager(downloads_dir=downloads_dir)
        self.event_logger = BrowserEventLogger(db_path=Path(activity_db_path))
        self.page_actions = PageActionsFeature(
            session_manager=self.session_manager,
            event_logger=self.event_logger,
        )
        self.network_inspector = NetworkInspectorFeature(
            session_manager=self.session_manager,
            event_logger=self.event_logger,
        )
        self.page_actions.set_network_inspector(self.network_inspector)
        self.downloads = DownloadsFeature(
            session_manager=self.session_manager,
            event_logger=self.event_logger,
            base_tmp_dir=downloads_dir,
        )
        self.browser_automation = BrowserAutomation(
            session_manager=self.session_manager,
            page_actions=self.page_actions,
            network_inspector=self.network_inspector,
            downloads=self.downloads,
            llm_settings={
                "model": str(browser_agent_cfg.get("model") or "gpt-5.2"),
                "temperature": float(browser_agent_cfg.get("temperature", 0.2)),
                "enable_web_search": bool(browser_agent_cfg.get("enable_web_search", False)),
                "max_tool_iterations": int(browser_agent_cfg.get("max_tool_iterations", 24) or 24),
                "kwargs": (
                    dict(browser_agent_cfg.get("kwargs"))
                    if isinstance(browser_agent_cfg.get("kwargs"), dict)
                    else {}
                ),
            },
            logger=self.logger,
        )

    async def process(self, input_data: Any) -> Dict[str, Any]:
        """Primary async processing endpoint for browser automation."""
        payload = self._normalize_payload(input_data)
        user_prompt = str(payload.get("user_prompt") or "").strip()
        if not user_prompt:
            raise ValueError("Missing required field: user_prompt")

        task_refiner_result = await self._run_task_refiner(payload)
        refined_payload = dict(payload)
        requested_include_loop_events = bool(payload.get("include_loop_events", True))
        refined_prompt = str(task_refiner_result.get("refined_user_prompt") or "").strip()
        refined_target_url = str(task_refiner_result.get("target_url") or "").strip()
        if refined_prompt:
            refined_payload["user_prompt"] = refined_prompt
        if refined_target_url:
            refined_payload["start_url"] = refined_target_url

        run_config = self._build_run_config(refined_payload)
        result = await self.browser_automation.run(
            run_config=run_config,
            tools=self._build_tools(),
            include_loop_events=True,
        )
        if isinstance(result, dict):
            raw_browser_output = result.get("output")
            summary_result = await self._run_summarizer(
                payload=payload,
                task_refiner_result=task_refiner_result,
                browser_result=result,
            )
            if raw_browser_output is not None:
                result["browser_output"] = raw_browser_output
            result["output"] = str(summary_result.get("content") or "").strip() or str(
                raw_browser_output or result.get("error") or ""
            )
            if not requested_include_loop_events:
                loop_payload = result.get("loop")
                if isinstance(loop_payload, dict):
                    loop_payload.pop("events", None)
            result["task_refiner"] = task_refiner_result
            result["summarizer"] = {
                "status": summary_result.get("status"),
                "requested_return_format": summary_result.get("requested_return_format"),
                "model": summary_result.get("model"),
            }
            if summary_result.get("error"):
                result["summarizer"]["error"] = summary_result.get("error")
        return result

    async def browse_url(self, input_data: Any) -> Dict[str, Any]:
        """Compatibility alias for `process`."""
        return await self.process(input_data)

    async def analyze(self, input_data: Any) -> Dict[str, Any]:
        """Compatibility alias for `process`."""
        return await self.process(input_data)

    def health(self, _data: Dict[str, Any]) -> Dict[str, Any]:
        """Return health and runtime wiring status."""
        return {
            "status": "ok",
            "agent": self.name,
            "automation_ready": self.browser_automation is not None,
        }

    def _normalize_payload(self, input_data: Any) -> Dict[str, Any]:
        if isinstance(input_data, str):
            return {"user_prompt": input_data}
        if not isinstance(input_data, dict):
            raise TypeError("Payload must be a JSON object or plain prompt string")

        payload = dict(input_data)
        nested = payload.get("input")
        if "user_prompt" not in payload and isinstance(nested, dict):
            payload = dict(nested)
        return payload

    def _build_tools(self) -> List[Any]:
        tools: List[Any] = []
        for feature in (self.page_actions, self.network_inspector, self.downloads):
            get_tools = getattr(feature, "get_tools", None)
            if callable(get_tools):
                out = get_tools()
                if isinstance(out, list):
                    tools.extend(out)
        return tools

    def _build_run_config(self, payload: Dict[str, Any]) -> RunConfig:
        user_prompt = str(payload.get("user_prompt") or "").strip()
        start_url = str(payload.get("start_url") or "").strip() or self._extract_first_url(user_prompt)
        system_prompt_override = str(payload.get("system_prompt") or "").strip()
        system_prompt = self.browser_system_prompt
        if system_prompt_override:
            system_prompt = f"{system_prompt}\n\nTask-specific constraints:\n{system_prompt_override}"

        return RunConfig(
            run_id=str(payload.get("run_id") or uuid.uuid4()),
            request_id=str(payload.get("request_id") or uuid.uuid4()),
            start_url=start_url or None,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            headless=bool(payload.get("headless", True)),
            extension_path=str(payload.get("extension_path") or "").strip() or None,
            timeout_ms=self._coerce_positive_int(payload.get("timeout_ms"), fallback=30000),
            capture_network=bool(payload.get("capture_network", True)),
            capture_console=bool(payload.get("capture_console", True)),
            capture_response_bodies=bool(payload.get("capture_response_bodies", True)),
            accept_downloads=bool(payload.get("accept_downloads", True)),
            ignore_https_errors=bool(
                payload.get("ignore_https_errors", self.runtime_cfg.get("ignore_https_errors", False))
            ),
        )

    def _extract_first_url(self, text: str) -> str:
        match = _URL_RE.search(str(text or ""))
        return str(match.group(0)).strip() if match else ""

    def _coerce_positive_int(self, value: Any, *, fallback: int) -> int:
        try:
            resolved = int(value)
        except Exception:
            return fallback
        return resolved if resolved > 0 else fallback

    def _create_task_refiner_llm_client(self) -> LLMClient:
        cfg = self.task_refiner_cfg or {}
        model = str(cfg.get("model") or "gpt-5.2")
        temperature = float(cfg.get("temperature", 0.0))
        enable_web_search = bool(cfg.get("enable_web_search", False))
        kwargs = cfg.get("kwargs", {})
        if not isinstance(kwargs, dict):
            kwargs = {}
        return LLMClient(
            model=model,
            temperature=temperature,
            enable_web_search=enable_web_search,
            **kwargs,
        )

    def _create_summarizer_llm_client(self) -> LLMClient:
        cfg = self.summarizer_cfg or {}
        model = str(cfg.get("model") or "gpt-5.2")
        temperature = float(cfg.get("temperature", 0.1))
        enable_web_search = bool(cfg.get("enable_web_search", False))
        kwargs = cfg.get("kwargs", {})
        if not isinstance(kwargs, dict):
            kwargs = {}
        return LLMClient(
            model=model,
            temperature=temperature,
            enable_web_search=enable_web_search,
            **kwargs,
        )

    async def _run_task_refiner(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raw_user_prompt = str(payload.get("user_prompt") or "").strip()
        start_url = str(payload.get("start_url") or "").strip() or self._extract_first_url(raw_user_prompt)
        llm = self._create_task_refiner_llm_client()
        prompt = render_prompt_template(
            str(self.task_refiner_user_prompt_template or ""),
            replacements={
                "user_prompt": raw_user_prompt,
                "start_url": start_url or "(none)",
            },
        )
        try:
            response = await llm.send_message(
                message=prompt,
                system_message=str(self.task_refiner_system_prompt or "").strip(),
            )
            parsed = self._extract_json_obj(str(getattr(response, "content", "") or ""))
            refined_user_prompt = str(parsed.get("refined_user_prompt") or "").strip() or raw_user_prompt
            target_url = str(parsed.get("target_url") or "").strip() or start_url
            return_format = str(parsed.get("return_format") or "").strip()
            raw_notes = parsed.get("notes")
            notes = [str(item).strip() for item in raw_notes if str(item).strip()] if isinstance(raw_notes, list) else []
            return {
                "status": "ok",
                "raw_user_prompt": raw_user_prompt,
                "refined_user_prompt": refined_user_prompt,
                "target_url": target_url,
                "return_format": return_format,
                "notes": notes,
            }
        except Exception as exc:
            return {
                "status": "fallback",
                "raw_user_prompt": raw_user_prompt,
                "refined_user_prompt": raw_user_prompt,
                "target_url": start_url,
                "return_format": "",
                "notes": [],
                "error": f"{type(exc).__name__}: {exc}",
            }

    async def _run_summarizer(
        self,
        *,
        payload: Dict[str, Any],
        task_refiner_result: Dict[str, Any],
        browser_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        requested_return_format = str(task_refiner_result.get("return_format") or "").strip()
        execution_summary = self._build_summarizer_context(
            task_refiner_result=task_refiner_result,
            browser_result=browser_result,
        )
        notes = task_refiner_result.get("notes")
        notes_text = "\n".join(
            f"- {str(item).strip()}"
            for item in (notes if isinstance(notes, list) else [])
            if str(item).strip()
        ) or "(none)"
        fallback_content = self._build_summary_fallback(
            payload=payload,
            task_refiner_result=task_refiner_result,
            browser_result=browser_result,
            execution_summary=execution_summary,
        )
        llm = self._create_summarizer_llm_client()
        prompt = render_prompt_template(
            str(self.summarizer_user_prompt_template or ""),
            replacements={
                "raw_user_prompt": str(task_refiner_result.get("raw_user_prompt") or payload.get("user_prompt") or ""),
                "refined_user_prompt": str(task_refiner_result.get("refined_user_prompt") or ""),
                "target_url": str(task_refiner_result.get("target_url") or browser_result.get("start_url") or "(none)"),
                "user_specified_return_format": requested_return_format or "(none specified)",
                "task_refiner_notes": notes_text,
                "execution_summary_json": json.dumps(
                    execution_summary,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
            },
        )
        try:
            response = await llm.send_message(
                message=prompt,
                system_message=str(self.summarizer_system_prompt or "").strip(),
            )
            content = str(getattr(response, "content", "") or "").strip()
            if not content:
                raise ValueError("Summarizer returned empty content")
            return {
                "status": "ok",
                "content": content,
                "requested_return_format": requested_return_format,
                "model": getattr(response, "model", None),
            }
        except Exception as exc:
            return {
                "status": "fallback",
                "content": fallback_content,
                "requested_return_format": requested_return_format,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _build_summarizer_context(
        self,
        *,
        task_refiner_result: Dict[str, Any],
        browser_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        loop_payload = browser_result.get("loop")
        if not isinstance(loop_payload, dict):
            loop_payload = {}
        events = loop_payload.get("events")
        activity_summary = self._extract_activity_from_events(events if isinstance(events, list) else [])
        errors: List[Dict[str, Any]] = list(activity_summary.get("errors", []))
        browser_error = str(browser_result.get("error") or "").strip()
        if browser_error:
            errors.insert(
                0,
                {
                    "source": "browser_run",
                    "message": self._truncate_text(browser_error, 500),
                },
            )
        loop_error = str(loop_payload.get("error_message") or "").strip()
        if loop_error:
            errors.append(
                {
                    "source": "agent_loop",
                    "message": self._truncate_text(loop_error, 500),
                }
            )
        return {
            "run_status": str(browser_result.get("status") or "unknown"),
            "start_url": str(browser_result.get("start_url") or task_refiner_result.get("target_url") or ""),
            "final_url": str(browser_result.get("final_url") or ""),
            "browser_output": self._truncate_text(str(browser_result.get("output") or ""), 2000),
            "browser_error": self._truncate_text(browser_error, 1000) if browser_error else None,
            "browser_state": self._summarize_browser_state(browser_result.get("browser_state")),
            "snapshot": self._summarize_snapshot(browser_result.get("snapshot")),
            "network": self._summarize_network(browser_result.get("network")),
            "downloads": self._summarize_downloads(browser_result.get("downloads")),
            "artifacts": self._summarize_artifacts(browser_result.get("artifacts")),
            "usage": self._sanitize_for_summary(browser_result.get("usage"), max_string=120, max_items=12, depth=2),
            "cost": browser_result.get("cost"),
            "loop": self._summarize_loop_payload(loop_payload),
            "activity": activity_summary.get("activity", []),
            "errors": errors,
        }

    def _summarize_loop_payload(self, loop_payload: Dict[str, Any]) -> Dict[str, Any]:
        events = loop_payload.get("events")
        return {
            "run_id": loop_payload.get("run_id"),
            "request_id": loop_payload.get("request_id"),
            "stop_reason": loop_payload.get("stop_reason"),
            "iteration_count": loop_payload.get("iteration_count"),
            "event_count": len(events) if isinstance(events, list) else 0,
            "total_usage": self._sanitize_for_summary(
                loop_payload.get("total_usage"),
                max_string=100,
                max_items=12,
                depth=2,
            ),
            "total_cost": loop_payload.get("total_cost"),
            "error_type": loop_payload.get("error_type"),
            "error_message": self._truncate_text(str(loop_payload.get("error_message") or ""), 500)
            if loop_payload.get("error_message")
            else None,
        }

    def _extract_activity_from_events(self, events: List[Any]) -> Dict[str, List[Dict[str, Any]]]:
        activity: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("event_type") or "")
            if event_type == "tool_call_success":
                activity.append(
                    {
                        "iteration": event.get("iteration"),
                        "tool": event.get("tool_name"),
                        "status": "success",
                        "args": self._sanitize_for_summary(
                            event.get("tool_args"),
                            max_string=160,
                            max_items=8,
                            depth=2,
                        ),
                        "result": self._sanitize_for_summary(
                            event.get("tool_result"),
                            max_string=220,
                            max_items=10,
                            depth=2,
                        ),
                    }
                )
            elif event_type == "tool_call_error":
                error_item = {
                    "iteration": event.get("iteration"),
                    "tool": event.get("tool_name"),
                    "status": "error",
                    "args": self._sanitize_for_summary(
                        event.get("tool_args"),
                        max_string=160,
                        max_items=8,
                        depth=2,
                    ),
                    "error_type": event.get("error_type"),
                    "error_message": self._truncate_text(str(event.get("error_message") or ""), 400),
                }
                activity.append(error_item)
                errors.append(error_item)
        max_items = 40
        if len(activity) > max_items:
            activity = activity[:max_items] + [
                {
                    "status": "truncated",
                    "message": f"{len(activity) - max_items} additional tool events omitted.",
                }
            ]
        if len(errors) > 20:
            errors = errors[:20]
        return {"activity": activity, "errors": errors}

    def _summarize_browser_state(self, browser_state: Any) -> Dict[str, Any]:
        if not isinstance(browser_state, dict):
            return {}
        tabs = browser_state.get("tabs")
        summarized_tabs: List[Dict[str, Any]] = []
        if isinstance(tabs, list):
            for item in tabs[:6]:
                if not isinstance(item, dict):
                    continue
                summarized_tabs.append(
                    {
                        "page_id": item.get("page_id"),
                        "is_active": item.get("is_active"),
                        "url": item.get("url"),
                        "title": self._truncate_text(str(item.get("title") or ""), 160)
                        if item.get("title")
                        else None,
                    }
                )
        return {
            "active_tab_id": browser_state.get("active_tab_id"),
            "url": browser_state.get("url"),
            "title": self._truncate_text(str(browser_state.get("title") or ""), 200)
            if browser_state.get("title")
            else None,
            "snapshot_id": browser_state.get("snapshot_id"),
            "tab_count": len(tabs) if isinstance(tabs, list) else 0,
            "tabs": summarized_tabs,
        }

    def _summarize_snapshot(self, snapshot: Any) -> Dict[str, Any]:
        if not isinstance(snapshot, dict):
            return {}
        refs = snapshot.get("refs")
        metadata = snapshot.get("metadata")
        return {
            "snapshot_id": snapshot.get("snapshot_id"),
            "page_id": snapshot.get("page_id"),
            "url": snapshot.get("url"),
            "title": self._truncate_text(str(snapshot.get("title") or ""), 200)
            if snapshot.get("title")
            else None,
            "ref_count": len(refs) if isinstance(refs, dict) else None,
            "metadata": self._sanitize_for_summary(metadata, max_string=140, max_items=10, depth=2),
        }

    def _summarize_downloads(self, downloads: Any) -> Dict[str, Any]:
        if not isinstance(downloads, list):
            return {"count": 0, "items": []}
        items: List[Dict[str, Any]] = []
        for entry in downloads[:10]:
            if not isinstance(entry, dict):
                continue
            items.append(
                {
                    "download_id": entry.get("download_id"),
                    "status": entry.get("status"),
                    "filename": entry.get("filename"),
                    "suggested_filename": entry.get("suggested_filename"),
                    "url": entry.get("url"),
                    "saved_path": entry.get("saved_path"),
                }
            )
        return {"count": len(downloads), "items": items}

    def _summarize_network(self, network: Any) -> Dict[str, Any]:
        if not isinstance(network, dict):
            return {}
        return {
            "counts": self._sanitize_for_summary(
                network.get("counts"),
                max_string=100,
                max_items=10,
                depth=2,
            ),
            "recent_requests": self._sanitize_for_summary(
                network.get("recent_requests"),
                max_string=180,
                max_items=10,
                depth=2,
            ),
            "failed_requests": self._sanitize_for_summary(
                network.get("failed_requests"),
                max_string=180,
                max_items=10,
                depth=2,
            ),
            "console_errors": self._sanitize_for_summary(
                network.get("console_errors"),
                max_string=180,
                max_items=10,
                depth=2,
            ),
            "recent_websocket": self._sanitize_for_summary(
                network.get("recent_websocket"),
                max_string=180,
                max_items=10,
                depth=2,
            ),
            "error": self._truncate_text(str(network.get("error") or ""), 300) if network.get("error") else None,
        }

    def _summarize_artifacts(self, artifacts: Any) -> Dict[str, Any]:
        if not isinstance(artifacts, dict):
            return {}
        return {
            "artifacts_dir": artifacts.get("artifacts_dir"),
            "downloads_dir": artifacts.get("downloads_dir"),
            "screenshots_dir": artifacts.get("screenshots_dir"),
            "user_data_dir": artifacts.get("user_data_dir"),
            "activity_db_path": artifacts.get("activity_db_path"),
        }

    def _sanitize_for_summary(
        self,
        value: Any,
        *,
        max_string: int = 200,
        max_items: int = 10,
        depth: int = 2,
    ) -> Any:
        if depth < 0:
            return "..."
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return self._truncate_text(value, max_string)
        if isinstance(value, dict):
            out: Dict[str, Any] = {}
            items = list(value.items())
            for index, (key, item) in enumerate(items):
                if index >= max_items:
                    out["truncated_fields"] = len(items) - max_items
                    break
                key_text = str(key)
                if key_text == "base64":
                    out[key_text] = "<omitted>"
                    continue
                if key_text == "snapshot_text":
                    snapshot_text = str(item or "")
                    out["snapshot_text_excerpt"] = self._truncate_text(snapshot_text, min(max_string, 500))
                    out["snapshot_text_length"] = len(snapshot_text)
                    continue
                if key_text == "refs" and isinstance(item, dict):
                    out["ref_count"] = len(item)
                    continue
                if key_text == "page_text" and isinstance(item, list):
                    out["page_text_count"] = len(item)
                    continue
                if key_text == "events" and isinstance(item, list):
                    out["event_count"] = len(item)
                    continue
                if key_text == "downloads" and isinstance(item, list):
                    out["download_count"] = len(item)
                    out["downloads"] = [
                        self._sanitize_for_summary(entry, max_string=max_string, max_items=6, depth=depth - 1)
                        for entry in item[:5]
                    ]
                    continue
                if key_text == "tabs" and isinstance(item, list):
                    out["tab_count"] = len(item)
                    out["tabs"] = [
                        self._sanitize_for_summary(entry, max_string=max_string, max_items=6, depth=depth - 1)
                        for entry in item[:5]
                    ]
                    continue
                out[key_text] = self._sanitize_for_summary(
                    item,
                    max_string=max_string,
                    max_items=max_items,
                    depth=depth - 1,
                )
            return out
        if isinstance(value, list):
            items = [
                self._sanitize_for_summary(item, max_string=max_string, max_items=max_items, depth=depth - 1)
                for item in value[:max_items]
            ]
            if len(value) > max_items:
                items.append(f"... {len(value) - max_items} more items")
            return items
        return self._truncate_text(str(value), max_string)

    def _truncate_text(self, text: str, limit: int) -> str:
        value = str(text or "")
        return value if len(value) <= limit else value[:limit] + "..."

    def _build_summary_fallback(
        self,
        *,
        payload: Dict[str, Any],
        task_refiner_result: Dict[str, Any],
        browser_result: Dict[str, Any],
        execution_summary: Dict[str, Any],
    ) -> str:
        goal = str(task_refiner_result.get("raw_user_prompt") or payload.get("user_prompt") or "").strip()
        plan = str(task_refiner_result.get("refined_user_prompt") or goal).strip()
        status = str(browser_result.get("status") or "unknown").strip()
        final_url = str(browser_result.get("final_url") or browser_result.get("start_url") or "").strip()
        browser_output = str(browser_result.get("output") or "").strip()
        activity = execution_summary.get("activity")
        errors = execution_summary.get("errors")
        network = execution_summary.get("network")
        downloads = execution_summary.get("downloads")
        artifacts = execution_summary.get("artifacts")

        lines: List[str] = ["## Goal", goal or "(unknown)"]
        lines.extend(["", "## Intended Plan", plan or "(unknown)"])
        lines.extend(["", "## Outcome"])
        if status == "success":
            lines.append(f"The browser run completed successfully{f' at `{final_url}`' if final_url else ''}.")
        elif status == "error":
            lines.append("The browser run did not complete successfully.")
        else:
            lines.append(f"The browser run ended with status `{status or 'unknown'}`.")

        if browser_output:
            lines.extend(["", "## Agent Result", self._truncate_text(browser_output, 1200)])

        if isinstance(activity, list) and activity:
            lines.extend(["", "## Actions Taken"])
            for item in activity[:8]:
                if not isinstance(item, dict):
                    continue
                tool = str(item.get("tool") or item.get("status") or "step")
                state = str(item.get("status") or "unknown")
                detail = str(item.get("error_message") or item.get("result") or "").strip()
                bullet = f"- `{tool}`: {state}"
                if detail:
                    bullet += f" - {self._truncate_text(detail, 220)}"
                lines.append(bullet)

        if isinstance(network, dict):
            counts = network.get("counts")
            failed_requests = network.get("failed_requests")
            console_errors = network.get("console_errors")
            if counts or failed_requests or console_errors:
                lines.extend(["", "## Network Evidence"])
                if isinstance(counts, dict) and counts:
                    lines.append(
                        "- Counts: "
                        f"requests={counts.get('requests', 0)}, "
                        f"failed_requests={counts.get('failed_requests', 0)}, "
                        f"console_errors={counts.get('console_errors', 0)}, "
                        f"websocket={counts.get('websocket', 0)}"
                    )
                if isinstance(failed_requests, list):
                    for item in failed_requests[:3]:
                        if not isinstance(item, dict):
                            continue
                        url = str(item.get("url") or "(unknown url)")
                        detail = str(item.get("error_text") or item.get("status_code") or "").strip()
                        lines.append(
                            f"- Failed request: `{self._truncate_text(url, 180)}`"
                            + (f" - {self._truncate_text(detail, 180)}" if detail else "")
                        )
                if isinstance(console_errors, list):
                    for item in console_errors[:3]:
                        if not isinstance(item, dict):
                            continue
                        detail = str(item.get("text") or "").strip()
                        if detail:
                            lines.append(f"- Console error: {self._truncate_text(detail, 220)}")

        if isinstance(downloads, dict) and downloads.get("count"):
            lines.extend(["", "## Downloads"])
            lines.append(f"- Detected downloads: {downloads.get('count')}")
            items = downloads.get("items")
            if isinstance(items, list):
                for item in items[:5]:
                    if not isinstance(item, dict):
                        continue
                    download_id = str(item.get("download_id") or "download")
                    status_text = str(item.get("status") or "unknown")
                    filename = str(item.get("filename") or item.get("suggested_filename") or "").strip()
                    saved_path = str(item.get("saved_path") or "").strip()
                    detail = filename or saved_path
                    lines.append(
                        f"- `{download_id}`: {status_text}"
                        + (f" - {self._truncate_text(detail, 220)}" if detail else "")
                    )

        if isinstance(artifacts, dict) and any(artifacts.values()):
            lines.extend(["", "## Artifacts"])
            for key in (
                "artifacts_dir",
                "downloads_dir",
                "screenshots_dir",
                "activity_db_path",
            ):
                value = str(artifacts.get(key) or "").strip()
                if value:
                    lines.append(f"- `{key}`: `{value}`")

        if isinstance(errors, list) and errors:
            lines.extend(["", "## Errors"])
            for item in errors[:5]:
                if not isinstance(item, dict):
                    continue
                source = str(item.get("tool") or item.get("source") or "error")
                message = str(item.get("error_message") or item.get("message") or "").strip()
                lines.append(f"- `{source}`: {self._truncate_text(message, 280) if message else 'Unknown error'}")

        return "\n".join(lines).strip()

    @staticmethod
    def _extract_json_obj(text: str) -> Dict[str, Any]:
        raw = str(text or "").strip()
        if not raw:
            return {}
        candidates = [raw]
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(raw[start : end + 1])
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
        return {}
