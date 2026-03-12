"""Browser automation runtime powered by the llm_core event loop."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from moose.framework.llm_core import LLMClient, create_llm_client_from_config

from .downloads import DownloadsFeature
from .models import RunConfig
from .network_inspector import NetworkInspectorFeature
from .ref_page_actions import PageActionsFeature
from .session import BrowserSessionManager


class BrowserAutomation:
    """Run one browser task using the event-driven LLM loop and browser tools."""

    def __init__(
        self,
        *,
        session_manager: BrowserSessionManager,
        page_actions: PageActionsFeature,
        network_inspector: Optional[NetworkInspectorFeature] = None,
        downloads: Optional[DownloadsFeature] = None,
        llm_settings: Optional[Dict[str, Any]] = None,
        logger: Any = None,
    ):
        self.session_manager = session_manager
        self.page_actions = page_actions
        self.network_inspector = network_inspector
        self.downloads = downloads
        self.llm_settings = llm_settings or {}
        self.logger = logger

    def _create_worker_llm_client(
        self, *, tools: List[Any], run_config: Optional[RunConfig] = None
    ) -> LLMClient:
        cfg = dict(self.llm_settings or {})
        cfg_kwargs = dict(cfg.get("kwargs") or {}) if isinstance(cfg.get("kwargs"), dict) else {}
        runtime_overrides: Dict[str, Any] = {}
        # So upload_image_sync uses the same timeout as browser (timeout_ms → seconds)
        if run_config is not None and "timeout" not in cfg and "timeout" not in cfg_kwargs:
            timeout_ms = getattr(run_config, "timeout_ms", None)
            if timeout_ms is not None and int(timeout_ms) > 0:
                runtime_overrides["timeout"] = int(timeout_ms) / 1000.0

        return create_llm_client_from_config(
            cfg,
            tools=tools,
            agent_name="playwright_agent",
            runtime_overrides=runtime_overrides,
        )

    def _build_task_message(self, run_config: RunConfig) -> str:
        user_prompt = str(run_config.user_prompt or "").strip()
        if not user_prompt:
            user_prompt = "Use the browser tools to complete the task."
        if run_config.start_url:
            return (
                f"Start URL: {run_config.start_url}\n"
                "The browser has already been opened for this URL. Continue the task using the browser tools.\n\n"
                f"User task:\n{user_prompt}"
            )
        return user_prompt

    def _bind_run_state(self, run_state: Any) -> None:
        self.page_actions.set_run_state(run_state)
        if self.network_inspector is not None:
            self.network_inspector.set_run_state(run_state)
        if self.downloads is not None:
            self.downloads.set_run_state(run_state)

    def _clear_run_state(self) -> None:
        try:
            self.page_actions.set_run_state(None)
        except Exception:
            pass
        if self.network_inspector is not None:
            try:
                self.network_inspector.set_run_state(None)
            except Exception:
                pass
        if self.downloads is not None:
            try:
                self.downloads.set_run_state(None)
            except Exception:
                pass

    async def _collect_network_summary(self, run_state: Any) -> Dict[str, Any]:
        inspector = self.network_inspector
        empty = {
            "ok": True,
            "counts": {
                "requests": 0,
                "failed_requests": 0,
                "console": 0,
                "console_errors": 0,
                "websocket": 0,
                "page_errors": 0,
            },
            "recent_requests": [],
            "failed_requests": [],
            "console_errors": [],
            "recent_websocket": [],
        }
        if inspector is None or run_state is None:
            return empty
        summarize_capture = getattr(inspector, "summarize_capture", None)
        if not callable(summarize_capture):
            return empty
        try:
            summary = await summarize_capture(run_state)
            return summary if isinstance(summary, dict) else empty
        except Exception as exc:
            payload = dict(empty)
            payload["ok"] = False
            payload["error"] = f"{type(exc).__name__}: {exc}"
            return payload

    async def _collect_downloads_payload(self, run_state: Any) -> Dict[str, Any]:
        if self.downloads is None or run_state is None:
            return {"ok": True, "downloads": []}
        try:
            payload = await self.downloads.list_downloads(run_state, limit=50)
            return payload if isinstance(payload, dict) else {"ok": True, "downloads": []}
        except Exception as exc:
            return {
                "ok": False,
                "downloads": [],
                "error": f"{type(exc).__name__}: {exc}",
            }

    async def _collect_browser_state(self, run_state: Any) -> Dict[str, Any]:
        if run_state is None:
            return {}
        try:
            payload = await self.page_actions.controller.describe_state(run_state)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _build_artifacts_payload(self, run_state: Any) -> Dict[str, Any]:
        metadata = getattr(run_state, "metadata", None) if run_state is not None else None
        metadata = metadata if isinstance(metadata, dict) else {}
        return {
            "artifacts_dir": metadata.get("artifacts_dir"),
            "downloads_dir": metadata.get("downloads_dir"),
            "screenshots_dir": metadata.get("screenshots_dir"),
            "user_data_dir": metadata.get("user_data_dir"),
            "activity_db_path": str(getattr(self.page_actions.event_logger, "db_path", "")),
        }

    def _sanitize_loop_payload(self, payload: Any) -> Any:
        """Remove provider-native raw responses from the Playwright return payload."""
        if isinstance(payload, dict):
            return {
                str(key): self._sanitize_loop_payload(value)
                for key, value in payload.items()
                if str(key) != "raw_response"
            }
        if isinstance(payload, list):
            return [self._sanitize_loop_payload(item) for item in payload]
        if isinstance(payload, tuple):
            return [self._sanitize_loop_payload(item) for item in payload]
        return payload

    async def run(
        self,
        *,
        run_config: RunConfig,
        tools: Optional[List[Any]] = None,
        include_loop_events: bool = True,
    ) -> Dict[str, Any]:
        """Execute one browser task and return a structured loop result."""

        resolved_prompt = self._build_task_message(run_config)
        run_state = None
        loop_payload: Optional[Dict[str, Any]] = None
        run_status = "error"
        run_error: Optional[str] = None
        browser_state: Dict[str, Any] = {}
        downloads_payload: Dict[str, Any] = {"ok": True, "downloads": []}
        network_payload: Dict[str, Any] = {
            "ok": True,
            "counts": {
                "requests": 0,
                "failed_requests": 0,
                "console": 0,
                "console_errors": 0,
                "websocket": 0,
                "page_errors": 0,
            },
            "recent_requests": [],
            "failed_requests": [],
            "console_errors": [],
            "recent_websocket": [],
        }

        try:
            run_state = await self.session_manager.start(run_config)
            self._bind_run_state(run_state)

            if self.page_actions and self.page_actions.event_logger:
                self.page_actions.event_logger.start()
                self.page_actions.event_logger.init_run(
                    run_id=str(run_state.run_id),
                    request_id=str(run_state.request_id),
                    agent_name="playwright_agent",
                )

            run_state.metadata["capture_network"] = bool(run_config.capture_network)
            run_state.metadata["capture_console"] = bool(run_config.capture_console)
            run_state.metadata["capture_response_bodies"] = bool(run_config.capture_response_bodies)

            if self.network_inspector is not None and (
                bool(run_config.capture_network) or bool(run_config.capture_console)
            ):
                await self.network_inspector.attach_listeners(run_state)
                await self.network_inspector.start_capture(run_state)

            if self.downloads is not None:
                await self.downloads.attach_listener(run_state)

            browser_tools = tools if isinstance(tools, list) and tools else self.page_actions.get_tools()
            llm = self._create_worker_llm_client(tools=browser_tools, run_config=run_config)
            loop_result = await llm.collect_agent_loop(
                message=resolved_prompt,
                system_message=run_config.system_prompt or None,
                raise_on_error=False,
            )
            loop_payload = self._sanitize_loop_payload(loop_result.to_dict())
            if not include_loop_events:
                loop_payload.pop("events", None)

            browser_state = await self._collect_browser_state(run_state)
            downloads_payload = await self._collect_downloads_payload(run_state)
            network_payload = await self._collect_network_summary(run_state)

            if not loop_result.ok or loop_result.final_response is None:
                raise RuntimeError(loop_result.error_message or "Agent loop completed without a final response")

            run_status = "success"
            return {
                "status": "success",
                "output": str(loop_result.final_response.content or ""),
                "model": getattr(loop_result.final_response, "model", None),
                "request_id": loop_result.request_id,
                "run_id": loop_result.run_id,
                "start_url": run_config.start_url,
                "final_url": browser_state.get("url"),
                "browser_state": browser_state,
                "snapshot": run_state.snapshot.to_dict() if getattr(run_state, "snapshot", None) else None,
                "downloads": downloads_payload.get("downloads", []),
                "network": network_payload,
                "loop": loop_payload,
                "usage": dict(loop_result.total_usage or {}),
                "cost": loop_result.total_cost,
                "artifacts": self._build_artifacts_payload(run_state),
            }
        except Exception as e:
            run_error = f"{type(e).__name__}: {e}"
            if self.logger:
                try:
                    self.logger.error("Browser automation error: %s", run_error)
                except Exception:
                    pass
            payload: Dict[str, Any] = {
                "status": "error",
                "error": run_error,
                "start_url": run_config.start_url,
            }
            if run_state is not None:
                browser_state = await self._collect_browser_state(run_state)
                downloads_payload = await self._collect_downloads_payload(run_state)
                network_payload = await self._collect_network_summary(run_state)
                payload["final_url"] = browser_state.get("url")
                payload["browser_state"] = browser_state
                payload["snapshot"] = (
                    run_state.snapshot.to_dict() if getattr(run_state, "snapshot", None) else None
                )
                payload["downloads"] = downloads_payload.get("downloads", [])
                payload["network"] = network_payload
                payload["artifacts"] = self._build_artifacts_payload(run_state)
            if loop_payload is not None:
                payload["loop"] = loop_payload
            return payload
        finally:
            if self.network_inspector is not None and run_state is not None:
                try:
                    await self.network_inspector.stop_capture(run_state)
                except Exception:
                    pass
                try:
                    await self.network_inspector.detach_listeners(run_state)
                except Exception:
                    pass
            if self.downloads is not None and run_state is not None:
                try:
                    await self.downloads.detach_listener(run_state)
                except Exception:
                    pass
            try:
                if self.page_actions and self.page_actions.event_logger and run_state is not None:
                    self.page_actions.event_logger.complete_run(
                        run_id=str(run_state.run_id),
                        status=run_status,
                        error=run_error,
                    )
                    self.page_actions.event_logger.close()
            except Exception:
                pass
            self._clear_run_state()
            await self.session_manager.shutdown(run_state)
