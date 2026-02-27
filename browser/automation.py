"""Browser automation runtime that executes one URL task with LLM + MCP tools."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from moose.framework.llm_core import LLMClient

from .downloads import DownloadsFeature
from .models import RunConfig
from .network_inspector import NetworkInspectorFeature
from .page_actions import PageActionsFeature
from .session import BrowserSessionManager


class BrowserAutomation:
    """Run one browser automation task using page-action MCP tools."""

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

    async def run(
        self,
        *,
        run_config: RunConfig,
        tools: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Execute one URL browser task and return structured result."""
        resolved_url = str(run_config.start_url or "").strip() or None
        if not resolved_url:
            raise ValueError("No URL available to browse")

        resolved_prompt = str(run_config.user_prompt or "").strip()
        resolved_system_prompt = str(run_config.system_prompt or "").strip()
        if not resolved_prompt:
            resolved_prompt = f"Open {resolved_url} and analyze the page."

        run_state = None
        run_status = "error"
        run_error: Optional[str] = None
        try:
            run_state = await self.session_manager.start(run_config)
            try:
                if self.page_actions and self.page_actions.event_logger:
                    self.page_actions.event_logger.start()
                    self.page_actions.event_logger.init_run(
                        run_id=str(run_state.run_id),
                        request_id=str(run_state.request_id),
                        agent_name="playwright_agent",
                    )
            except Exception:
                pass

            # Bind state for MCP tools.
            self.page_actions.set_run_state(run_state)
            if self.network_inspector is not None:
                try:
                    self.network_inspector.set_run_state(run_state)
                    # Propagate run-level capture flags so listener handlers can honor config.
                    run_state.metadata["capture_network"] = bool(run_config.capture_network)
                    run_state.metadata["capture_console"] = bool(run_config.capture_console)
                    run_state.metadata["capture_response_bodies"] = bool(run_config.capture_response_bodies)
                    if bool(run_config.capture_network) or bool(run_config.capture_console):
                        await self.network_inspector.attach_listeners(run_state)
                        await self.network_inspector.start_capture(run_state)
                except Exception as e:
                    if self.logger:
                        self.logger.warning("Failed to start network inspector: %s", e)

            browser_tools = tools if isinstance(tools, list) and tools else self.page_actions.get_tools()
            llm = self._create_worker_llm_client(tools=browser_tools)

            response = await llm.send_message(
                message=resolved_prompt,
                system_message=resolved_system_prompt or None,
            )

            final_page = self.session_manager.get_active_page(run_state)
            final_url = None
            if final_page is not None:
                try:
                    final_url = final_page.url
                except Exception:
                    final_url = None

            run_status = "success"
            return {
                "status": "success",
                "url": resolved_url,
                "final_url": final_url or run_state.current_url or resolved_url,
                "output": str(response.content or ""),
                "model": response.model,
                "usage": response.usage,
                "cost": response.cost,
                "request_id": response.request_id,
            }
        except Exception as e:
            run_error = f"{type(e).__name__}: {e}"
            if self.logger:
                try:
                    self.logger.error("Browser automation error for url=%s: %s", resolved_url, e)
                except Exception:
                    pass
            return {
                "status": "error",
                "url": resolved_url,
                "error": f"{type(e).__name__}: {e}",
            }
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
                try:
                    self.network_inspector.set_run_state(None)
                except Exception:
                    pass
            try:
                self.page_actions.set_run_state(None)
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
            await self.session_manager.shutdown(run_state)

    def _create_worker_llm_client(self, *, tools: List[Any]) -> LLMClient:
        """Create a fresh LLM client for one URL worker run."""
        cfg = self.llm_settings or {}
        model = str(cfg.get("model") or "gpt-5.2")
        temperature = float(cfg.get("temperature", 0.2))
        enable_web_search = bool(cfg.get("enable_web_search", False))
        kwargs = cfg.get("kwargs", {})
        if not isinstance(kwargs, dict):
            kwargs = {}

        return LLMClient(
            model=model,
            temperature=temperature,
            enable_web_search=enable_web_search,
            tools=tools,
            **kwargs,
        )
