from __future__ import annotations

import json
from typing import Any, Callable, Dict, List


class DispatcherAgentNode:
    """Dispatcher workflow node with planning/finalization steps."""

    def __init__(
        self,
        *,
        create_llm_client: Callable[[Dict[str, Any], List[Any]], Any],
        build_tools: Callable[[], List[Any]],
        system_prompt: str,
        dispatcher_cfg: Dict[str, Any],
        normalize_dispatch_plan: Callable[[Dict[str, Any]], Dict[str, Any]],
        resolve_content_mode: Callable[[str, str, Dict[str, Any]], str],
        max_dispatch_rounds: int,
    ) -> None:
        self._create_llm_client = create_llm_client
        self._build_tools = build_tools
        self._system_prompt = system_prompt
        self._dispatcher_cfg = dispatcher_cfg
        self._normalize_dispatch_plan = normalize_dispatch_plan
        self._resolve_content_mode = resolve_content_mode
        self._max_dispatch_rounds = max_dispatch_rounds

    async def run(self, state: Dict[str, Any], *, extract_json_obj: Callable[[str], Dict[str, Any]]) -> Dict[str, Any]:
        """Plan routing or produce final response based on dispatcher mode."""
        mode = str(state.get("dispatcher_mode") or "plan")
        if mode == "final":
            return await self._finalize(state)
        return await self._plan(state, extract_json_obj=extract_json_obj)

    async def _plan(self, state: Dict[str, Any], *, extract_json_obj: Callable[[str], Dict[str, Any]]) -> Dict[str, Any]:
        dispatch_round = int(state.get("dispatch_round", 0) or 0)
        max_rounds = int(state.get("max_dispatch_rounds", self._max_dispatch_rounds) or self._max_dispatch_rounds)
        if dispatch_round >= max_rounds:
            updated = dict(state)
            updated["dispatcher_mode"] = "final"
            return updated

        llm = self._create_llm_client(self._dispatcher_cfg, self._build_tools())
        task = self._build_planning_request(state)
        response = await llm.send_message(message=task, system_message=self._system_prompt)
        parsed = extract_json_obj(str(getattr(response, "content", "") or ""))
        decision = self._normalize_dispatch_plan(parsed)
        return self._apply_plan(state, decision)

    async def _finalize(self, state: Dict[str, Any]) -> Dict[str, Any]:
        llm = self._create_llm_client(self._dispatcher_cfg, [])
        payload = {
            "mode": "finalize",
            "target_url": state.get("target_url"),
            "user_prompt": state.get("user_prompt"),
            "subagent_outputs": state.get("subagent_outputs", {}),
            "merged_subagent_output": state.get("merged_subagent_output", {}),
            "instruction": "Produce concise final answer for the user based only on evidence summaries.",
        }
        response = await llm.send_message(message=json.dumps(payload, ensure_ascii=False), system_message=self._system_prompt)
        content = str(getattr(response, "content", "") or "").strip()
        updated = dict(state)
        updated["final_response"] = {
            "status": "success",
            "dispatcher_response": content,
            "target_url": state.get("target_url"),
            "dispatch_round": state.get("dispatch_round"),
            "subagent_outputs": state.get("subagent_outputs", {}),
        }
        return updated

    @staticmethod
    def route_from_dispatcher(state: Dict[str, Any]) -> Any:
        """Conditional router for dispatcher fan-out."""
        if state.get("final_response") is not None:
            return "__end__"
        agents = list(state.get("selected_agents") or [])
        execution_mode = str(state.get("execution_mode") or "parallel")
        if not agents:
            return "dispatcher"
        if execution_mode == "parallel" and len(agents) > 1:
            return agents
        return agents[0]

    @staticmethod
    def route_from_merge(_state: Dict[str, Any]) -> str:
        """Route merge output back to dispatcher for final response."""
        return "dispatcher"

    @staticmethod
    def merge_subagent_outputs(state: Dict[str, Any]) -> Dict[str, Any]:
        """Fan-in merge node for analyst outputs before dispatcher finalization."""
        expected = list(state.get("expected_agents") or [])
        outputs = dict(state.get("subagent_outputs") or {})
        if expected and not set(expected).issubset(set(outputs.keys())):
            return state
        merged_items = [outputs.get(agent) for agent in expected if outputs.get(agent) is not None]
        updated = dict(state)
        updated["merged_subagent_output"] = {"expected_agents": expected, "items": merged_items}
        updated["dispatcher_mode"] = "final"
        return updated

    def _build_planning_request(self, state: Dict[str, Any]) -> str:
        target_url = str(state.get("target_url") or "")
        user_prompt = str(state.get("user_prompt") or "")
        return (
            "Create a dispatch plan for the current browser investigation.\n\n"
            "# Target URL\n"
            f"- target_url: {target_url}\n"
            "# Instructions"
            f"{user_prompt}\n"
            "# Allowed Agents\n"
            "- allowed_agents: [network_analyst, runtime_analyst]\n\n"
            "# Output Requirement\n"
            "Return JSON only, matching the dispatcher planning contract in system prompt."
        )

    def _apply_plan(self, state: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
        updated = dict(state)
        updated["dispatch_round"] = int(state.get("dispatch_round", 0) or 0) + 1
        updated["dispatch_plan"] = decision
        if decision["finish"]:
            updated["final_response"] = {
                "status": "success",
                "dispatcher_response": decision.get("final_response") or "",
                "target_url": state.get("target_url"),
                "dispatch_round": updated["dispatch_round"],
                "subagent_outputs": state.get("subagent_outputs", {}),
            }
            return updated

        selected_agents = decision["selected_agents"]
        updated["selected_agents"] = selected_agents
        updated["expected_agents"] = selected_agents
        updated["subagent_tasks"] = decision.get("subagent_tasks", {})
        updated["execution_mode"] = decision.get("execution_mode", "parallel")
        updated["content_mode"] = self._resolve_content_mode(
            str(decision.get("content_mode") or "distilled"),
            str(state.get("user_prompt") or ""),
            updated["subagent_tasks"],
        )
        if not selected_agents:
            updated["dispatcher_mode"] = "final"
        return updated
