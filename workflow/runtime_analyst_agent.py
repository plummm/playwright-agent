from __future__ import annotations

import json
from typing import Any, Callable, Dict, List


class RuntimeAnalystNode:
    """Runtime analyst workflow node with deterministic aggregation."""

    def __init__(
        self,
        *,
        network_inspector: Any,
        create_llm_client: Callable[[Dict[str, Any], List[Any]], Any],
        build_tools: Callable[[], List[Any]],
        role_cfg: Dict[str, Any],
        system_prompt: str,
        max_records_per_subagent: int,
    ) -> None:
        self._network_inspector = network_inspector
        self._create_llm_client = create_llm_client
        self._build_tools = build_tools
        self._role_cfg = role_cfg
        self._system_prompt = system_prompt
        self._max_records_per_subagent = max_records_per_subagent

    async def run(
        self,
        state: Dict[str, Any],
        *,
        normalize_envelope: Callable[[str, Dict[str, Any]], Dict[str, Any]],
        extract_json_obj: Callable[[str], Dict[str, Any]],
    ) -> Dict[str, Any]:
        run_state = state.get("run_state")
        max_records = int(state.get("max_records_per_subagent", self._max_records_per_subagent) or 1)
        task_entry = (state.get("subagent_tasks") or {}).get("runtime_analyst")
        if isinstance(task_entry, dict):
            task = str(task_entry.get("task") or state.get("user_prompt") or "")
            filters = task_entry.get("filter") if isinstance(task_entry.get("filter"), dict) else {}
        else:
            task = str(task_entry or state.get("user_prompt") or "")
            filters = {}
        items = await self._load_console_records(run_state=run_state, max_records=max_records, filters=filters)
        per_record_results = await self._analyze_each_record(
            task=task,
            target_url=state.get("target_url"),
            items=items,
            extract_json_obj=extract_json_obj,
        )
        envelope_payload = self._aggregate(per_record_results)
        envelope_payload["task_id"] = str(state.get("dispatch_round") or "1")
        envelope = normalize_envelope("runtime_analyst", envelope_payload)
        updated = dict(state)
        outputs = dict(state.get("subagent_outputs") or {})
        outputs["runtime_analyst"] = envelope
        updated["subagent_outputs"] = outputs
        return updated

    async def _load_console_records(
        self, *, run_state: Any, max_records: int, filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        console_page = {"items": []}
        if self._network_inspector is not None and run_state is not None:
            try:
                console_page = await self._network_inspector.list_console(
                    run_state,
                    limit=max_records,
                    offset=0,
                    level=str(filters.get("level") or ""),
                    text_contains=str(filters.get("text_contains") or ""),
                )
            except Exception:
                pass
        return list(console_page.get("items") or [])

    async def _analyze_each_record(
        self,
        *,
        task: str,
        target_url: Any,
        items: List[Dict[str, Any]],
        extract_json_obj: Callable[[str], Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        llm = self._create_llm_client(self._role_cfg, self._build_tools())
        per_record_results: List[Dict[str, Any]] = []
        for entry in items:
            prompt = self._build_record_prompt(task=task, target_url=target_url, entry=entry)
            parsed_item = await self._call_record_llm(llm=llm, prompt=prompt, extract_json_obj=extract_json_obj)
            per_record_results.append(self._normalize_record_result(parsed_item))
        return per_record_results

    @staticmethod
    def _build_record_prompt(*, task: str, target_url: Any, entry: Dict[str, Any]) -> Dict[str, Any]:
        flat_record = {
            "level": entry.get("level"),
            "text": entry.get("text"),
            "location": entry.get("location"),
            "args": entry.get("args"),
        }
        return {
            "task": task,
            "target_url": target_url,
            "record": flat_record,
        }

    async def _call_record_llm(self, *, llm: Any, prompt: Dict[str, Any], extract_json_obj: Callable[[str], Dict[str, Any]]) -> Dict[str, Any]:
        try:
            per_resp = await llm.send_message(
                message=json.dumps(prompt, ensure_ascii=False),
                system_message=self._system_prompt,
            )
            return extract_json_obj(str(getattr(per_resp, "content", "") or ""))
        except Exception as e:
            return {
                "runtime_result": {
                    "status": "error",
                    "summary": f"LLM analysis failed: {type(e).__name__}: {e}",
                    "findings": [],
                    "evidence_refs": [],
                }
            }

    @staticmethod
    def _normalize_record_result(parsed_item: Dict[str, Any]) -> Dict[str, Any]:
        result_obj = parsed_item.get("runtime_result", parsed_item)
        if not isinstance(result_obj, dict):
            result_obj = {"summary": str(result_obj)}
        return {
            "status": str(result_obj.get("status") or "ok"),
            "severity": str(result_obj.get("severity") or "info"),
            "summary": str(result_obj.get("summary") or ""),
            "findings": result_obj.get("findings") if isinstance(result_obj.get("findings"), list) else [],
            "evidence_refs": result_obj.get("evidence_refs") if isinstance(result_obj.get("evidence_refs"), list) else [],
        }

    @staticmethod
    def _aggregate(per_record_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        status_counts = {"error": 0, "warning": 0, "ok": 0}
        findings: List[str] = []
        evidence_refs: List[str] = []
        for item in per_record_results:
            st = str(item.get("status") or "ok").lower()
            if st not in status_counts:
                st = "ok"
            status_counts[st] += 1
            for finding in item.get("findings", []):
                if isinstance(finding, str) and finding and finding not in findings:
                    findings.append(finding)
            for evidence in item.get("evidence_refs", []):
                if isinstance(evidence, str) and evidence and evidence not in evidence_refs:
                    evidence_refs.append(evidence)
        if status_counts["error"] > 0:
            overall_status = "error"
        elif status_counts["warning"] > 0:
            overall_status = "warning"
        else:
            overall_status = "ok"
        summary = (
            f"Analyzed {len(per_record_results)} runtime records: "
            f"{status_counts['error']} error, {status_counts['warning']} warning, {status_counts['ok']} ok."
        )
        return {
            "status": overall_status,
            "summary": summary,
            "findings": findings[:20],
            "evidence_refs": evidence_refs[:50],
        }
