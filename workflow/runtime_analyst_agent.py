from __future__ import annotations

import asyncio
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
        self._max_parallel_llm_calls = self._safe_int(role_cfg.get("max_parallel_llm_calls"), default=10, min_value=1)
        self._llm_retry_max_attempts = self._safe_int(role_cfg.get("llm_retry_max_attempts"), default=3, min_value=1)
        self._llm_retry_base_delay_seconds = self._safe_float(
            role_cfg.get("llm_retry_base_delay_seconds"), default=0.8, min_value=0.05
        )
        self._llm_retry_max_delay_seconds = self._safe_float(
            role_cfg.get("llm_retry_max_delay_seconds"), default=8.0, min_value=0.1
        )

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
        analysis_result = await self._analyze_each_record(
            task=task,
            target_url=state.get("target_url"),
            items=items,
            extract_json_obj=extract_json_obj,
        )
        per_record_results = analysis_result.get("items", [])
        llm_metrics = analysis_result.get("llm_metrics", {})
        envelope_payload = self._aggregate(per_record_results)
        envelope_payload["task_id"] = str(state.get("dispatch_round") or "1")
        envelope = normalize_envelope("runtime_analyst", envelope_payload)
        # Return partial state update so parallel branches can merge via reducer.
        return {
            "subagent_outputs": {"runtime_analyst": envelope},
            "llm_metrics": llm_metrics or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "total_cost": 0.0},
        }

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
    ) -> Dict[str, Any]:
        if not items:
            return {
                "items": [],
                "llm_metrics": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "total_cost": 0.0},
            }
        llm = self._create_llm_client(self._role_cfg, self._build_tools())
        semaphore = asyncio.Semaphore(self._max_parallel_llm_calls)

        async def _analyze_one(index: int, entry: Dict[str, Any]) -> tuple[int, Dict[str, Any], Dict[str, Any]]:
            async with semaphore:
                prompt = self._build_record_prompt(task=task, target_url=target_url, entry=entry)
                parsed_item, metrics = await self._call_record_llm(
                    llm=llm, prompt=prompt, extract_json_obj=extract_json_obj
                )
                return index, self._normalize_record_result(parsed_item), metrics

        tasks = [asyncio.create_task(_analyze_one(idx, entry)) for idx, entry in enumerate(items)]
        indexed_results = await asyncio.gather(*tasks)
        indexed_results.sort(key=lambda x: x[0])
        out_items: List[Dict[str, Any]] = []
        totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "total_cost": 0.0}
        for _, item, metrics in indexed_results:
            out_items.append(item)
            totals = self._merge_metrics(totals, metrics)
        return {"items": out_items, "llm_metrics": totals}

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

    async def _call_record_llm(
        self, *, llm: Any, prompt: Dict[str, Any], extract_json_obj: Callable[[str], Dict[str, Any]]
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        last_error: Exception | None = None
        for attempt in range(1, self._llm_retry_max_attempts + 1):
            try:
                per_resp = await llm.send_message(
                    message=json.dumps(prompt, ensure_ascii=False),
                    system_message=self._system_prompt,
                )
                parsed = extract_json_obj(str(getattr(per_resp, "content", "") or ""))
                return parsed, self._response_metrics(per_resp)
            except Exception as e:
                last_error = e
                if attempt >= self._llm_retry_max_attempts or not self._is_retryable_llm_error(e):
                    break
                await asyncio.sleep(self._backoff_delay_seconds(attempt))

        return (
            {
                "runtime_result": {
                    "status": "error",
                    "summary": f"LLM analysis failed: {type(last_error).__name__}: {last_error}",
                    "findings": [],
                    "evidence_refs": [],
                }
            },
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "total_cost": 0.0},
        )

    @staticmethod
    def _safe_int(value: Any, *, default: int, min_value: int) -> int:
        try:
            parsed = int(value)
        except Exception:
            parsed = default
        return max(min_value, parsed)

    @staticmethod
    def _safe_float(value: Any, *, default: float, min_value: float) -> float:
        try:
            parsed = float(value)
        except Exception:
            parsed = default
        return max(min_value, parsed)

    def _backoff_delay_seconds(self, attempt: int) -> float:
        exp_delay = self._llm_retry_base_delay_seconds * (2 ** max(0, attempt - 1))
        return min(self._llm_retry_max_delay_seconds, exp_delay)

    @staticmethod
    def _is_retryable_llm_error(error: Exception) -> bool:
        text = str(error or "").lower()
        retry_markers = (
            "429",
            "rate limit",
            "too many requests",
            "throttl",
            "temporarily unavailable",
            "service unavailable",
            "timeout",
            "timed out",
            "connection reset",
        )
        return any(marker in text for marker in retry_markers)

    @staticmethod
    def _response_metrics(response: Any) -> Dict[str, Any]:
        usage = getattr(response, "usage", {}) or {}
        return {
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
            "total_cost": float(getattr(response, "cost", 0.0) or 0.0),
        }

    @staticmethod
    def _merge_metrics(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(left or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "total_cost": 0.0})
        out["input_tokens"] = int(out.get("input_tokens", 0) or 0) + int(right.get("input_tokens", 0) or 0)
        out["output_tokens"] = int(out.get("output_tokens", 0) or 0) + int(right.get("output_tokens", 0) or 0)
        out["total_tokens"] = int(out.get("total_tokens", 0) or 0) + int(right.get("total_tokens", 0) or 0)
        out["total_cost"] = float(out.get("total_cost", 0.0) or 0.0) + float(right.get("total_cost", 0.0) or 0.0)
        return out

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
        finding_evidence_pairs: List[Dict[str, Any]] = []
        for item in per_record_results:
            st = str(item.get("status") or "ok").lower()
            if st not in status_counts:
                st = "ok"
            status_counts[st] += 1
            item_findings = item.get("findings", []) if isinstance(item.get("findings"), list) else []
            item_evidence_refs = item.get("evidence_refs", []) if isinstance(item.get("evidence_refs"), list) else []
            normalized_item_evidence_refs = [
                str(e).strip() for e in item_evidence_refs if str(e).strip()
            ]
            for finding in item_findings:
                if isinstance(finding, str) and finding:
                    finding_evidence_pairs.append(
                        {
                            "finding": finding,
                            "evidence_refs": list(normalized_item_evidence_refs),
                        }
                    )
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
            "finding_evidence_pairs": finding_evidence_pairs[:200],
        }
