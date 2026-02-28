from __future__ import annotations

import json
from typing import Any, Callable, Dict, List


class NetworkAnalystNode:
    """Network analyst workflow node with deterministic aggregation."""

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
        task_entry = (state.get("subagent_tasks") or {}).get("network_analyst")
        if isinstance(task_entry, dict):
            task = str(task_entry.get("task") or state.get("user_prompt") or "")
            filters = task_entry.get("filter") if isinstance(task_entry.get("filter"), dict) else {}
        else:
            task = str(task_entry or state.get("user_prompt") or "")
            filters = {}
        records = await self._load_records(run_state=run_state, max_records=max_records, filters=filters)
        prepared_records = await self._prepare_records(
            run_state=run_state,
            records=records,
            content_mode=str(state.get("content_mode") or "distilled"),
        )
        per_request_results = await self._analyze_each_record(
            task=task,
            target_url=state.get("target_url"),
            prepared_records=prepared_records,
            extract_json_obj=extract_json_obj,
        )
        envelope_payload = self._aggregate(per_request_results)
        envelope_payload["task_id"] = str(state.get("dispatch_round") or "1")
        envelope = normalize_envelope("network_analyst", envelope_payload)
        updated = dict(state)
        outputs = dict(state.get("subagent_outputs") or {})
        outputs["network_analyst"] = envelope
        updated["subagent_outputs"] = outputs
        return updated

    async def _load_records(self, *, run_state: Any, max_records: int, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        records_page = {"items": []}
        if self._network_inspector is not None and run_state is not None:
            try:
                records_page = await self._network_inspector.list_requests(
                    run_state,
                    limit=max_records,
                    offset=0,
                    resource_type=str(filters.get("resource_type") or ""),
                    mime_type=str(filters.get("mime_type") or ""),
                    status_min=filters.get("status_min"),
                    status_max=filters.get("status_max"),
                    url_contains=str(filters.get("url_contains") or ""),
                    url_regex=str(filters.get("url_regex") or ""),
                )
            except Exception:
                pass
        return list(records_page.get("items") or [])

    async def _prepare_records(self, *, run_state: Any, records: List[Dict[str, Any]], content_mode: str) -> List[Dict[str, Any]]:
        prepared_records: List[Dict[str, Any]] = []
        for record in records:
            row = dict(record or {})
            row = await self._fill_missing_body(run_state=run_state, row=row)
            if content_mode != "full_body":
                row = self._truncate_record_body(row)
            prepared_records.append(row)
        return prepared_records

    async def _fill_missing_body(self, *, run_state: Any, row: Dict[str, Any]) -> Dict[str, Any]:
        response_body = row.get("response_body")
        if response_body is None and self._network_inspector is not None and run_state is not None:
            url = str(row.get("url") or "").strip()
            if url:
                try:
                    fallback = await self._network_inspector.fetch_url_content(run_state, url)
                except Exception:
                    fallback = {"ok": False}
                if fallback.get("ok"):
                    row["response_body"] = fallback.get("body")
                    row["response_body_kind"] = fallback.get("body_kind")
                    row["response_body_truncated"] = bool(fallback.get("body_truncated", False))
                    row["fallback_status_code"] = fallback.get("status_code")
        return row

    @staticmethod
    def _truncate_record_body(row: Dict[str, Any]) -> Dict[str, Any]:
        for key in ("response_body", "post_data"):
            val = row.get(key)
            if isinstance(val, str) and len(val) > 1200:
                row[key] = val[:1200] + "...<truncated-for-distilled-mode>"
        return row

    async def _analyze_each_record(
        self,
        *,
        task: str,
        target_url: Any,
        prepared_records: List[Dict[str, Any]],
        extract_json_obj: Callable[[str], Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        llm = self._create_llm_client(self._role_cfg, self._build_tools())
        per_request_results: List[Dict[str, Any]] = []
        for row in prepared_records:
            prompt = self._build_record_prompt(task=task, target_url=target_url, row=row)
            parsed_item = await self._call_record_llm(llm=llm, prompt=prompt, extract_json_obj=extract_json_obj)
            per_request_results.append(self._normalize_record_result(parsed_item=parsed_item, row=row))
        return per_request_results

    @staticmethod
    def _build_record_prompt(*, task: str, target_url: Any, row: Dict[str, Any]) -> Dict[str, Any]:
        flat_record = {
            "method": row.get("method"),
            "resource_type": row.get("resource_type"),
            "post_data": row.get("post_data"),
            "response_body": row.get("response_body"),
            "response_body_truncated": bool(row.get("response_body_truncated")),
            "error_text": row.get("error_text"),
            "is_download": bool(row.get("is_download")),
            "download_reason": row.get("download_reason"),
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
                "request_result": {
                    "status": "error",
                    "summary": f"LLM analysis failed: {type(e).__name__}: {e}",
                    "findings": [],
                    "evidence_refs": [],
                }
            }

    @staticmethod
    def _normalize_record_result(*, parsed_item: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
        result_obj = parsed_item.get("request_result", parsed_item)
        if not isinstance(result_obj, dict):
            result_obj = {"summary": str(result_obj)}
        return {
            "request_id": str(result_obj.get("request_id") or row.get("request_id") or ""),
            "url": str(result_obj.get("url") or row.get("url") or ""),
            "status": str(result_obj.get("status") or "ok"),
            "severity": str(result_obj.get("severity") or "info"),
            "summary": str(result_obj.get("summary") or ""),
            "findings": result_obj.get("findings") if isinstance(result_obj.get("findings"), list) else [],
            "evidence_refs": result_obj.get("evidence_refs") if isinstance(result_obj.get("evidence_refs"), list) else [],
        }

    @staticmethod
    def _aggregate(per_request_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        status_counts = {"error": 0, "warning": 0, "ok": 0}
        findings: List[str] = []
        evidence_refs: List[str] = []
        for item in per_request_results:
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
            f"Analyzed {len(per_request_results)} requests: "
            f"{status_counts['error']} error, {status_counts['warning']} warning, {status_counts['ok']} ok."
        )
        return {
            "status": overall_status,
            "summary": summary,
            "findings": findings[:20],
            "evidence_refs": evidence_refs[:50],
        }
