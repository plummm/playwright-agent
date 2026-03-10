"""CDP-backed semantic snapshot capture and ref resolution."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from .event_logger import BrowserEventLogger
from .models import RunState, SemanticRef, SemanticSnapshot
from .session import BrowserSessionManager

_CANDIDATE_SELECTOR = ",".join(
    [
        "a[href]",
        "button",
        "input:not([type='hidden'])",
        "textarea",
        "select",
        "option",
        "summary",
        "label[for]",
        "[role]",
        "[tabindex]",
        "[contenteditable='']",
        "[contenteditable='true']",
    ]
)

_SEMANTIC_INFO_JS = """
function() {
  const el = this;
  const style = window.getComputedStyle(el);
  const rect = el.getBoundingClientRect();
  const tag = (el.tagName || '').toLowerCase();
  const roleAttr = (el.getAttribute('role') || '').trim().toLowerCase();
  const labelledBy = (el.getAttribute('aria-labelledby') || '').trim();
  let labelledByText = '';
  if (labelledBy) {
    labelledByText = labelledBy
      .split(/\\s+/)
      .map((id) => {
        const node = document.getElementById(id);
        return node ? (node.innerText || node.textContent || '') : '';
      })
      .join(' ')
      .replace(/\\s+/g, ' ')
      .trim();
  }
  const text = (el.innerText || el.textContent || '').replace(/\\s+/g, ' ').trim();
  const placeholder = (el.getAttribute('placeholder') || '').trim();
  const ariaLabel = (el.getAttribute('aria-label') || '').trim();
  const title = (el.getAttribute('title') || '').trim();
  const alt = (el.getAttribute('alt') || '').trim();
  const href = typeof el.href === 'string' ? el.href : (el.getAttribute('href') || '');
  const tabIndex = typeof el.tabIndex === 'number' ? el.tabIndex : null;
  const disabled = !!(el.disabled || el.getAttribute('aria-disabled') === 'true');
  const checked = typeof el.checked === 'boolean' ? el.checked : null;
  const selected = typeof el.selected === 'boolean' ? el.selected : null;
  const expandedAttr = el.getAttribute('aria-expanded');
  const expanded = expandedAttr === 'true' ? true : expandedAttr === 'false' ? false : null;
  const inputType = tag === 'input' ? ((el.getAttribute('type') || 'text').trim().toLowerCase()) : '';
  const name = ariaLabel || labelledByText || alt || title || placeholder || (tag === 'input' ? (el.value || '') : '') || text;
  const interactiveRoles = new Set([
    'button', 'link', 'textbox', 'searchbox', 'checkbox', 'radio', 'switch',
    'menuitem', 'menuitemcheckbox', 'menuitemradio', 'option', 'tab', 'combobox'
  ]);
  const interactive = !!(
    href ||
    tag === 'button' ||
    tag === 'a' ||
    tag === 'input' ||
    tag === 'textarea' ||
    tag === 'select' ||
    tag === 'summary' ||
    tag === 'option' ||
    el.isContentEditable ||
    roleAttr && interactiveRoles.has(roleAttr) ||
    (tabIndex !== null && tabIndex >= 0) ||
    typeof el.onclick === 'function'
  );
  const visible = !!(
    rect.width > 0 &&
    rect.height > 0 &&
    style.visibility !== 'hidden' &&
    style.display !== 'none'
  );
  return {
    tag_name: tag,
    role: roleAttr,
    name,
    text,
    placeholder,
    href: href || null,
    input_type: inputType || null,
    value: 'value' in el ? String(el.value || '') : null,
    disabled,
    checked,
    selected,
    expanded,
    tab_index: tabIndex,
    interactive,
    visible,
    content_editable: !!el.isContentEditable,
    bbox: {
      x: rect.x,
      y: rect.y,
      width: rect.width,
      height: rect.height
    }
  };
}
""".strip()

_VISIBLE_TEXT_JS = """
() => {
  const nodes = Array.from(document.querySelectorAll('h1,h2,h3,h4,p,li,th,td,caption'));
  const seen = new Set();
  const out = [];
  for (const node of nodes) {
    const rect = node.getBoundingClientRect();
    const style = window.getComputedStyle(node);
    if (rect.width <= 0 || rect.height <= 0) continue;
    if (style.visibility === 'hidden' || style.display === 'none') continue;
    const text = (node.innerText || node.textContent || '').replace(/\\s+/g, ' ').trim();
    if (!text) continue;
    if (text.length < 3 || text.length > 240) continue;
    if (seen.has(text)) continue;
    seen.add(text);
    out.push(text);
    if (out.length >= 20) break;
  }
  return out;
}
""".strip()


class SemanticSnapshotManager:
    """Capture semantic browser snapshots and resolve refs back to DOM nodes."""

    def __init__(
        self,
        session_manager: BrowserSessionManager,
        event_logger: Optional[BrowserEventLogger] = None,
    ):
        self.session_manager = session_manager
        self.event_logger = event_logger
        self._run_state: Optional[RunState] = None

    def set_run_state(self, run_state: Optional[RunState]) -> None:
        self._run_state = run_state

    def _require_run_state(self) -> RunState:
        if self._run_state is None:
            raise RuntimeError("SemanticSnapshotManager run_state is not set")
        return self._run_state

    def _log_action(self, run_state: RunState, *, action: str, payload: Dict[str, Any]) -> None:
        if self.event_logger is None:
            return
        try:
            self.event_logger.log_action_event(
                {
                    "run_id": run_state.run_id,
                    "request_id": run_state.request_id,
                    "event_type": action,
                    "ts": time.time(),
                    "payload": payload,
                }
            )
        except Exception:
            pass

    def _extract_ax_value(self, raw: Any) -> str:
        if isinstance(raw, dict):
            value = raw.get("value")
            if isinstance(value, str):
                return value
            return self._extract_ax_value(value)
        if isinstance(raw, list):
            out = [self._extract_ax_value(item) for item in raw]
            return " ".join(part for part in out if part).strip()
        return str(raw or "").strip()

    def _extract_ax_node(self, ax_payload: Any) -> Dict[str, Any]:
        if not isinstance(ax_payload, dict):
            return {}
        for node in ax_payload.get("nodes", []) or []:
            if not isinstance(node, dict):
                continue
            ignored = node.get("ignored")
            if ignored is True:
                continue
            properties = node.get("properties") or []
            prop_map: Dict[str, Any] = {}
            if isinstance(properties, list):
                for item in properties:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("name") or "").strip()
                    if not name:
                        continue
                    prop_map[name] = self._extract_ax_value(item.get("value"))
            return {
                "role": self._extract_ax_value(node.get("role")),
                "name": self._extract_ax_value(node.get("name")),
                "description": self._extract_ax_value(node.get("description")),
                "properties": prop_map,
            }
        return {}

    def _normalize_ref_info(
        self,
        *,
        node_id: int,
        backend_node_id: Optional[int],
        object_id: Optional[str],
        dom_info: Dict[str, Any],
        ax_info: Dict[str, Any],
        page_id: str,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(dom_info, dict):
            return None
        visible = bool(dom_info.get("visible"))
        interactive = bool(dom_info.get("interactive"))
        if not visible or not interactive:
            return None

        role = str(ax_info.get("role") or dom_info.get("role") or "").strip().lower()
        if role == "generic":
            role = str(dom_info.get("role") or "").strip().lower()
        name = str(ax_info.get("name") or dom_info.get("name") or "").strip()
        text = str(dom_info.get("text") or "").strip()
        href = dom_info.get("href")
        tag_name = str(dom_info.get("tag_name") or "").strip().lower()
        if tag_name == "label" and not name and text:
            name = text
        if tag_name == "input" and str(dom_info.get("input_type") or "").lower() == "hidden":
            return None
        if not role:
            if href:
                role = "link"
            elif tag_name in {"input", "textarea"}:
                role = "textbox"
            elif tag_name == "select":
                role = "combobox"
            elif tag_name == "option":
                role = "option"
            elif tag_name == "button":
                role = "button"
            else:
                role = tag_name or "node"

        display_name = name or text or str(dom_info.get("placeholder") or "").strip() or tag_name or role
        description_bits = [role]
        if display_name:
            description_bits.append(json.dumps(display_name))
        if href:
            description_bits.append(f"href={href}")
        placeholder = str(dom_info.get("placeholder") or "").strip()
        if placeholder and placeholder != display_name:
            description_bits.append(f"placeholder={json.dumps(placeholder)}")
        value = str(dom_info.get("value") or "").strip()
        if value and value != display_name and len(value) <= 80:
            description_bits.append(f"value={json.dumps(value)}")
        description = " ".join(bit for bit in description_bits if bit).strip()

        return {
            "page_id": page_id,
            "node_id": int(node_id),
            "backend_node_id": int(backend_node_id) if backend_node_id is not None else None,
            "object_id": object_id,
            "role": role,
            "name": display_name,
            "tag_name": tag_name,
            "text": text,
            "href": href,
            "value": value or None,
            "placeholder": placeholder or None,
            "input_type": dom_info.get("input_type"),
            "disabled": bool(dom_info.get("disabled")),
            "checked": dom_info.get("checked"),
            "selected": dom_info.get("selected"),
            "expanded": dom_info.get("expanded"),
            "bbox": dom_info.get("bbox") if isinstance(dom_info.get("bbox"), dict) else {},
            "description": description,
            "extra": {
                "ax_description": ax_info.get("description"),
                "ax_properties": ax_info.get("properties"),
                "tab_index": dom_info.get("tab_index"),
                "content_editable": dom_info.get("content_editable"),
            },
        }

    async def _call_function_on_object(
        self,
        session: Any,
        *,
        object_id: str,
        function_declaration: str,
        arguments: Optional[List[Dict[str, Any]]] = None,
        return_by_value: bool = True,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "objectId": object_id,
            "functionDeclaration": function_declaration,
            "awaitPromise": True,
            "userGesture": True,
            "returnByValue": return_by_value,
        }
        if arguments:
            payload["arguments"] = arguments
        return await session.send("Runtime.callFunctionOn", payload)

    async def capture(
        self,
        run_state: Optional[RunState] = None,
        *,
        max_refs: int = 80,
        include_page_text: bool = True,
    ) -> Dict[str, Any]:
        current = run_state or self._require_run_state()
        page = self.session_manager.get_active_page(current)
        if page is None:
            page = await self.session_manager.new_page(current)
        self.session_manager.sync_pages(current, preferred_page=page)

        page_id = str(current.active_page_id or "")
        if not page_id:
            raise RuntimeError("Unable to resolve active tab for snapshot capture")

        session = await self.session_manager.get_cdp_session(current, page=page)
        document = await session.send("DOM.getDocument", {"depth": 1, "pierce": True})
        root_node = (document or {}).get("root") or {}
        root_node_id = root_node.get("nodeId")
        if root_node_id is None:
            raise RuntimeError("Failed to resolve DOM root for semantic snapshot")

        query = await session.send(
            "DOM.querySelectorAll",
            {
                "nodeId": int(root_node_id),
                "selector": _CANDIDATE_SELECTOR,
            },
        )
        node_ids = list((query or {}).get("nodeIds") or [])

        refs: Dict[str, SemanticRef] = {}
        lines: List[str] = []
        next_ref_index = 1

        for raw_node_id in node_ids:
            if len(refs) >= max_refs:
                break
            try:
                node_id = int(raw_node_id)
            except Exception:
                continue

            try:
                described = await session.send("DOM.describeNode", {"nodeId": node_id, "depth": 0})
            except Exception:
                continue
            node_info = (described or {}).get("node") or {}
            backend_node_id = node_info.get("backendNodeId")
            resolve_payload: Dict[str, Any]
            try:
                if backend_node_id is not None:
                    resolve_payload = await session.send(
                        "DOM.resolveNode",
                        {"backendNodeId": int(backend_node_id)},
                    )
                else:
                    resolve_payload = await session.send("DOM.resolveNode", {"nodeId": node_id})
            except Exception:
                continue
            object_id = (((resolve_payload or {}).get("object") or {}).get("objectId") or None)
            if not object_id:
                continue

            try:
                runtime_result = await self._call_function_on_object(
                    session,
                    object_id=object_id,
                    function_declaration=_SEMANTIC_INFO_JS,
                    arguments=None,
                    return_by_value=True,
                )
                dom_info = ((runtime_result or {}).get("result") or {}).get("value")
            except Exception:
                dom_info = None
            if not isinstance(dom_info, dict):
                continue

            try:
                ax_payload = await session.send(
                    "Accessibility.getPartialAXTree",
                    {
                        "backendNodeId": int(backend_node_id)
                        if backend_node_id is not None
                        else None,
                        "fetchRelatives": False,
                    },
                )
            except Exception:
                ax_payload = {}

            normalized = self._normalize_ref_info(
                node_id=node_id,
                backend_node_id=int(backend_node_id) if backend_node_id is not None else None,
                object_id=object_id,
                dom_info=dom_info,
                ax_info=self._extract_ax_node(ax_payload),
                page_id=page_id,
            )
            if normalized is None:
                continue

            ref_name = f"r{next_ref_index}"
            next_ref_index += 1
            semantic_ref = SemanticRef(ref=ref_name, **normalized)
            refs[ref_name] = semantic_ref
            lines.append(f"[{ref_name}] {semantic_ref.description}")

        page_text_items: List[str] = []
        if include_page_text:
            try:
                raw_page_text = await page.evaluate(_VISIBLE_TEXT_JS)
                if isinstance(raw_page_text, list):
                    page_text_items = [str(item).strip() for item in raw_page_text if str(item).strip()]
            except Exception:
                page_text_items = []

        title = ""
        try:
            title = str(await page.title() or "")
        except Exception:
            title = ""

        tabs = await self.session_manager.describe_tabs(current)
        rendered_parts = [
            f"URL: {str(getattr(page, 'url', '') or current.current_url or '')}",
            f"Title: {title or '(untitled)'}",
            f"Active tab: {page_id}",
            "Tabs:",
        ]
        rendered_parts.extend(
            f"- {tab['page_id']}{' [active]' if tab.get('is_active') else ''}: {tab.get('url') or '(empty)'}"
            for tab in tabs
        )
        if page_text_items:
            rendered_parts.append("Visible text:")
            rendered_parts.extend(f"- {item}" for item in page_text_items)
        rendered_parts.append("Interactive refs:")
        if lines:
            rendered_parts.extend(lines)
        else:
            rendered_parts.append("- No visible interactive refs found.")

        rendered = "\n".join(rendered_parts).strip()
        snapshot = SemanticSnapshot(
            snapshot_id=self.session_manager.next_snapshot_id(current),
            page_id=page_id,
            url=str(getattr(page, "url", "") or current.current_url or ""),
            title=title,
            created_at=time.time(),
            rendered=rendered,
            refs=refs,
            metadata={
                "ref_count": len(refs),
                "page_text": page_text_items,
                "tabs": tabs,
                "capture_truncated": len(node_ids) > len(refs),
            },
        )
        current.snapshot = snapshot
        current.current_url = snapshot.url

        public_refs = {
            ref_name: {
                "description": ref_obj.description,
                "role": ref_obj.role,
                "name": ref_obj.name,
                "tag_name": ref_obj.tag_name,
                "href": ref_obj.href,
                "page_id": ref_obj.page_id,
            }
            for ref_name, ref_obj in refs.items()
        }
        result = {
            "ok": True,
            "snapshot_id": snapshot.snapshot_id,
            "active_tab_id": page_id,
            "url": snapshot.url,
            "title": snapshot.title,
            "ref_count": len(refs),
            "snapshot_text": rendered,
            "refs": public_refs,
        }
        self._log_action(current, action="semantic_snapshot", payload=result)
        return result

    def require_snapshot(self, run_state: Optional[RunState] = None) -> SemanticSnapshot:
        current = run_state or self._require_run_state()
        snapshot = getattr(current, "snapshot", None)
        if snapshot is None:
            raise RuntimeError("No semantic snapshot is available. Call browser_snapshot first.")
        if str(snapshot.page_id or "") != str(current.active_page_id or ""):
            raise RuntimeError("Snapshot is stale because the active tab changed. Call browser_snapshot again.")
        return snapshot

    def get_ref(self, ref: str, run_state: Optional[RunState] = None) -> SemanticRef:
        snapshot = self.require_snapshot(run_state)
        ref_name = str(ref or "").strip()
        if not ref_name:
            raise ValueError("ref is required")
        resolved = snapshot.refs.get(ref_name)
        if resolved is None:
            raise ValueError(f"Unknown ref: {ref_name}")
        return resolved

    async def resolve_ref_object(
        self,
        ref: str,
        run_state: Optional[RunState] = None,
    ) -> tuple[RunState, Any, SemanticRef, str]:
        current = run_state or self._require_run_state()
        semantic_ref = self.get_ref(ref, current)
        if semantic_ref.page_id != str(current.active_page_id or ""):
            await self.session_manager.set_active_page(current, semantic_ref.page_id)

        session = await self.session_manager.get_cdp_session(current, page_id=semantic_ref.page_id)
        resolve_args: Dict[str, Any] = {}
        if semantic_ref.backend_node_id is not None:
            resolve_args["backendNodeId"] = int(semantic_ref.backend_node_id)
        elif semantic_ref.node_id is not None:
            resolve_args["nodeId"] = int(semantic_ref.node_id)
        else:
            raise RuntimeError(f"Ref {semantic_ref.ref} cannot be resolved back to a DOM node")

        resolved = await session.send("DOM.resolveNode", resolve_args)
        object_id = (((resolved or {}).get("object") or {}).get("objectId") or None)
        if not object_id:
            raise RuntimeError(f"Ref {semantic_ref.ref} is no longer attached to the page. Call browser_snapshot again.")

        semantic_ref.object_id = object_id
        return current, session, semantic_ref, object_id

    async def call_function_on_ref(
        self,
        ref: str,
        *,
        function_declaration: str,
        arguments: Optional[List[Dict[str, Any]]] = None,
        return_by_value: bool = True,
        run_state: Optional[RunState] = None,
    ) -> Any:
        current, session, semantic_ref, object_id = await self.resolve_ref_object(ref, run_state)
        try:
            if semantic_ref.backend_node_id is not None:
                await session.send(
                    "DOM.scrollIntoViewIfNeeded",
                    {"backendNodeId": int(semantic_ref.backend_node_id)},
                )
        except Exception:
            pass
        response = await self._call_function_on_object(
            session,
            object_id=object_id,
            function_declaration=function_declaration,
            arguments=arguments,
            return_by_value=return_by_value,
        )
        return ((response or {}).get("result") or {}).get("value")
