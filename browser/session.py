"""Playwright session lifecycle helpers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import RunConfig, RunState

try:
    from playwright.async_api import async_playwright
except Exception:
    async_playwright = None  # type: ignore


class BrowserSessionManager:
    """Manage Playwright browser/context/page lifecycle."""

    def __init__(self, downloads_dir: str = "/tmp/playwright_agent"):
        self.downloads_dir = str(downloads_dir or "/tmp/playwright_agent")

    async def start(self, run_config: RunConfig) -> RunState:
        if async_playwright is None:
            raise ImportError(
                "Playwright is not available. Install with: pip install playwright "
                "and install browser binaries."
            )

        base_dir = Path(self.downloads_dir) / run_config.run_id
        downloads_path = base_dir / "downloads"
        screenshots_path = base_dir / "screenshots"
        user_data_dir = base_dir / "user_data"
        downloads_path.mkdir(parents=True, exist_ok=True)
        screenshots_path.mkdir(parents=True, exist_ok=True)

        pw = await async_playwright().start()
        launch_args = [
            "--no-sandbox",
            "--disable-dev-shm-usage",
        ]

        browser = None
        context = None

        if run_config.extension_path:
            ext_path = Path(run_config.extension_path)
            if not ext_path.exists():
                await pw.stop()
                raise FileNotFoundError(f"Extension path not found: {ext_path}")

            launch_args.extend(
                [
                    f"--disable-extensions-except={str(ext_path)}",
                    f"--load-extension={str(ext_path)}",
                ]
            )
            user_data_dir.mkdir(parents=True, exist_ok=True)
            context = await pw.chromium.launch_persistent_context(
                str(user_data_dir),
                channel="chromium",
                headless=bool(run_config.headless),
                accept_downloads=bool(run_config.accept_downloads),
                ignore_https_errors=bool(run_config.ignore_https_errors),
                downloads_path=str(downloads_path),
                args=launch_args,
            )
        else:
            browser = await pw.chromium.launch(
                headless=bool(run_config.headless),
                downloads_path=str(downloads_path),
                args=launch_args,
            )
            context = await browser.new_context(
                accept_downloads=bool(run_config.accept_downloads),
                ignore_https_errors=bool(run_config.ignore_https_errors),
            )

        context.set_default_timeout(float(run_config.timeout_ms))
        context.set_default_navigation_timeout(float(run_config.timeout_ms))

        page = context.pages[0] if context.pages else await context.new_page()
        self._apply_page_timeouts(page, timeout_ms=run_config.timeout_ms)
        if run_config.start_url:
            await page.goto(run_config.start_url, wait_until="domcontentloaded")

        state = RunState(
            run_id=run_config.run_id,
            request_id=run_config.request_id,
            active=True,
            started_at=time.time(),
            current_url=page.url or run_config.start_url,
            playwright=pw,
            browser_context=context,
            page=page,
            metadata={
                "browser": browser,
                "downloads_dir": str(downloads_path),
                "screenshots_dir": str(screenshots_path),
                "artifacts_dir": str(base_dir),
                "user_data_dir": str(user_data_dir),
                "timeout_ms": int(run_config.timeout_ms),
                "persistent_context": bool(run_config.extension_path),
                "headless": bool(run_config.headless),
            },
        )
        self._ensure_runtime_storage(state)
        self.sync_pages(state, preferred_page=page)
        return state

    def _ensure_runtime_storage(self, run_state: RunState) -> Dict[str, Any]:
        metadata = run_state.metadata if isinstance(run_state.metadata, dict) else {}
        run_state.metadata = metadata
        metadata.setdefault("pages", {})
        metadata.setdefault("page_index", {})
        metadata.setdefault("page_counter", 0)
        metadata.setdefault("downloads", {})
        metadata.setdefault("download_counter", 0)
        metadata.setdefault("snapshot_counter", 0)
        metadata.setdefault("snapshot_invalidated_reason", "")
        return metadata

    def _page_key(self, page: Any) -> str:
        return str(id(page))

    def _apply_page_timeouts(self, page: Any, *, timeout_ms: int) -> None:
        try:
            page.set_default_timeout(float(timeout_ms))
            page.set_default_navigation_timeout(float(timeout_ms))
        except Exception:
            pass

    def sync_pages(self, run_state: RunState, preferred_page: Optional[Any] = None) -> List[str]:
        metadata = self._ensure_runtime_storage(run_state)
        context = getattr(run_state, "browser_context", None)
        if context is None:
            return []

        pages_by_id = metadata["pages"]
        page_index = metadata["page_index"]
        page_counter = int(metadata.get("page_counter", 0) or 0)
        open_keys: set[str] = set()
        new_page_ids: List[str] = []
        timeout_ms = int(metadata.get("timeout_ms", 30000) or 30000)

        try:
            context_pages = list(context.pages or [])
        except Exception:
            context_pages = []

        for page in context_pages:
            key = self._page_key(page)
            open_keys.add(key)
            page_id = page_index.get(key)
            if page_id is None:
                page_counter += 1
                page_id = f"tab_{page_counter}"
                page_index[key] = page_id
                pages_by_id[page_id] = {
                    "page": page,
                    "page_key": key,
                    "page_id": page_id,
                    "cdp_session": None,
                    "created_at": time.time(),
                    "opener_page_id": run_state.active_page_id,
                    "last_url": str(getattr(page, "url", "") or ""),
                    "last_title": "",
                    "is_closed": False,
                }
                new_page_ids.append(page_id)
            record = pages_by_id.get(page_id) or {}
            record["page"] = page
            record["page_key"] = key
            record["page_id"] = page_id
            record["is_closed"] = False
            record["last_url"] = str(getattr(page, "url", "") or record.get("last_url") or "")
            pages_by_id[page_id] = record
            self._apply_page_timeouts(page, timeout_ms=timeout_ms)

        for page_id, record in list(pages_by_id.items()):
            key = str(record.get("page_key") or "")
            page = record.get("page")
            is_closed = key not in open_keys
            if page is not None and not is_closed:
                try:
                    is_closed = bool(page.is_closed())
                except Exception:
                    is_closed = False
            record["is_closed"] = is_closed
            pages_by_id[page_id] = record

        metadata["page_counter"] = page_counter

        target_page = preferred_page
        if target_page is None and run_state.active_page_id:
            record = pages_by_id.get(run_state.active_page_id) or {}
            page = record.get("page")
            if page is not None and not bool(record.get("is_closed")):
                target_page = page
        if target_page is None:
            for record in pages_by_id.values():
                page = record.get("page")
                if page is not None and not bool(record.get("is_closed")):
                    target_page = page
                    break

        if target_page is not None:
            active_key = self._page_key(target_page)
            run_state.active_page_id = page_index.get(active_key) or run_state.active_page_id
            run_state.page = target_page
            run_state.current_url = str(getattr(target_page, "url", "") or run_state.current_url or "")
        else:
            run_state.page = None
            run_state.active_page_id = None

        return new_page_ids

    def get_page_record(self, run_state: RunState, page_id: str) -> Optional[Dict[str, Any]]:
        metadata = self._ensure_runtime_storage(run_state)
        record = metadata["pages"].get(str(page_id))
        return record if isinstance(record, dict) else None

    def get_page_by_id(self, run_state: RunState, page_id: str) -> Any:
        record = self.get_page_record(run_state, page_id)
        if record is None or bool(record.get("is_closed")):
            return None
        return record.get("page")

    def get_active_page(self, run_state: Optional[RunState]):
        if run_state is None:
            return None
        self.sync_pages(run_state)
        if run_state.active_page_id:
            record = self.get_page_record(run_state, run_state.active_page_id)
            page = (record or {}).get("page")
            if page is not None and not bool((record or {}).get("is_closed")):
                return page
        return getattr(run_state, "page", None)

    async def new_page(self, run_state: RunState, *, make_active: bool = True):
        context = getattr(run_state, "browser_context", None)
        if context is None:
            raise RuntimeError("Browser context is not initialized")
        page = await context.new_page()
        self._apply_page_timeouts(page, timeout_ms=int((run_state.metadata or {}).get("timeout_ms", 30000)))
        self.sync_pages(run_state, preferred_page=page if make_active else None)
        if make_active:
            run_state.page = page
        self.invalidate_snapshot(run_state, reason="new_page")
        return page

    async def set_active_page(self, run_state: RunState, page_id: str):
        page = self.get_page_by_id(run_state, page_id)
        if page is None:
            raise ValueError(f"Unknown or closed tab: {page_id}")
        run_state.active_page_id = str(page_id)
        run_state.page = page
        run_state.current_url = str(getattr(page, "url", "") or run_state.current_url or "")
        self.invalidate_snapshot(run_state, reason="tab_changed")
        return page

    async def close_page(self, run_state: RunState, page_id: str) -> Dict[str, Any]:
        record = self.get_page_record(run_state, page_id)
        page = (record or {}).get("page")
        if page is None or bool((record or {}).get("is_closed")):
            raise ValueError(f"Unknown or closed tab: {page_id}")
        try:
            await page.close()
        finally:
            self.sync_pages(run_state)
            if run_state.active_page_id == page_id:
                self.invalidate_snapshot(run_state, reason="tab_closed")
        return {"ok": True, "page_id": str(page_id)}

    async def describe_tabs(self, run_state: RunState) -> List[Dict[str, Any]]:
        self.sync_pages(run_state)
        metadata = self._ensure_runtime_storage(run_state)
        tabs: List[Dict[str, Any]] = []
        for page_id, record in (metadata.get("pages") or {}).items():
            page = record.get("page")
            title = str(record.get("last_title") or "")
            if page is not None and not bool(record.get("is_closed")):
                try:
                    title = str(await page.title() or "")
                except Exception:
                    title = str(record.get("last_title") or "")
            record["last_title"] = title
            tabs.append(
                {
                    "page_id": str(page_id),
                    "url": str(record.get("last_url") or ""),
                    "title": title,
                    "is_active": str(page_id) == str(run_state.active_page_id or ""),
                    "is_closed": bool(record.get("is_closed")),
                    "opener_page_id": record.get("opener_page_id"),
                }
            )
        tabs.sort(key=lambda item: item["page_id"])
        return tabs

    async def get_cdp_session(
        self,
        run_state: RunState,
        *,
        page: Optional[Any] = None,
        page_id: Optional[str] = None,
    ) -> Any:
        target_page = page
        if target_page is None:
            if page_id:
                target_page = self.get_page_by_id(run_state, page_id)
            else:
                target_page = self.get_active_page(run_state)
        if target_page is None:
            raise RuntimeError("No active page is available")

        self.sync_pages(run_state, preferred_page=target_page)
        record = self.get_page_record(run_state, run_state.active_page_id or "")
        if record is None:
            raise RuntimeError("Unable to resolve page record for active tab")

        cached = record.get("cdp_session")
        if cached is not None:
            return cached

        context = getattr(run_state, "browser_context", None)
        if context is None:
            raise RuntimeError("Browser context is not initialized")

        session = await context.new_cdp_session(target_page)
        for command in ("DOM.enable", "Accessibility.enable", "Runtime.enable", "Page.enable"):
            try:
                await session.send(command)
            except Exception:
                pass
        record["cdp_session"] = session
        return session

    def next_snapshot_id(self, run_state: RunState) -> str:
        metadata = self._ensure_runtime_storage(run_state)
        counter = int(metadata.get("snapshot_counter", 0) or 0) + 1
        metadata["snapshot_counter"] = counter
        return f"snapshot_{counter}"

    def invalidate_snapshot(self, run_state: RunState, *, reason: str = "") -> None:
        run_state.snapshot = None
        metadata = self._ensure_runtime_storage(run_state)
        metadata["snapshot_invalidated_reason"] = str(reason or "")

    def next_download_id(self, run_state: RunState) -> str:
        metadata = self._ensure_runtime_storage(run_state)
        counter = int(metadata.get("download_counter", 0) or 0) + 1
        metadata["download_counter"] = counter
        return f"download_{counter}"

    async def shutdown(self, run_state: Optional[RunState]) -> None:
        if run_state is None:
            return

        try:
            context = getattr(run_state, "browser_context", None)
            if context is not None:
                await context.close()
        except Exception:
            pass

        try:
            browser = (run_state.metadata or {}).get("browser")
            if browser is not None:
                await browser.close()
        except Exception:
            pass

        try:
            pw = getattr(run_state, "playwright", None)
            if pw is not None:
                await pw.stop()
        except Exception:
            pass

        run_state.active = False
        run_state.ended_at = time.time()
