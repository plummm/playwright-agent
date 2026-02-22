"""Playwright session lifecycle scaffold."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

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

            # Extensions require persistent context.
            context = await pw.chromium.launch_persistent_context(
                str(user_data_dir),
                channel="chromium",
                headless=bool(run_config.headless),
                accept_downloads=bool(run_config.accept_downloads),
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
            )

        context.set_default_timeout(float(run_config.timeout_ms))
        context.set_default_navigation_timeout(float(run_config.timeout_ms))

        page = context.pages[0] if context.pages else await context.new_page()
        page.set_default_timeout(float(run_config.timeout_ms))
        page.set_default_navigation_timeout(float(run_config.timeout_ms))

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
        return state

    def get_active_page(self, run_state: RunState):
        if run_state is None:
            return None

        page = getattr(run_state, "page", None)
        if page is not None:
            try:
                if not page.is_closed():
                    return page
            except Exception:
                pass

        context = getattr(run_state, "browser_context", None)
        if context is None:
            return None

        try:
            for candidate in context.pages:
                if not candidate.is_closed():
                    run_state.page = candidate
                    return candidate
        except Exception:
            return None
        return None

    async def new_page(self, run_state: RunState):
        context = getattr(run_state, "browser_context", None)
        if context is None:
            raise RuntimeError("Browser context is not initialized")

        page = await context.new_page()
        timeout_ms = int((run_state.metadata or {}).get("timeout_ms", 30000))
        page.set_default_timeout(float(timeout_ms))
        page.set_default_navigation_timeout(float(timeout_ms))
        run_state.page = page
        run_state.current_url = page.url or run_state.current_url
        return page

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
