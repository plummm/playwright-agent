"""Shared models for PlaywrightAgent browser runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RunConfig:
    """Immutable run-level configuration."""

    run_id: str
    request_id: str
    start_url: Optional[str] = None
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    headless: bool = True
    extension_path: Optional[str] = None
    timeout_ms: int = 30000
    capture_network: bool = True
    capture_console: bool = True
    capture_response_bodies: bool = True
    accept_downloads: bool = True


@dataclass
class RunState:
    """Mutable runtime state for an active run."""

    run_id: str
    request_id: str
    active: bool = False
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    current_url: Optional[str] = None
    playwright: Any = None
    browser_context: Any = None
    page: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NetworkEvent:
    """Normalized network activity event."""

    run_id: str
    ts: float
    event_type: str
    method: Optional[str] = None
    url: Optional[str] = None
    resource_type: Optional[str] = None
    status_code: Optional[int] = None
    request_headers: Dict[str, str] = field(default_factory=dict)
    response_headers: Dict[str, str] = field(default_factory=dict)
    post_data: Optional[str] = None
    response_body_path: Optional[str] = None
    mime_type: Optional[str] = None
    page_url: Optional[str] = None
    frame_url: Optional[str] = None
    error_text: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsoleEvent:
    """Normalized browser console event."""

    run_id: str
    ts: float
    level: str
    text: str
    page_url: Optional[str] = None
    location: Dict[str, Any] = field(default_factory=dict)
    args_json: Optional[str] = None
