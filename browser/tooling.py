"""Shared MCP tool helpers for the Playwright agent browser runtime."""

from __future__ import annotations

from typing import Any, Callable, List, Optional


def mcp_tool(
    _func: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
    examples: Optional[List[str]] = None,
) -> Any:
    """Decorator to mark a method as an MCP-exposed tool."""

    def _decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        setattr(func, "_is_mcp_tool", True)
        setattr(func, "_mcp_name", name or func.__name__)
        setattr(func, "_mcp_examples", examples or [])
        return func

    if _func is None:
        return _decorate
    return _decorate(_func)
