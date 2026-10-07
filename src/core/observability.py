from __future__ import annotations

import os
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from langsmith import traceable

from .config import get_settings

F = TypeVar("F", bound=Callable[..., Any])


def configure_observability() -> bool:
    """Configure LangSmith tracing from validated project settings."""
    settings = get_settings()
    if not settings.langchain_tracing_v2 or not settings.langchain_api_key:
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
    return True


def traced(name: str, run_type: str = "chain") -> Callable[[F], F]:
    """Add a LangSmith run span while keeping tracing optional."""
    return traceable(name=name, run_type=run_type)
