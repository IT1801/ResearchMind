"""Shared application foundation."""

from .config import Settings, get_settings
from .exceptions import RAGError

__all__ = ["RAGError", "Settings", "get_settings"]
