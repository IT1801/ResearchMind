from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any


class TaskWorker:
    """Small async task boundary that can be replaced by ARQ or Celery."""

    def __init__(self) -> None:
        self.jobs: dict[str, Callable[..., Awaitable[Any]]] = {}

    def register(self, name: str, task: Callable[..., Awaitable[Any]]) -> None:
        self.jobs[name] = task

    async def run(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return await self.jobs[name](*args, **kwargs)


worker = TaskWorker()
