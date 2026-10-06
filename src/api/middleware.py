from __future__ import annotations

import time
import uuid

from fastapi import Request

from ..core.logging import get_logger

logger = get_logger(__name__)


async def request_context_middleware(request: Request, call_next):
    """Attach a request ID and emit duration metadata for API tracing."""
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    response.headers["x-response-time-ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
    logger.info("API request completed", extra={"request_id": request_id, "path": request.url.path})
    return response
