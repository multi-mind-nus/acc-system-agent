"""Small, allowlisted JSON events; never log document or model content."""

import json
import logging
from datetime import UTC, datetime
from time import perf_counter

logger = logging.getLogger("agent.requests")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())


def elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 2)


def event(name: str, **fields: str | int | float | None) -> None:
    logger.info(json.dumps({"timestamp": datetime.now(UTC).isoformat(), "event": name, **fields}))
