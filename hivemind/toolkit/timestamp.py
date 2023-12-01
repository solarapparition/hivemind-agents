"""Timestamp functionality."""

from datetime import datetime, timezone


def utc_timestamp() -> str:
    """Generate an id based on the current timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M-%S-%f")
