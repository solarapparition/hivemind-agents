"""Text formatting utilities."""

from textwrap import dedent

def dedent_and_strip(text: str) -> str:
    """Dedent and strip text."""
    return dedent(text).strip()
