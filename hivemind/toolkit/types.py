"""Common types used in Hivemind."""

from pathlib import Path
from typing import Callable, Protocol


class HivemindAgent(Protocol):
    """Interface for Hivemind agents."""

    @property
    def name(self) -> str:
        """Name of the agent."""
        raise NotImplementedError

    @property
    def work_dir(self) -> Path:
        """Working directory for the agent's files."""
        raise NotImplementedError

    def run(self, message: str) -> tuple[str, Callable[[str], str]]:
        """Run the agent with a message, and a way to continue the conversation. Rerunning this method starts a new conversation."""
        raise NotImplementedError
