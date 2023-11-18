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
    def output_dir(self) -> Path:
        """Directory for the agent's output files."""
        raise NotImplementedError

    def run(self, message: str) -> tuple[str, Callable[[str], str]]:
        """Run the agent with a message, and a way to continue the conversation. Rerunning this method starts a new conversation."""
        raise NotImplementedError


class HivemindReply(Protocol):
    """Interface for Hivemind replies."""

    content: str
    """The content of the reply."""

    def continue_conversation(self, message: str) -> str:
        """Continue the conversation with a message."""
        raise NotImplementedError
