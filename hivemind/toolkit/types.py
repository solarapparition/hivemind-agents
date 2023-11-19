"""Common types used in Hivemind."""

from pathlib import Path
from typing import Callable, Protocol


class HivemindReply(Protocol):
    """Interface for Hivemind replies."""

    content: str
    """The content of the reply."""

    async def continue_conversation(self, message: str) -> str:
        """Continue the conversation with a message."""
        raise NotImplementedError


class HivemindAgent(Protocol):
    """Interface for Hivemind agents."""

    output_dir: Path

    @property
    def name(self) -> str:
        """Name of the agent."""
        raise NotImplementedError

    def run(self, message: str) -> HivemindReply:
        """Run the agent with a message, and a way to continue the conversation. Rerunning this method starts a new conversation."""
        raise NotImplementedError
