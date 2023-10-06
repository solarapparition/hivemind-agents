"""Agents for performing specific, simple tasks."""

from dataclasses import dataclass

from autogen import (
    AssistantAgent,
    UserProxyAgent,
)

from hivemind.config import BASE_WORK_DIR
from hivemind.toolkit.autogen import (
    is_termination_msg,
    DEFAULT_LLM_CONFIG as llm_config,
    continue_agent_conversation,
)

from typing import Callable


@dataclass
class WebDaemon:
    """Daemon to perform general tasks on the web."""

    @property
    def work_dir(self) -> str:
        """Return the working directory for the daemon."""
        return str(BASE_WORK_DIR / "web_daemon")

    @property
    def name(self) -> str:
        """Return the name of the daemon."""
        return "web_daemon"

    def run(
        self,
        message: str,
    ) -> tuple[str, Callable[[str], str]]:
        """Run the daemon."""

        user_proxy = UserProxyAgent(
            name=f"{self.name}_user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            is_termination_msg=is_termination_msg,
            code_execution_config={"work_dir": self.work_dir},
            llm_config=llm_config,
            system_message="Reply TERMINATE if the task has been solved at full satisfaction. Otherwise, reply CONTINUE, or the reason why the task is not solved yet.",
        )
        assistant = AssistantAgent(
            name=f"{self.name}_assistant",
            llm_config=llm_config,
        )
        user_proxy.initiate_chat(assistant, message=message)
        return user_proxy.last_message()["content"], continue_agent_conversation(
            user_proxy, assistant
        )


def test() -> None:
    """Test the daemon."""
    daemon = WebDaemon()
    reply, continue_conversation = daemon.run(
        "Retrieve the contents of this paper and save it to a file: https://arxiv.org/abs/2308.08155"
    )
    print(reply)


if __name__ == "__main__":
    test()
