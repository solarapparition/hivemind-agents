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
)


@dataclass
class WebDaemon:
    """Daemon to perform general tasks on the web."""

    @property
    def work_dir(self) -> str:
        """Return the working directory for the daemon."""
        return str(BASE_WORK_DIR / "web_daemon")

    def run(
        self,
        message: str,
        # override get_human_input in agent to change how to get human input
    ) -> None:
        """Run the daemon."""

        user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode="TERMINATE",
            max_consecutive_auto_reply=10,
            is_termination_msg=is_termination_msg,
            code_execution_config={"work_dir": self.work_dir},
            llm_config=llm_config,
            system_message="Reply TERMINATE if the task has been solved at full satisfaction. Otherwise, reply CONTINUE, or the reason why the task is not solved yet.",
        )
        assistant = AssistantAgent(
            name="assistant",
            llm_config=llm_config,
        )
        user_proxy.initiate_chat(
            assistant,
            message=message,
        )


def test() -> None:
    """Test the daemon."""
    daemon = WebDaemon()
    daemon.run(
        "Retrieve the contents of this paper and save it to a file: https://arxiv.org/abs/2308.08155"
    )

if __name__ == "__main__":
    test()
