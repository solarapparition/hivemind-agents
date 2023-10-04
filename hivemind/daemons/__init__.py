"""Agents for performing specific, simple tasks."""

from typing import Any
from dataclasses import dataclass

from autogen import (
    ConversableAgent,
    AssistantAgent,
    UserProxyAgent,
    config_list_from_json,
)

ConfigDict = dict[str, Any]

config_list: list[ConfigDict] = config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt4", "gpt-4-32k"],
    },
)

llm_config: ConfigDict = {
    "request_timeout": 600,
    "seed": 42,
    "config_list": config_list,
    "temperature": 0,
}


def is_termination_msg(message: dict[str, str]) -> bool:
    """Check if a message is a termination message."""
    return message.get("content", "").rstrip().endswith("TERMINATE")


WORK_DIR = ".data/shared_workspace/web_daemon"

@dataclass
class WebDaemon:
    """Daemon to retrieve info from the web."""

    def run(
        self,
        message: str,
        user_proxy: ConversableAgent | None = None,
        # override get_human_input in agent to change how to get human input
    ) -> None:
        """Run the daemon."""

        default_user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode="TERMINATE",
            max_consecutive_auto_reply=10,
            is_termination_msg=is_termination_msg,
            code_execution_config={"work_dir": WORK_DIR},
            llm_config=llm_config,
            system_message="Reply TERMINATE if the task has been solved at full satisfaction. Otherwise, reply CONTINUE, or the reason why the task is not solved yet.",
        )
        if user_proxy is None:
            user_proxy = default_user_proxy
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
