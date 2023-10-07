"""Agent for researching information about topics."""

from typing import Callable
from dataclasses import dataclass

from autogen import (
    AssistantAgent,
    UserProxyAgent,
)

from hivemind.toolkit.autogen import (
    is_termination_msg,
    ConfigDict,
    DEFAULT_CONFIG_LIST as config_list,
    DEFAULT_LLM_CONFIG as llm_config,
    continue_agent_conversation,
)
from hivemind.toolkit.research import research
from hivemind.config import BASE_WORK_DIR


@dataclass
class ResearchDaemon:
    """Daemon to research detailed info about a topic."""

    @property
    def name(self) -> str:
        """Name of the daemon."""
        return "research_daemon"

    @property
    def llm_config(self) -> ConfigDict:
        """Return the config for the LLM."""

        return {
            "functions": [
                {
                    "name": "research",
                    "description": "research about a given topic, returning the research material including reference links",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The topic to be researched about",
                            }
                        },
                        "required": ["query"],
                    },
                },
            ],
            "raise_on_ratelimit_or_timeout": None,
            "request_timeout": 600,
            "seed": 42,
            "config_list": config_list,
            "temperature": 0,
        }

    @property
    def work_dir(self) -> str:
        """Return the working directory for the daemon."""
        return str(BASE_WORK_DIR / "research_daemon")

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
            function_map={
                "research": research,
            },
        )
        assistant = AssistantAgent(
            name=f"{self.name}_assistant",
            llm_config=self.llm_config,
            system_message="Fulfill the user's search request using your research function.",
        )
        user_proxy.initiate_chat(
            assistant,
            message=message,
        )
        user_proxy.stop_reply_at_receive(assistant)

        return user_proxy.last_message()["content"], continue_agent_conversation(
            user_proxy, assistant
        )


def test() -> None:
    """Test the daemon."""
    daemon = ResearchDaemon()
    reply, continue_conversation = daemon.run(
        "Find information on the 'autogen' framework for llms."
    )
    print(reply)


if __name__ == "__main__":
    test()
