"""Simple question-answering oracle for a single resource."""

from typing import Callable
from dataclasses import dataclass

from os import makedirs
from embedchain import App
from embedchain.config import LlmConfig, ChromaDbConfig
from autogen import (
    AssistantAgent,
    UserProxyAgent,
)

from hivemind.config import BASE_WORK_DIR
from hivemind.toolkit.autogen_support import (
    ConfigDict,
    DEFAULT_CONFIG_LIST as config_list,
    DEFAULT_LLM_CONFIG as llm_config,
    is_termination_msg,
    continue_agent_conversation,
    get_last_user_reply,
)
from hivemind.toolkit.resource_query import validate, query_resource


@dataclass
class QuestionAnswerOracle:
    """Daemon to research detailed info about a topic."""

    @property
    def name(self) -> str:
        """Name of the agent."""
        return "qna_oracle"

    @property
    def llm_config(self) -> ConfigDict:
        """Return the config for the LLM."""

        return {
            "functions": [
                {
                    "name": "query_resource",
                    "description": "Get the answer to a simple natural language question from a resource.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "resource_location": {
                                "type": "string",
                                "description": "The location (URI) of the resource.",
                            },
                            "query": {
                                "type": "string",
                                "description": "The question to ask the resource.",
                            },
                        },
                        "required": ["resource_location", "query"],
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
        return str(BASE_WORK_DIR / "qna_oracle")

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
            system_message="Reply TERMINATE after you receive an answer to the query.",
            function_map={
                "query_resource": query_resource,
            },
        )
        assistant = AssistantAgent(
            name=f"{self.name}_assistant",
            llm_config=self.llm_config,
            system_message="Convert the user's request to a query and resource location and use the `query_resource` function to get the answer.",
        )
        continue_conversation = continue_agent_conversation(user_proxy, assistant)
        if error := validate(message):
            return error, continue_conversation
        user_proxy.initiate_chat(
            assistant,
            message=message,
        )
        return get_last_user_reply(user_proxy, assistant), continue_conversation


def test() -> None:
    """Test the daemon."""
    daemon = QuestionAnswerOracle()
    reply, continue_conversation = daemon.run(
        "Tell me about the recent history of OpenAI using the page at https://en.wikipedia.org/wiki/OpenAI",
    )
    print(reply)


if __name__ == "__main__":
    test()
