"""Toolkit for supporting Autogen package."""

from typing import Any, Callable
from autogen import config_list_from_json, ConversableAgent, UserProxyAgent, AssistantAgent

ConfigDict = dict[str, Any]

DEFAULT_CONFIG_LIST: list[ConfigDict] = config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt4", "gpt-4", "gpt-4-32k"],
    },
)

DEFAULT_LLM_CONFIG: ConfigDict = {
    "raise_on_ratelimit_or_timeout": None,
    "request_timeout": 600,
    "seed": 42,
    "config_list": DEFAULT_CONFIG_LIST,
    "temperature": 0,
}


def is_termination_msg(message: dict[str, str]) -> bool:
    """Check if a message is a termination message."""
    return bool(message.get("content")) and message.get(
        "content", ""
    ).rstrip().endswith("TERMINATE")

def get_last_user_reply(user_proxy: ConversableAgent, assistant: ConversableAgent) -> str:
    """Get assistant's last reply in a message thread. Assume that the last message is a "TERMINATE" message from the user to the assistant."""
    return user_proxy.chat_messages[assistant][-3]["content"]

def continue_agent_conversation(
    user_proxy: ConversableAgent, assistant: ConversableAgent
) -> Callable[[str], str]:
    """Construct function to continue a conversation with an agent."""

    def continue_conversation(message: str) -> str:
        """Continue the conversation with an agent."""
        user_proxy.initiate_chat(assistant, message=message, clear_history=False)
        return get_last_user_reply(user_proxy, assistant)
        # return user_proxy.last_message()["content"]

    return continue_conversation
