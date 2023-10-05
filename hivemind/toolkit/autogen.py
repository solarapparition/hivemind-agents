"""Toolkit for supporting Autogen package."""

from typing import Any
from autogen import config_list_from_json

ConfigDict = dict[str, Any]

DEFAULT_CONFIG_LIST: list[ConfigDict] = config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt4", "gpt-4-32k"],
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
