"""Model utilities."""

from typing import Sequence
from langchain.schema import BaseMessage
from langchain.chat_models.base import BaseChatModel
from langchain.chat_models import ChatOpenAI

smart_model = ChatOpenAI(temperature=0, model_name="gpt-4", verbose=True)  # type: ignore
creative_model = ChatOpenAI(temperature=0.7, model_name="gpt-4", verbose=True)  # type: ignore
super_creative_model = ChatOpenAI(temperature=1.0, model_name="gpt-4", verbose=True)  # type: ignore
fast_model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", verbose=True)  # type: ignore
broad_model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k", verbose=True)  # type: ignore


def query_model(
    model: BaseChatModel,
    messages: Sequence[BaseMessage],
    color: int = 0,
    preamble: str | None = None,
) -> str:
    """Query an LLM chat model. `preamble` is printed before the result."""
    if preamble is not None:
        print(f"\033[1;34m{preamble}\033[0m")
    result = model(list(messages)).content
    print(f"\033[{color}m{result}\033[0m")
    return result
