"""Agent that can browse a webpage and communicate its state."""

from typing import Callable, Any, Protocol, Sequence
from functools import cached_property
from contextlib import suppress
from pathlib import Path
from dataclasses import dataclass

import langchain
from autogen import UserProxyAgent, AssistantAgent
from langchain.schema import SystemMessage, HumanMessage, BaseMessage, AIMessage
from langchain.chat_models import ChatAnthropic
from langchain.cache import SQLiteCache
from browserpilot.agents.gpt_selenium_agent import GPTSeleniumAgent

from hivemind.config import (
    BASE_WORK_DIR,
    LANGCHAIN_CACHE_DIR,
    TEST_DIR,
    BROWSERPILOT_DATA_DIR,
    CHROMEDRIVER_LOCATION,
)
from hivemind.toolkit.models import query_model, exact_model
from hivemind.toolkit.text_formatting import dedent_and_strip, extract_blocks
from hivemind.toolkit.autogen_support import get_last_reply
from hivemind.toolkit.browserpilot_support import run_with_instructions

langchain.llm_cache = SQLiteCache(
    database_path=str(LANGCHAIN_CACHE_DIR / ".langchain.db")
)

from hivemind.toolkit.autogen_support import (
    is_termination_msg,
    ConfigDict,
    DEFAULT_CONFIG_LIST,
    DEFAULT_LLM_CONFIG as llm_config,
    continue_agent_conversation,
)


class HivemindAgent(Protocol):
    """Interface for Hivemind agents."""

    @property
    def name(self) -> str:
        """Name of the agent."""
        ...

    @property
    def work_dir(self) -> Path:
        """Working directory for the agent's files."""
        ...

    def run(self, message: str) -> tuple[str, Callable[[str], str]]:
        """Run the agent with a message, and a way to continue the conversation. Rerunning this method starts a new conversation."""
        ...


def make_hivemind_user_proxy(
    agent: HivemindAgent,
    function_map: dict[str, Callable[[Any], Any]],
    llm_config: ConfigDict,
) -> UserProxyAgent:
    """Make a user proxy agent."""
    return UserProxyAgent(
        name=f"{agent.name}_user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=is_termination_msg,
        code_execution_config={"work_dir": agent.work_dir},
        llm_config=llm_config,
        system_message="Reply TERMINATE if the task has been solved or cannot be solved by the assistant.",
        function_map=function_map,
    )


def find_llm_validation_error(validation_messages: Sequence[BaseMessage]) -> str | None:
    """Run validation using an LLM. A specific output format is expected:
    If the message has no errors:
    ```text
    N/A
    ```

    If the message has errors:
    ```text
    <error>
    ```
    Where <error> contains the string "Error:".
    """
    result = query_model(exact_model, validation_messages, printout=False)
    error = extract_blocks(result, "text")
    if not error or ("N/A" not in error[-1] and "Error:" not in error[-1]):
        raise ValueError(f"Unable to extract error validation result:\n{error}")
    error_text = error[-1].strip()
    return error_text if "Error:" in error_text else None

# def agent_go_to_url(browserpilot_agent: GPTSeleniumAgent, url: str) -> None:
#     """Go to a URL."""
#     instructions = dedent_and_strip(
#         f"""
#         Go to the URL `{url}`.
#         """
#     )
#     run_with_instructions(browserpilot_agent, instructions)

# def go_to_url(url: str) -> None:
#     """Go to a URL."""
#     print("BLAH")

@dataclass
class BrowserDaemon:
    """Agent for browsing and reading webpages instead of scraping them. Can perform basic actions like clicking on elements and typing in textareas from natural language."""

    @property
    def name(self) -> str:
        """Name of the agent."""
        return "browser_daemon"

    @property
    def page_id(self) -> str:
        """ID of the page."""
        raise NotImplementedError

    @cached_property
    def browserpilot_agent(self) -> GPTSeleniumAgent:
        """Browserpilot instance for interacting with the driver."""
        return GPTSeleniumAgent(
            chromedriver_path=str(CHROMEDRIVER_LOCATION),
            close_after_completion=False,
            model_for_instructions="gpt-4",
            model_for_responses="gpt-3.5-turbo",
            user_data_dir=str(BROWSERPILOT_DATA_DIR),
        )

    @property
    def work_dir(self) -> Path:
        """Return the working directory for the daemon."""
        return BASE_WORK_DIR / "browser_daemon"

    @property
    def llm_config(self) -> ConfigDict:
        """Return the config for the agent's interfacing LLM."""
        return {
            "functions": [
                {
                    "name": "go_to_url",
                    "description": "Go to a particular URL.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL to go to.",
                            }
                        },
                        "required": ["url"],
                    },
                },
                # {
                #     "name": "click_on_element",
                #     "description": "Click on a particular page element.",
                #     "parameters": {
                #         "type": "object",
                #         "properties": {
                #             "element_description": {
                #                 "type": "string",
                #                 "description": "A natural language description of the element to click on. Do not use full sentences. Example: \"the 'contact us' link\"",
                #             }
                #         },
                #         "required": ["element_description"],
                #     },
                # },
                # type_text
                # zoom_into_section
                # zoom_out
                # read_section_text
            ],
            "raise_on_ratelimit_or_timeout": None,
            "request_timeout": 600,
            "seed": 42,
            "config_list": DEFAULT_CONFIG_LIST,
            "temperature": 0,
        }

    @property
    def allowed_actions(self) -> str:
        """Return a list of allowed actions."""
        return dedent_and_strip(
            """
            - go to a particular URL
            - zoom in to a subsection of the page
            - zoom out on the page
            - read the text in a subsection of the page
            - scroll up or down
            - interact with a specific element on the page (click, type, etc.)
            """
        )

    def validate(self, message: str) -> tuple[bool, str]:
        """Validate a message, and return an error message if it is invalid."""
        instructions = """
        You are a message validation bot. Your purpose is to check for specific components that must be present in a message, and to return an error message if they are missing.

        The message you received is:
        ```text
        {message}
        ```

        The message is supposed to be a natural language command to a web browsing agent to do ONE of the following:
        {allowed_actions}

        If the message can't be classified as a command for one of these actions, or contains multiple actions, then it is invalid.

        Output your validation result as a markdown `text` block. If the message has no errors, output the following:
        ```text
        N/A
        ```

        If the message has errors, output the following (fill in the error message):
        ```text
        Error: Your message was invalid due to asking me to do something I cannot do.
        Original Message: "{message}"
        ```
        """
        instructions = dedent_and_strip(instructions).format(
            message=message, allowed_actions=self.allowed_actions
        )
        messages = [SystemMessage(content=instructions)]
        error = find_llm_validation_error(messages)
        if error:
            error = f"{error}\nAllowed Actions: I am only able to do one and only one of the following actions:\n{self.allowed_actions}"
        return error is None, "" if error is None else error

    def go_to_url(self, url: str) -> str:
        """Go to a URL."""
        instructions = dedent_and_strip(
            f"""
            Go to the URL `{url}`.
            """
        )
        run_with_instructions(self.browserpilot_agent, instructions)
        return f"Successfully navigated to {url}."

    def run(
        self,
        message: str,
    ) -> tuple[str, Callable[[str], str]]:
        """Run the agent."""

        user_proxy = make_hivemind_user_proxy(
            agent=self,
            function_map={
                "go_to_url": self.go_to_url,
            },
            llm_config=self.llm_config,
        )
        assistant = AssistantAgent(
            name=f"{self.name}_assistant",
            llm_config=self.llm_config,
            system_message="Use one of your functions to fulfill the user's request.",
        )
        continue_conversation = continue_agent_conversation(user_proxy, assistant)
        validated, error = self.validate(message)
        if not validated:
            return error, continue_conversation
        user_proxy.initiate_chat(
            assistant,
            message=message,
        )
        return get_last_reply(user_proxy, assistant), continue_conversation

def test_browserpilot() -> None:
    """Test the browserpilot agent."""
    agent = GPTSeleniumAgent(
        chromedriver_path=str(CHROMEDRIVER_LOCATION),
        close_after_completion=False,
        model_for_instructions="gpt-4",
        model_for_responses="gpt-3.5-turbo",
        user_data_dir=str(BROWSERPILOT_DATA_DIR),
    )
    run_with_instructions(agent, "Go to https://google.com")
    agent.driver.quit()


def test_go_to_url() -> None:
    """Test go_to_url."""
    agent = BrowserDaemon()
    result, _ = agent.run("Go to https://google.com")
    agent.browserpilot_agent.driver.quit()
    print(result)


def test_validate() -> None:
    """Test validation."""
    validated, _ = BrowserDaemon().validate("Make a sandwich.")
    assert not validated
    validated, _ = BrowserDaemon().validate("Go to https://google.com")
    assert validated


# TODO: zoom in to header
# ....
# TODO: workflow: "go to the autogen repository and figure out what i said about environments with decomposable tasks"
# ....
# TODO: work out sequence that takes a natural language input and converts it to a browserpilot command, then returns the result of that command
# ....
# TODO: convert action to command > use instructions from browserpilot readme
# TODO: send command to browserpilot
# TODO: MVP: purely text-based browser
# > TODO: convert image of page to element list
# > idea for screenreader


def test() -> None:
    """Test the agent."""
    # test_validate()
    test_go_to_url()

test()

# breakpoint()  # print(*(message.content for message in messages), sep="\n\n")

# from browserpilot.agents.gpt_selenium_agent import GPTSeleniumAgent

# instructions = """Go to Google.com
# Find all textareas.
# Find the first visible textarea.
# Click on the first visible textarea.
# Type in "buffalo buffalo buffalo buffalo buffalo" and press enter.
# Wait 2 seconds."""

# instructions_2 = """
# Find all anchor elements that link to Wikipedia.
# Click on the first one.
# Wait for 2 seconds.
# Scroll down the page.
# Wait for 10 seconds."""

# agent = GPTSeleniumAgent(
#     instructions,
#     ".data/drivers/chromedriver/chromedriver",
#     close_after_completion=False,
#     model_for_instructions="gpt-4",
#     model_for_responses="gpt-3.5-turbo",
# )
# agent.run()
# agent.set_instructions(instructions_2)
# agent.run()
