"""Agent that can browse a webpage and communicate its state."""

from pathlib import Path
from dataclasses import dataclass

import langchain
from langchain.schema import SystemMessage, HumanMessage, BaseMessage, AIMessage
from langchain.chat_models import ChatAnthropic
from langchain.cache import SQLiteCache

from hivemind.config import LANGCHAIN_CACHE_DIR, TEST_DIR, BROWSERPILOT_DATA_DIR
from hivemind.toolkit.models import query_model
from hivemind.toolkit.text_formatting import dedent_and_strip, extract_blocks

langchain.llm_cache = SQLiteCache(
    database_path=str(LANGCHAIN_CACHE_DIR / ".langchain.db")
)


@dataclass
class WebpageOracle:
    """Agent to inspect the contents of a webpage with varying degrees of detail."""

    html: str
    """HTML of the webpage."""

    message_history: list[str]
    """Message history of the agent."""

    _breadcrumbs: list[str] | None = None
    """Breadcrumbs for where the agent is focused on the page."""

    @property
    def role_instructions(self) -> str:
        """Main role instructions for the agent role."""
        return dedent_and_strip(
            """
            # PURPOSE
            Behave like a self-assembling program whose purpose is to represent the contents of the page (given to you as an HTML) in a concise, distilled format to a blind user at various levels of detail specified by the user.
            """
        )

    @property
    def html_context(self) -> str:
        """Instructions for the HTML."""
        return dedent_and_strip(
            """
            # HTML
            Here is the HTML of the page you are representing:

            ```html
            {html}
            ```
            """
        ).format(html=self.html)

    @property
    def base_instructions(self) -> str:
        """Base instructions for the agent."""
        return dedent_and_strip(
            """
            # INSTRUCTIONS
            You are a read-only program, and do NOT have the ability to interact with the page.

            Your goal is to run the remainder of the chat as a fully functioning program that is ready for user input once this prompt is received.
            """
        )

    @property
    def model(self) -> ChatAnthropic:
        """The model for the agent."""
        return ChatAnthropic(
            temperature=0, model="claude-2", max_tokens_to_sample=50000, verbose=False
        )  # hardcoded for now; agent must process large amounts of html text

    def extract_page_outline(self) -> str:
        """Extract the outline of the page."""
        instructions = dedent_and_strip(
            """
            Please give me a top-level, hierarchical outline of the contents on the page.
            Include the TITLE of the page.
            Include the TOP-LEVEL sections on the page.
            Include also the MOST IMPORTANT interactive elements on the page, and their element type (e.g. <a>, <input>, <button>, etc.). 
            Include also the MOST IMPORTANT text elements on the page.
            Enclose the outline in a markdown code block:
            ```markdown
            {outline}
            ```
            """
        )
        messages = [
            SystemMessage(content=self.role_instructions),
            SystemMessage(content=self.html_context),
            SystemMessage(content=self.base_instructions),
            HumanMessage(content=instructions),
        ]
        result = query_model(self.model, messages, printout=False).strip()
        if result := extract_blocks(result, block_type="markdown"):
            return result[0].strip()
        raise ValueError("Could not extract page outline.")

    def update_page(self, html: str) -> None:
        """Update the HTML of the webpage."""
        self.html = html

    @property
    def title(self) -> str:
        """Return the title of the page."""
        return self.html.split("<title>")[1].split("</title>")[0]

    @property
    def breadcrumbs(self) -> tuple[str, ...]:
        """Breadcrumbs for where the agent is focused on the page."""
        return () if self._breadcrumbs is None else tuple(self._breadcrumbs)

    @property
    def page_outline(self) -> str:
        """Outline of the page."""
        return self.extract_page_outline()

    @property
    def full_page_context(self) -> str:
        """Context for the current view of the page."""
        return dedent_and_strip(
            """
            # CURRENT VIEW
            You are currently viewing the full page. Here is the high-level outline of the page:
            ```
            {page_outline}
            ```
            """
        ).format(page_outline=self.page_outline)

    @property
    def current_section_name(self) -> str:
        """Return the name of the current section.
        Possibly not a unique identifier, so only use for display purposes."""
        return self.breadcrumbs[-1]

    _section_outlines: dict[tuple[str, ...], str] | None = None

    @property
    def section_outlines(self) -> dict[tuple[str, ...], str]:
        """All currently generated section outlines."""
        if self._section_outlines is None:
            return {
                (): self.page_outline,
            }
        return self._section_outlines

    @property
    def section_outline(self) -> str:
        """Outline of the current section."""
        return self.section_outlines[self.breadcrumbs]

    @property
    def breadcrumb_display(self) -> str:
        """Display of the breadcrumb trail to the current section."""
        return " > ".join(("Root page", *self.breadcrumbs))

    @property
    def section_context(self) -> str:
        """Context for the current section of the page."""
        return dedent_and_strip(
            """
            # CURRENT VIEW
            You are currently viewing the `{section}` section of the page. Here is the high-level outline of the `{section}` section:
            ```
            {section_outline}
            ```
            Here is the section breadcrumb trail:
            {breadcrumbs}
            """
        ).format(
            section=self.current_section_name,
            section_outline=self.section_outline,
            breadcrumbs=self.breadcrumb_display,
        )

    @property
    def current_view_context(self) -> str:
        """Context for the current view of the page."""
        return self.section_context if self.breadcrumbs else self.full_page_context

    def extract_section_outline(self, section: str) -> str:
        """Zoom in on a section of the page."""
        instructions = dedent_and_strip(
            """
            Please give me a high-level, hierarchical outline of the contents of the `{section}` SUBSECTION of the section you are viewing.
            Include the next-level subsections nested WITHIN the {section} subsection.
            Include the most important INTERACTIVE elements within the {section} subsection, and their element type (e.g. <a>, <input>, <button>, etc.). 
            Enclose the subsection outline in a markdown code block:
            ```markdown
            {{outline}}
            ```
            """
        )
        messages = [
            SystemMessage(content=self.role_instructions),
            SystemMessage(content=self.html_context),
            SystemMessage(content=self.current_view_context),
            SystemMessage(content=self.base_instructions),
            HumanMessage(content=instructions.format(section=section)),
        ]
        result = query_model(self.model, messages, printout=False).strip()
        if result := extract_blocks(result, block_type="markdown"):
            return result[0].strip()
        raise ValueError("Could not extract section outline.")


def test_page_outline() -> None:
    """Test webpage oracle ability to generate page outline."""
    page = Path(TEST_DIR / "cleaned_page.html").read_text(encoding="utf-8")
    oracle = WebpageOracle(html=page, message_history=[])
    print(oracle.extract_page_outline())  # expect hierarchical outline of page


def test_section_outline() -> None:
    """Test webpage oracle ability to generate section outline."""
    page = Path(TEST_DIR / "cleaned_page.html").read_text(encoding="utf-8")
    oracle = WebpageOracle(html=page, message_history=[])
    print(
        oracle.extract_section_outline("Readme")
    )  # expect hierarchical outline of Readme section


def test_zoom() -> None:
    """Test webpage oracle ability to zoom in on a section of the page."""
    page = Path(TEST_DIR / "cleaned_page.html").read_text(encoding="utf-8")
    oracle = WebpageOracle(html=page, message_history=[])
    print(oracle.zoom("Readme"))
    breakpoint()


if __name__ == "__main__":
    # test_page_outline()
    test_section_outline()
    # test_zoom()

# TODO: zoom system
# ....
# > TODO: element search functionality
# TODO: wrapper around browserpilot > navigator daemon
# TODO: convert action to command > use instructions from browserpilot readme
# TODO: give command to page
# TODO: send command to browserpilot
# > TODO: convert image of page to element list
# > idea for screenreader

breakpoint()  # print(*(message.content for message in messages), sep="\n\n")

# from browserpilot.agents.gpt_selenium_agent import GPTSeleniumAgent

# instructions_1 = """Go to Google.com
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

# instructions_3 = """Go to https://github.com/handrew/browserpilot"""

# agent = GPTSeleniumAgent(
#     instructions_3,
#     ".drivers/chromedriver/chromedriver",
#     close_after_completion=False,
#     model_for_instructions="gpt-4",
#     model_for_responses="gpt-3.5-turbo",
#     user_data_dir=BROWSERPILOT_DATA_DIR,
# )
# agent.run()
# # elements = agent._GPTSeleniumAgent__get_html_elements_for_llm()
# elements = agent._remove_blacklisted_elements_and_attributes().find_all()
# breakpoint()
# # agent.set_instructions(instructions_2)
# # agent.run()
