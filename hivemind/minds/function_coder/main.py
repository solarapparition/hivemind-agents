"""Recursive codebase management agent."""

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
from textwrap import dedent
from typing import Protocol, Any, Sequence, Self

import langchain
from langchain.schema import SystemMessage, HumanMessage, BaseMessage
from langchain.cache import SQLiteCache
from hivemind.toolkit.text_extraction import extract_block, extract_blocks
from hivemind.toolkit.text_formatting import dedent_and_strip
from hivemind.toolkit.yaml_tools import default_yaml, as_yaml_str
from hivemind.toolkit.models import (
    query_model,
    precise_model,
)

langchain.llm_cache = SQLiteCache(database_path=".data/.langchain.db")

WORKSPACE_DIR = Path("hivemind/minds/function_coder/.workspace")
NO_CODE_AVAILABLE = "No code available."
NO_HELPERS_AVAILABLE = "No helpers available."
NO_BACKGROUND_AVAILABLE = "No background available."
INTERNAL = "internal"
YELLOW = "\033[33m"
END = "\033[0m"
PYTHON = "python"


@dataclass
class HumanCoder:
    """A human who can code."""


@dataclass
class AICodingAgent:
    """An agent based at a directory that can code."""

    data_dir: Path
    helpers_dir: Path
    data: dict[str, Any]
    state: dict[str, Any]

    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return self.data_dir.name

    @property
    def helpers(self) -> list[str]:
        """Get helper agents."""
        helpers: list[str] = self.data["helpers"]
        return helpers

    @property
    def helper_agents(self) -> list["FunctionAgent"]:
        """Get helper agents."""
        helper_agents = [
            load_function_agent(name, self.helpers_dir) for name in self.helpers
        ]
        return helper_agents

    @property
    def helper_function_names(self) -> list[str]:
        """Get the names of the helpers maintained by this agent."""
        function_names = [
            helper_agent.function_name for helper_agent in self.helper_agents
        ]
        return function_names

    def add_helper(self, helper_name: str) -> None:
        """Add a factor agent."""
        self.helpers.append(helper_name)

    def save(self) -> None:
        """Save the agent to disk."""
        os.makedirs(self.data_dir, exist_ok=True)
        default_yaml.dump(self.data, self.data_dir / "agent.yml")
        default_yaml.dump(self.state, self.data_dir / "state.yml")

    @classmethod
    def from_data_dir(cls, data_dir: Path, helper_lookup_dir: Path) -> Self:
        """Create a codebase agent from a data directory."""
        data = default_yaml.load(data_dir / "agent.yml")
        recorded_helpers: list[str] = data["helpers"]
        existing_helpers = [
            helper_name
            for helper_name in recorded_helpers
            if (helper_lookup_dir / helper_name).exists()
        ]
        data["helpers"] = existing_helpers
        state = default_yaml.load(data_dir / "state.yml")
        return cls(
            data_dir=data_dir, data=data, state=state, helpers_dir=helper_lookup_dir
        )


# @lru_cache
def load_function_agent(name: str, project_dir: Path) -> "FunctionAgent":
    """Load a function agent."""
    function_agent = FunctionAgent.from_name(name, project_dir)
    return function_agent


def extract_function_core(
    function_agent_names: Sequence[str], function_dir: Path
) -> list[dict[str, str]]:
    """Extract core info from function agents."""
    function_agents = [
        load_function_agent(name, function_dir) for name in function_agent_names
    ]
    function_specs: list[dict[str, str]] = [
        function_agent.specs
        for function_agent in function_agents
        if function_agent.specs
    ]
    function_specs_core = [
        {
            "name": specs["name"],
            "signature": specs["signature"],
            "description": specs["description"],
        }
        for specs in function_specs
    ]
    return function_specs_core


def warn_multiple_code_units(
    matching_units: int, pattern: str, code_file: Path
) -> None:
    """Warn about multiple code units matching a pattern."""
    logging.warning(
        f"{YELLOW}Code extraction: multiple (%s) code units found at {code_file} matching code pattern: '%s'. Using first one.{END}",
        matching_units,
        pattern,
    )


def extract_code_unit(function_file: Path, pattern: str, language: str) -> str:
    """Extract a particular unit of the code. Units are: functions, classes, and variables/imports."""
    code_text = function_file.read_text()
    code_units = split_into_units(code_text, language)
    matching_code_units = [unit for unit in code_units if re.search(pattern, unit)]
    if len(matching_code_units) == 0:
        raise ValueError(
            f"Code extraction: no code units found matching code pattern: '{pattern}'"
        )
    if len(matching_code_units) > 1:
        warn_multiple_code_units(len(matching_code_units), pattern, function_file)
    return matching_code_units[0].strip()


@dataclass
class FunctionAgent(AICodingAgent):
    """An agent responsible for maintaining a function."""

    data_dir: Path
    helpers_dir: Path
    data: dict[str, Any]
    state: dict[str, Any]

    @classmethod
    def from_name(cls, name: str, project_dir: Path) -> Self:
        """Create a function agent from a data directory."""
        data_dir = project_dir / name
        return cls.from_data_dir(data_dir, helper_lookup_dir=project_dir)

    @classmethod
    def from_data(
        cls,
        data_dir: Path,
        helpers_dir: Path,
        specs: dict[str, str],
        parent: str,
        language: str,
        helpers: list[str] | None = None,
    ) -> Self:
        """Create a function agent from data."""
        helpers = helpers or []
        data = {
            "specs": specs,
            "parent": parent,
            "helpers": helpers,
            "language": language,
        }
        state = {}
        return cls(data_dir=data_dir, data=data, state=state, helpers_dir=helpers_dir)

    @property
    def source_of_truth(self) -> str:
        """Get the source of truth for the function."""
        source_of_truth = self.data.get("source_of_truth")
        return str(source_of_truth) if source_of_truth else INTERNAL

    @source_of_truth.setter
    def source_of_truth(self, source_of_truth: str) -> None:
        """Set the source of truth for the function."""
        self.data["source_of_truth"] = source_of_truth

    @property
    def locator(self) -> str | None:
        """Get the locator for the function."""
        if self.source_of_truth == INTERNAL:
            return None

        default_locator_map = {
            PYTHON: f"def {self.function_name}",
        }
        locator = self.data.get("locator") or default_locator_map[self.language]
        return locator

    @property
    def function_name(self) -> str:
        """Get the name of the function."""
        return self.data["specs"]["name"]

    @property
    def language(self) -> str:
        """Get the language for the function maintained by this agent."""
        return self.data["language"]

    @property
    def specs(self) -> dict[str, str] | None:
        """Get the specs for the function maintained by this agent."""
        specs: dict[str, str] | None = self.data.get("specs")
        if not specs:
            return None
        return specs

    @specs.setter
    def specs(self, new_specs: dict[str, str]) -> None:
        """Set the specs for the function maintained by this agent."""
        self.data["specs"] = new_specs

    @property
    def specs_text(self) -> str:
        """Get the text version of the specs."""
        if not self.specs:
            return "No specs available."
        updated_specs = self.specs
        if self.specs.get("background") is None:
            updated_specs = self.specs | {"background": NO_BACKGROUND_AVAILABLE}
        specs_text = as_yaml_str(updated_specs)
        return specs_text

    @property
    def signature(self) -> str | None:
        """Get the signature for the function."""
        if not self.specs:
            return None
        return self.specs["signature"]

    @property
    def code(self) -> str | None:
        """Get the code for the function maintained by this agent."""
        if self.source_of_truth == INTERNAL:
            return self.data.get("code")
        if Path(self.source_of_truth).exists():
            locator = self.locator
            if not locator:
                raise ValueError(
                    "Code extraction: no locator provided for non-internal code source."
                )
            code = extract_code_unit(Path(self.source_of_truth), locator, self.language)
            return code
        raise ValueError(
            f"Code extraction: code source `{self.source_of_truth}` not found."
        )

    def set_code(self, code: str) -> None:
        """Set the code for the function."""
        if self.source_of_truth == INTERNAL:
            self.data["code"] = code
            return
        if Path(self.source_of_truth).exists():
            code_file = Path(self.source_of_truth)
            soc_code_text = code_file.read_text(encoding="utf-8")
            current_code_block = self.code
            if not current_code_block:
                raise ValueError(
                    f"Code injection: no existing code found for function `{self.function_name} at {code_file}"
                )
            updated_soc_code_text = soc_code_text.replace(current_code_block, code)
            code_file.write_text(updated_soc_code_text, encoding="utf-8")

    @property
    def code_text(self) -> str:
        """Get the text version of the code."""
        code = self.code
        if not code:
            return NO_CODE_AVAILABLE
        return code

    @property
    def code_generated(self) -> bool:
        """Check if code for the function has been generated."""
        return "code" in self.data

    def code_generation_messaging(self) -> tuple[BaseMessage, ...]:
        """Create messages that generate the code."""
        instructions = """
        Generate code for the function, based on the following process:
        1. Analyze the previous information given about the function, and brainstorm some insights about implementing the function, such as reminders about good coding practices relevant to the situation, the technical approach to take, and what's implied but not explicity stated. Don't repeat the information previously given, but infer some new insights from it.
        2. Identify any external libraries that may be useful for implementing the function, as well as information from the Background that is relevant to the function.
        3. Write a step-by-step procedure for what the code must do, in order to implement the function. Strongly avoid side effects for functions that return values--i.e. functions should either return a value or have side effects, but not both.
        4. Based on the step-by-step procedure, write the names, signatures, and summaries of additional helper functions in {language} that would help make implementing the function more concise; be aggressive about identifying helpers if the function is too complex to be written in 20 lines or less. Don't re-write existing helper functions. Output the new helper functions in the following format:
        ```yaml
        - "name": |-
            <helper_function_name_1>
          "signature": |-
            <helper_function_signature_1>
          "description": |-
            <helper_function_description_1>
        - <same for other helper functions>
        ```
        5. Write the implementation code for the function. Favor a concise, strongly typed style that adheres to the conventions of {language}. Include a single-sentence docstring for the function, but no other comments--the code should be self-documenting.
        Balance and optimize for the following pillars of good code:
          - Readability
          - Testability
          - Maintainability
        Use the helper functions (both existing ones and the new ones you specified) in the implementation. Do NOT write the code for the helper functions. Output the code in the following format:
        ```{language}
        <imports>
        <function_code>
        ```
        """
        instructions = dedent(instructions).strip().format(language=self.language)
        messages = (
            *self.base_identity_messaging,
            SystemMessage(content=instructions),
        )
        return messages

    def generate_code(self) -> tuple[str, str | None]:
        """Generate code for the function."""
        messages = self.code_generation_messaging()
        # breakpoint()  # print(*(message.content for message in messages), sep="\n\n---message---\n\n")
        result = query_model(precise_model, messages, color=31)
        code = extract_block(result, self.language)
        if not code:
            raise ValueError(
                f"Code extraction: no code found in generation of `{self.function_name}`."
            )
        helper_specs = extract_block(result, "yaml")
        return code, helper_specs

    @property
    def helpers_text(self) -> str:
        """Get the text for the helpers maintained by this agent."""
        if not self.helpers:
            return NO_HELPERS_AVAILABLE
        helper_info = extract_function_core(self.helpers, self.helpers_dir)
        helper_info = as_yaml_str(helper_info)
        # breakpoint()
        # helpers_text = dump_yaml_str(self.helper_function_names)
        return helper_info

    def set_helpers(self, helper_names: Sequence[str]) -> None:
        """Set the helpers for the function."""
        self.data["helpers"] = helper_names

    @property
    def function_text(self) -> str:
        """Get the text for the function maintained by this agent."""
        text = """
        Function Specifications:
        ```text
        {specs}
        ```
        
        Helper Functions:
        ```yaml
        {helpers}
        ```

        Function Code:
        ```{language}
        {code}
        ```
        """
        text = dedent_and_strip(text).format(
            language=self.language,
            specs=self.specs_text,
            code=self.code_text,
            helpers=self.helpers_text,
        )
        return text

    @property
    def role_text(self) -> str:
        """Get the role of the agent."""
        text = """
        You are an senior software architect responsible for creating and maintaining the information about a function in a codebase, specifically:
        - specifications for the function, including background technical information required to understand the function
        - implementation code for the function
        - helpers for the function
        """
        text = dedent_and_strip(text)
        return text

    @property
    def base_identity_messaging(self) -> tuple[BaseMessage, ...]:
        """Base role for the function agent."""
        messages = (
            SystemMessage(content=self.role_text),
            SystemMessage(content=self.function_text),
        )
        return messages

    @property
    def feedback_request(self) -> str:
        """Get a feedback request."""
        instructions = """
        Please review the function information and provide feedback:
        """
        instructions = dedent_and_strip(instructions)
        # instructions = dedent_and_strip(instructions).format(
        #     function_text=self.function_text
        # )
        return instructions

    def code_update_messaging(self, feedback: str) -> tuple[BaseMessage, ...]:
        """Get messages for updating the function from feedback."""
        instructions = """
        Your current task is to update the function code based on the feedback below:
        ```text
        {feedback}
        ```

        Go through the following process to perform the update:
        1. Think through what new information the feedback provides, that wasn't known before.
        2. Identify any aspects about the feedback (terms, references, directions, etc.) that aren't clear.
        3. Make some reasonable deductions and assumptions about the feedback, especially any parts of it that are unclear.
        4. Write a detailed, step-by-step procedure to update the `Function Code`, in order to implement the feedback. Updates can include adding new helper functions. Be as specific, technical, and thorough as possible to avoid ambiguities and mistakes.
        5. If helper functions must be added or modified, then output their specifications. The `description` in the specs should be as detailed and technical as possible and include helpful implementation code snippets.
        ```yaml
        - "name": |-
            <new/updated helper function 1 name>
          "signature": |-
            <new/updated helper function 1 signature>
          "description": |-
            <new/updated helper function 1 description>
        - <same for other helper functions>
        ```
        Otherwise, output an empty list:
        ```yaml
        []
        ```
        6. Update the `Function Code` section of your function information with the changes you identified. Do NOT include code for helper functions here--those will be implemented separately. Format the output the following way:
        ```{language}
        <main function code>
        ```
        """
        instructions = dedent_and_strip(instructions).format(
            feedback=feedback, language=self.language
        )
        messages = (
            *self.base_identity_messaging,
            SystemMessage(content=instructions),
        )
        return messages

    def set_function_info(
        self, new_specs: dict[str, str], new_code: str, new_helpers: list[str]
    ) -> None:
        """Set the function info."""
        self.data["specs"] = new_specs
        code = self.code
        if code:
            self.data["code"] = new_code
        if self.helpers:
            self.data["helpers"] = new_helpers

    def update_from_feedback(self, feedback: str) -> None:
        """Update the function based on feedback."""
        messages = self.code_update_messaging(feedback)
        # breakpoint()  # print(*(message.content for message in messages), sep="\n\n---message---\n\n")
        result = query_model(precise_model, messages, color=31)
        updated_code_blocks = extract_blocks(result, self.language)
        if not updated_code_blocks:
            raise ValueError(
                f"Code extraction: no code found in update of `{self.function_name}`."
            )
        updated_code = updated_code_blocks[-1]
        new_or_updated_helpers = extract_block(result, "yaml") or "[]"
        new_or_updated_helpers = default_yaml.load(new_or_updated_helpers)
        new_helper_agents = [
            FunctionAgent.from_data(
                data_dir=self.helpers_dir / helper_specs["name"],
                helpers_dir=self.helpers_dir,
                specs=helper_specs,
                parent=self.function_name,
                helpers=[],
                language=self.language,
            )
            for helper_specs in new_or_updated_helpers
        ]
        # breakpoint()
        for new_helper_agent in new_helper_agents:
            if not new_helper_agent.specs:
                continue
            if new_helper_agent.function_name not in self.helper_function_names:
                self.add_helper(new_helper_agent.function_name)
                new_helper_agent.save()
                continue
            existing_helper_agent = load_function_agent(
                new_helper_agent.function_name, self.helpers_dir
            )
            existing_helper_agent.specs = (
                existing_helper_agent.specs | new_helper_agent.specs
                if existing_helper_agent.specs
                else new_helper_agent.specs
            )
            existing_helper_agent.save()
        self.set_code(updated_code)
        return

        # updated_code = None if updated_code == NO_CODE_AVAILABLE else updated_code

        """
        6. Identify changes that would need to be made to the `Specifications` section of your function information, then make those changes. Output the updated specifications in the following format (the signature also duplicates the name; this is intended):
        ```yaml
        "name": |-
          <name>
        "signature": |-
          <signature>
        "summary": |-
          <summary>
        "details": |-
          <details>
        ```
        """

        new_helper_names = [
            new_helper_agent.function_name for new_helper_agent in new_helper_agents
        ]
        updated_specs = extract_blocks(result, "yaml")[-2]
        updated_specs = default_yaml.load(updated_specs)
        # breakpoint()
        self.set_function_info(updated_specs, updated_code, new_helper_names)

    def write_all(self, outfile: Path) -> None:
        """Write function and all helpers to a file. Helpers are written first."""
        for helper_agent in self.helper_agents:
            helper_agent.write_all(outfile)
        with open(outfile, "a", encoding="utf-8") as file:
            code = self.code
            if not code:
                return
            file.write(code)
            file.write("\n\n")


@dataclass
class CodebaseAgent(AICodingAgent):
    """Lead agent for the codebase."""

    data_dir: Path
    data: dict[str, Any]
    state: dict[str, Any]

    @classmethod
    def from_project_dir(cls, project_dir: Path) -> Self:
        """Create a codebase agent from a project directory."""
        return cls.from_data_dir(project_dir, project_dir)

    @classmethod
    def init_at(cls, project_dir: Path, language: str) -> None:
        """Initialize a codebase agent at a project directory."""
        helpers: list[str] = []
        data = {
            "language": language,
            "helpers": helpers,
        }
        state = {}
        codebase_agent = cls(
            data_dir=project_dir, helpers_dir=project_dir, data=data, state=state
        )
        codebase_agent.save()


CodingAgent = AICodingAgent | HumanCoder


@dataclass
class HumanReviewer:
    """A human who can review code."""


class DirectorAgent(Protocol):
    """An agent that can direct development of the codebase."""

    def decide(self, message: HumanMessage) -> str:
        """Decide on a choice."""
        ...


def get_choice(
    options: Sequence[str],
    director: DirectorAgent,
    requester_name: str,
    information: str,
) -> int:
    """Get a choice from the director."""
    choice_instructions = """
    {requester_name}: Information:

    {information}
    
    Please choose one of the following options (enter only the number), or 0 to go back:

    {options}
    """
    options = [f"{idx + 1}. {option}" for idx, option in enumerate(options)]
    choice_instructions = dedent_and_strip(choice_instructions).format(
        requester_name=requester_name,
        options="\n".join(options),
        information=information,
    )
    message = HumanMessage(content=choice_instructions)
    choice = director.decide(message)
    choice = int(choice)
    return choice


def get_basic_info(director: DirectorAgent) -> tuple[str, str, str, str | None]:
    """Get specs for a function."""
    name_instructions = """
    Provide the name of the function agent to be created. Provide only the name, without any type signature or other information.
    """
    name_instructions = dedent_and_strip(name_instructions)
    message = HumanMessage(content=name_instructions)
    name = director.decide(message)
    signature_instructions = """
    Provide the type signature of the function, including its name, input types, and output type.
    """
    signature_instructions = dedent_and_strip(signature_instructions)
    message = HumanMessage(content=signature_instructions)
    signature = director.decide(message)
    description_instructions = """
    Provide a short description of what the function does.
    """
    description_instructions = dedent_and_strip(description_instructions)
    message = HumanMessage(content=description_instructions)
    description = director.decide(message)
    parent_instructions = """
    Provide the name of the agent responsible for supervising the function. Press Enter to use the default codebase-level agent.
    """
    parent_instructions = dedent_and_strip(parent_instructions)
    message = HumanMessage(content=parent_instructions)
    parent_name = director.decide(message) or None
    return name, signature, description, parent_name


def specify_function_agent(
    project_dir: Path,
    director: DirectorAgent,
    codebase_agent_name: str,
    language: str,
) -> FunctionAgent:
    """Specify a function agent using a director's input."""
    (
        function_name,
        function_signature,
        function_description,
        parent_name,
    ) = get_basic_info(director)
    function_specs = {
        "name": function_name,
        "signature": function_signature,
        "description": function_description,
    }
    parent_name = parent_name or codebase_agent_name
    function_dir = project_dir / function_name
    function_agent = FunctionAgent.from_data(
        data_dir=function_dir,
        helpers_dir=project_dir,
        specs=function_specs,
        parent=parent_name,
        helpers=[],
        language=language,
    )
    return function_agent


@dataclass
class HumanDirector:
    """A human director for the project."""

    def decide(self, message: HumanMessage) -> str:
        """Decide on a choice."""
        instructions = "\n" + message.content + "\n\n>>> "
        return input(instructions)


def get_input(request: str, director: DirectorAgent, requester_name: str) -> str:
    """Get info from the director."""
    message = HumanMessage(content=request)
    info = director.decide(message)
    return info


def get_function_reading_info(
    director: DirectorAgent, default_search_text: str | None = None
) -> tuple[Path, str]:
    """Get info about a function to read."""
    function_file_instructions = """
    Provide the path to the file containing the function you want to read.
    """
    function_file_instructions = dedent_and_strip(function_file_instructions)
    message = HumanMessage(content=function_file_instructions)
    function_file = Path(director.decide(message))
    function_search_text_instructions = """
    Provide a regex pattern that uniquely identifies the function you want to read in the file.
    """
    function_search_text_instructions = dedent_and_strip(
        function_search_text_instructions
    )
    if default_search_text:
        function_search_text_instructions += (
            f" Press Enter to use the default search text: '{default_search_text}'."
        )
    message = HumanMessage(content=function_search_text_instructions)
    function_search_text = director.decide(message)
    if not function_search_text and default_search_text:
        function_search_text = default_search_text
    return function_file, function_search_text


def is_pyfunc_new_unit_start(
    current_line: str, previous_line: str, current_unit: list[str]
) -> bool:
    """Determines if a new unit starts in code."""
    if not current_line or current_line.isspace():
        return False
    if current_line.startswith((" ", "\t", ")", "]", "}")):
        return False
    if previous_line and previous_line.strip().startswith("@"):
        return False
    if not current_unit:
        return False
    return True


def split_into_units(code_text: str, language: str) -> list[str]:
    """Split Python code text into basic units: variables/imports, functions, classes."""
    if language.lower() != PYTHON:
        raise ValueError("Only Python language is supported")
    code_units: list[str] = []
    current_unit: list[str] = []
    lines = code_text.split("\n")
    for i, line in enumerate(lines):
        if i > 0 and is_pyfunc_new_unit_start(line, lines[i - 1], current_unit):
            code_units.append("\n".join(current_unit))
            current_unit = []
        current_unit.append(line)
    if current_unit:
        code_units.append("\n".join(current_unit))
    return code_units


def get_outfile(director: DirectorAgent) -> Path:
    """Get the file to write the function to."""
    outfile_instructions = """
    Provide the path to the file to write the function to.
    """
    outfile_instructions = dedent_and_strip(outfile_instructions)
    message = HumanMessage(content=outfile_instructions)
    outfile = Path(director.decide(message))
    return outfile


def default_function_search_text(signature: str | None, language: str) -> str | None:
    """Get the default function search text--usually this is just the name."""
    if not signature:
        return None
    if language.lower() == PYTHON:
        return f"def {signature.split('(')[0]}"
    raise ValueError(f"Language not supported: {language}.")


def run_coder(
    agent_stack: Sequence[CodingAgent],
    director: DirectorAgent,
    project_dir: Path,
    language: str,
) -> Sequence[CodingAgent]:
    """Run the coding system using a particular coding agent as a starting point."""

    current_agent = agent_stack[-1]
    if isinstance(current_agent, HumanCoder):
        return []

    options = []
    if isinstance(current_agent, CodebaseAgent):
        helper_agent_options = current_agent.helper_function_names
        options = [
            "create function agent",
            *helper_agent_options,
        ]
        choice = get_choice(options, director, current_agent.name, "<TBD>")
        if choice == 0:
            return agent_stack[:-1]
        if choice == 1:
            function_agent = specify_function_agent(
                project_dir, director, current_agent.name, language
            )
            current_agent.add_helper(function_agent.name)
            current_agent.save()
            function_agent.save()
            return [*agent_stack, function_agent]
        if 1 < choice <= len(options):
            helper_agent_name = current_agent.helpers[choice - 2]
            helper_agent = FunctionAgent.from_name(helper_agent_name, project_dir)
            return [*agent_stack, helper_agent]
        raise ValueError(f"Invalid choice: {choice}.")

    # otherwise, this is supposed to be a function agent
    if not isinstance(current_agent, FunctionAgent):
        raise ValueError(
            f"Invalid agent type: {type(current_agent)}. Expected to be `FunctionAgent`."
        )
    function_agent = current_agent
    helper_agent_options = function_agent.helper_function_names
    options = [
        "feedback",
        "generate",
        "read",
        "write all",
        *helper_agent_options,
    ]
    choice = get_choice(
        options, director, function_agent.name, function_agent.function_text
    )
    if choice == 0:
        return agent_stack[:-1]
    if choice == 1:
        feedback = get_input(
            function_agent.feedback_request, director, function_agent.name
        )
        function_agent.update_from_feedback(feedback)
        function_agent.save()
        return agent_stack
    if choice == 2:
        code, raw_helper_specs = function_agent.generate_code()
        raw_helper_specs = raw_helper_specs or "[]"
        new_helper_specs: list[dict[str, str]] | str = default_yaml.load(raw_helper_specs)
        if isinstance(new_helper_specs, str):
            new_helper_specs = []
        new_helper_agents = [
            FunctionAgent.from_data(
                data_dir=project_dir / specs["name"],
                helpers_dir=project_dir,
                specs=specs,
                parent=function_agent.name,
                helpers=[],
                language=function_agent.language,
            )
            for specs in new_helper_specs
            if not (project_dir / specs["name"]).exists()
        ]
        function_agent.set_code(code)
        for helper_agent in new_helper_agents:
            function_agent.add_helper(helper_agent.name)
            helper_agent.save()
        function_agent.save()
        return agent_stack
    if choice == 3:
        default_search_text = default_function_search_text(
            function_agent.signature, function_agent.language
        )
        function_file, function_search_text = get_function_reading_info(
            director, default_search_text
        )
        code = extract_code_unit(function_file, function_search_text, language)
        # breakpoint()  # /Users/solarapparition/repos/gx_translation/gx_translation/translation.py # def is_list_field
        function_agent.source_of_truth = str(function_file)
        function_agent.save()
        return agent_stack
    if choice == 4:
        outfile = get_outfile(director)
        function_agent.write_all(outfile)
        return agent_stack
    if 4 < choice <= len(options):
        helper_agent_name = current_agent.helpers[choice - 5]
        helper_agent = FunctionAgent.from_name(helper_agent_name, project_dir)
        return [*agent_stack, helper_agent]
    raise ValueError(f"Invalid choice: {choice}.")


def get_project_name(director: DirectorAgent, project_dir: Path) -> str:
    """Get the name of the project."""
    project_name_instructions = """
    Choose the name of the project to work on.
    """
    project_name_instructions = dedent_and_strip(project_name_instructions)
    options = [name.name for name in project_dir.iterdir() if name.is_dir()]
    option_chosen = get_choice(
        options, director, "FunctionCoder", project_name_instructions
    )
    option_chosen -= 1
    if option_chosen in range(len(options)):
        project_name = options[option_chosen]
        return project_name
    if option_chosen == -1:
        project_name = get_input(
            "Provide the name of the project to create.", director, "FunctionCoder"
        )
        return project_name
    raise ValueError(f"Invalid choice: {option_chosen}.")


def main() -> None:
    """Run example."""

    human_director = HumanDirector()
    project_name = get_project_name(human_director, WORKSPACE_DIR)
    project_dir = WORKSPACE_DIR / project_name

    if not project_dir.exists():
        CodebaseAgent.init_at(project_dir, language=PYTHON)
    agent_stack = [HumanCoder(), CodebaseAgent.from_project_dir(project_dir)]

    while True:
        if not agent_stack:
            break
        agent_stack = run_coder(
            agent_stack, human_director, project_dir, language=PYTHON
        )


if __name__ == "__main__":
    main()
