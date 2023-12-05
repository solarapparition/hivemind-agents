"""Component forge."""

import sys
import traceback
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import total_ordering
from pathlib import Path
from typing import Any, Callable, List, Sequence, TypeVar, Generic, Self
from colorama import Fore

from langchain.schema import SystemMessage

from hivemind.config import configure_langchain_cache
from hivemind.toolkit.models import super_creative_model, query_model
from hivemind.toolkit.text_formatting import dedent_and_strip
from hivemind.toolkit.text_extraction import extract_blocks

AGENT_COLOR = Fore.RED


T = TypeVar("T")
NA_STRING = "N/A"
configure_langchain_cache(Path(".data/hephaestus/llm_cache.db"))


@dataclass
class ValidationResult:
    """Validation result for a component."""

    description: str
    passed: bool
    feedback: str

    def __str__(self) -> str:
        """Get the string representation of the validation result."""
        template = """
        Description: {description}
        Passed: {passed}
        Feedback:
        {feedback}
        """
        return dedent_and_strip(template).format(
            description=self.description, passed=self.passed, feedback=self.feedback
        )


@dataclass
@total_ordering
class Component(Generic[T]):
    """A component of some system."""

    content: str
    to_runnable: Callable[[str], T]
    validation_results: list[ValidationResult] = field(default_factory=list)

    @property
    def runnable(self) -> T:
        """Get the runnable version of the component."""
        return self.to_runnable(self.content)

    @property
    def rating(self) -> float | None:
        """Get the score of the component."""
        return (
            sum(
                validation_result.passed
                for validation_result in self.validation_results
            )
            / len(self.validation_results)
            if self.validation_results
            else None
        )

    @property
    def failed_validation_results(self) -> list[ValidationResult]:
        """Get the failed validation results."""
        return [
            validation_result
            for validation_result in self.validation_results
            if not validation_result.passed
        ]

    def __str__(self) -> str:
        """Get the string representation of the component."""
        template = """
        Variation Content:
        ```start_of_variation_content
        {content}
        ```end_of_variation_content

        Variation Rating: {rating}/1.0

        Failed Validation Results for Variation:

        {failed_validation_results}
        """

        failed_validation_results = (
            "\n\n---\n\n".join(
                str(validation_result)
                for validation_result in self.failed_validation_results
            )
            or NA_STRING
        )

        return dedent_and_strip(template).format(
            content=self.content,
            rating=self.rating,
            failed_validation_results=failed_validation_results,
        )

    def _manual_lt(self, other: Self) -> bool:
        """Manually compare components."""
        prompt = """
        Compare the following two components:
        ## Component 1
        {component_1}

        ## Component 2
        {component_2}
        """
        prompt = dedent_and_strip(prompt).format(
            component_1=str(self), component_2=str(other)
        )
        print(prompt)
        choice = input("Which component is better? (1/2): ")
        while choice not in ["1", "2"]:
            choice = input("Please enter 1 or 2: ")
        return choice == "2"

    def __lt__(self, other: Self) -> bool:
        """Compare components."""
        if self.rating and other.rating and self.rating < other.rating:
            return True
        return self._manual_lt(other)


class ComponentTypeName(Enum):
    """Type of component."""

    FUNCTION = "function"


ComponentPool = List[Component[T]]


@dataclass
class Validation(Generic[T]):
    """A validation for a component."""

    name: str
    description: str
    validation_function: Callable[[Component[T]], ValidationResult]

    def __call__(self, component: Component[T]) -> ValidationResult:
        """Run the validation."""
        return self.validation_function(component)

    def __str__(self) -> str:
        """Get the string representation of the validation."""
        template = """
        Name: {name}
        Description: {description}
        """
        return dedent_and_strip(template).format(
            name=self.name, description=self.description
        )

    @classmethod
    def from_test_function(
        cls,
        test_function: Callable[[T], None],
    ) -> Self:
        """Create a validation from a test function. Captures printout from the function as feedback."""
        name = test_function.__name__
        description = test_function.__doc__ or NA_STRING

        def validation_function(component: Component[T]) -> ValidationResult:
            """Run the validation."""
            try:
                test_function(component.runnable)
            except Exception:  # pylint: disable=broad-except
                stack_trace = "\n".join(traceback.format_exception(*sys.exc_info()))
                return ValidationResult(
                    description=description,
                    passed=False,
                    feedback=stack_trace,
                )
            return ValidationResult(
                description=description,
                passed=True,
                feedback=NA_STRING,
            )

        return cls(
            name=name,
            description=description,
            validation_function=validation_function,
        )


@dataclass
class ComponentForge(Generic[T]):
    """Forge and maintain some component."""

    name: str
    component_type: ComponentTypeName
    description: str
    usage_context: str
    top_variations_size_limit: int
    to_runnable: Callable[[str], T]
    dependencies: list[Component[T]] = field(default_factory=list)
    variations: list[Component[T]] = field(default_factory=list)
    top_variations: list[Component[T]] = field(default_factory=list)
    validations: list[Validation[T]] = field(default_factory=list)

    @property
    def top_variations_printout(self) -> str:
        """Get the top variations as a string."""
        header = "Variation {number}\n----------------\n\n"
        return (
            "\n\n".join(
                "\n".join(
                    [
                        header.format(number=index + 1),
                        str(variation),
                    ]
                )
                for index, variation in enumerate(reversed(self.top_variations))
            )
            or NA_STRING
        )

    @property
    def dependencies_printout(self) -> str:
        """Get the dependencies as a string."""
        return (
            "\n".join(str(dependency) for dependency in self.dependencies) or NA_STRING
        )

    def generate_variation(self) -> Component[T]:
        """Generate variations of the component."""
        context = """
        ## MISSION
        You are a {component_type} forging agent, able to create and improve upon variations of a {component_type} based on specifications and feedback, given below.

        ## {COMPONENT_TYPE} DESCRIPTION
        The {component_type} you are working on is called `{component_name}`, with the following description:
        ```start_of_description
        {component_description}
        ```end_of_description

        ## USAGE EXAMPLES
        Here are some examples of `{component_name}` being used:
        ```start_of_usage_examples
        {usage_context}
        ```end_of_usage_examples

        ## DEPENDENCIES
        Here are some other {component_type}s that `{component_name}` depends on:
        ```start_of_dependencies
        {dependencies}
        ```end_of_dependencies

        ## VARIATION TRAJECTORY
        Here are the variations of `{component_name}` that you have created so far, as well as feedback on them, if available, in order from worst to best:
        ```start_of_variation_trajectory
        {variation_trajectory}
        ```end_of_variation_trajectory
        """
        context = dedent_and_strip(context).format(
            component_type=self.component_type.value,
            COMPONENT_TYPE=self.component_type.value.upper(),
            component_name=self.name,
            component_description=self.description,
            usage_context=self.usage_context,
            dependencies=self.dependencies_printout,
            variation_trajectory=self.top_variations_printout,
        )

        request = """
        ## REQUEST: CREATE VARIATION
        Use the following reasoning process to decide what to do next:
        ```start_of_reasoning_steps
        1. Review COMPONENT DESCRIPTION to understand the component's purpose, functionality, and scope. Critical for foundational understanding. Ensure comprehension of the component's core role and characteristics.
        2. Examine COMPONENT TYPE to clarify the nature of the component - function, class, LLM prompt, etc. Determines the framework for modifications and enhancements. Type influences structural and syntactical constraints.
        3. Analyze USAGE EXAMPLES for practical applications and scenarios where the component is implemented. Usage scenarios reveal real-world demands and performance expectations. Understand context to align improvements with practical needs.
        4. Assess DEPENDENCIES to identify external components influencing or being influenced by this component. Dependencies highlight integration points and potential constraints or compatibilities. Essential for ensuring seamless integration of improvements.
        5. Study VARIATION TRAJECTORY, specifically the progression from worst to best variations. Focus on RATING and VALIDATION_RESULTS to discern patterns of improvements and persisting weaknesses. Learn from past iterations to avoid repeating errors and to build upon successes.
        6. Synthesize information from steps 1-5 to identify areas of potential enhancement. Cross-reference component's purpose, type, usage, dependencies, and historical trajectory. Look for patterns, gaps, or unexplored avenues that suggest opportunities for improvement.
        7. Generate a new component variation, incorporating insights gained from the reasoning steps. Ensure that this iteration addresses identified weaknesses, leverages strengths, and aligns with the component's purpose and usage context. Balance innovation with feasibility and compatibility with existing systems.
        ```end_of_reasoning_steps

        In your reply, you must include output from ALL steps of the reasoning process, in this format:
        1. {step_1_output}
        2. {step_2_output}
        [...]
        7. Component Variation:
        ```start_of_component_generation_output
        {component_generation_output}
        ```end_of_action_choice_output
        Any additional comments or thoughts can be added before or after the required output.
        """
        request = dedent_and_strip(request)

        messages = [
            SystemMessage(content=context),
            SystemMessage(content=request),
        ]
        breakpoint()
        result = query_model(
            super_creative_model,
            messages,
            printout=True,
            color=AGENT_COLOR,
            preamble=f"Creating variation for component `{self.name}...",
        ).strip()
        component_generation_output = extract_blocks(
            result, "start_of_component_generation_output"
        )
        assert component_generation_output is not None
        return Component(component_generation_output[0], self.to_runnable)

    def validate(self, component: Component[T]) -> None:
        """Run and record validation for a component."""
        for validation in self.validations:
            component.validation_results.append(validation(component))

    def add_validation(self, validation: Validation[T]) -> None:
        """Add a new validation."""
        self.validations.append(validation)

    def add_tests_as_validations(self, *test_functions: Callable[[T], None]) -> None:
        """Add a test function as a validation."""
        for test_function in test_functions:
            self.add_validation(Validation.from_test_function(test_function))

    def integrate_variation(self, variation: Component[T]) -> None:
        """Add a variation to the variations list."""
        self.variations.append(variation)
        if len(self.top_variations) < self.top_variations_size_limit:
            self.top_variations.append(variation)
        elif variation > self.top_variations[-1]:
            self.top_variations[-1] = variation
        else:
            return
        self.top_variations.sort(key=lambda variation: variation)

    def __str__(self) -> str:
        """Get the string representation of the component forge."""
        template = """
        Name: {name}
        Type: {component_type}

        Description:
        {description}

        Usage Context:
        {usage_context}

        Dependencies:
        {dependencies}

        Top Variations:
        {top_variations}
        """
        return dedent_and_strip(template).format(
            name=self.name,
            component_type=self.component_type.value,
            description=self.description,
            usage_context=self.usage_context,
            dependencies=self.dependencies_printout,
            top_variations=self.top_variations_printout,
        )

    def create_variation(self) -> None:
        """Create a variation of the component, validate it, and add it to the variations list."""
        variation = self.generate_variation()
        self.validate(variation)
        self.integrate_variation(variation)

    def serialize(self) -> dict[str, Any]:
        """Serialize the component forge."""

        excluded_types: tuple[type, ...] = (Callable,)  # type: ignore

        def enum_dict_factory(data: Sequence[tuple[str, Any]]) -> dict[str, Any]:
            return {
                key: (value.name if isinstance(value, Enum) else value)
                for key, value in data
                if not isinstance(value, excluded_types)
            }

        return asdict(
            self,
            dict_factory=enum_dict_factory,
        )


# update usage context based on feedback items
# > remove caching


forge_test_name = "process_employee_data"
forge_test_description = """
A Python function that processes and normalizes datasets.
Accepts a list of dictionaries, each representing a data record.
Cleans data by removing empty fields and standardizing text (e.g., case normalization).
Normalizes income data, scaling values to a proportion of 100000.
Outputs a list of dictionaries with cleaned and normalized data.

Parameters and Return Type:
- Input: data_records (List[Dict[str, Union[str, int, float]]]) - A list of dictionaries, each containing various data fields.
- Output: List[Dict[str, Union[str, float]]] - Processed data records.
"""
forge_test_usage_context = """
raw_data = [
    {"name": "John Doe", "age": 30, "income": 50000, "comment": ""},
    {"name": "jane smith", "age": "", "income": 70000, "comment": "senior analyst"}
]
processed_data = process_employee_data(raw_data)
print(processed_data)
# [
    # {"name": "John Doe", "age": 30, "income": 0.5, "comment": None},  # assuming income is normalized to 0-1 scale
    # {"name": "Jane Smith", "age": None, "income": 0.7, "comment": "Senior Analyst"}  # text fields are capitalized, missing age represented as None
# ]
"""


def convert_test_function_to_runnable(test_function: str) -> Callable[..., Any]:
    """Convert a test function to a runnable function."""
    exec(test_function)  # pylint: disable=exec-used
    return locals()[test_function.split("\n")[0].split("(")[0].replace("def ", "")]


VARIATIONS_SIZE_LIMIT_TEST = 3


forge_test = ComponentForge[Callable[..., Any]](
    name=forge_test_name,
    component_type=ComponentTypeName.FUNCTION,
    description=dedent_and_strip(forge_test_description),
    usage_context=dedent_and_strip(forge_test_usage_context),
    top_variations_size_limit=VARIATIONS_SIZE_LIMIT_TEST,
    to_runnable=convert_test_function_to_runnable,
)


def test_generate_variation() -> None:
    """Test generate variation."""
    assert (variation := forge_test.generate_variation())
    print(variation.content)


def test_create_variation() -> None:
    """Test create variation."""
    assert not forge_test.top_variations
    forge_test.create_variation()
    assert forge_test.top_variations
    print(forge_test)


def test_variation_comparison() -> None:
    """Test variation comparison."""
    variation_1 = Component[str]("variation 1", to_runnable=lambda x: x)
    variation_2 = Component[str]("variation 2", to_runnable=lambda x: x)
    if variation_1 < variation_2:
        print("variation 1 is better than variation 2")
    else:
        print("variation 2 is better than variation 1")


def test_serialize() -> None:
    """Test serialization."""
    print(json.dumps(forge_test.serialize()))


def validation_test_functions() -> list[Callable[..., None]]:
    """Validation test functions."""

    def test_name_capitalization(process_employee_data: Callable[..., Any]) -> None:
        """Test that the employee name is capitalized."""
        input_data = [{"name": "jane doe"}]
        expected_output = [{"name": "Jane Doe"}]
        assert (
            actual_output := process_employee_data(input_data)
        ) == expected_output, f"Name not capitalized: expected: {expected_output}, actual: {actual_output}"

    def test_income_normalization(process_employee_data: Callable[..., Any]) -> None:
        """Test that the income is normalized."""
        input_data = [{"income": 50000}, {"income": 200000}]
        expected_output = [{"income": 0.5}, {"income": 2.0}]
        assert (
            actual_output := process_employee_data(input_data)
        ) == expected_output, f"Income not normalized: expected: {expected_output}, actual: {actual_output}"

    def test_missing_age(process_employee_data: Callable[..., Any]) -> None:
        """Test that the missing age is represented as None."""
        input_data = [{"age": ""}]
        expected_output = [{"age": None}]
        assert (
            actual_output := process_employee_data(input_data)
        ) == expected_output, f"Missing age not represented as None: expected: {expected_output}, actual: {actual_output}"

    def test_comment_standardization(process_employee_data: Callable[..., Any]) -> None:
        """Test that the comment is standardized."""
        input_data = [{"comment": ""}, {"comment": "junior analyst"}]
        expected_output = [{"comment": None}, {"comment": "Junior Analyst"}]
        assert (
            actual_output := process_employee_data(input_data)
        ) == expected_output, f"Comment not standardized: expected: {expected_output}, actual: {actual_output}"

    def test_other_values(process_employee_data: Callable[..., Any]) -> None:
        """Test that other values are preserved."""
        input_data = [{"other": {}}]  # type: ignore
        expected_output = [{"other": {}}]  # type: ignore
        assert (
            actual_output := process_employee_data(input_data)
        ) == expected_output, f"Other values not preserved: expected: {expected_output}, actual: {actual_output}"

    return [
        test_name_capitalization,
        test_income_normalization,
        test_missing_age,
        test_comment_standardization,
        test_other_values,
    ]


def test_validate() -> None:
    """Test validation."""
    forge_test.add_tests_as_validations(*validation_test_functions())
    variation = forge_test.generate_variation()
    forge_test.validate(variation)
    for validation_result in variation.validation_results:
        print(validation_result)


def test_multiple_variations() -> None:
    """Test full process."""
    forge_test.add_tests_as_validations(*validation_test_functions())
    for _ in range(VARIATIONS_SIZE_LIMIT_TEST):
        forge_test.create_variation()


def test() -> None:
    """Run test."""
    # test_generate_variation()
    # test_create_variation()
    # test_variation_comparison()
    # test_serialize()
    # test_validate()
    # test_multiple_variations()


test()

breakpoint()

# class PromptOptimizer:
#     """Optimizer for a code component of some system."""

#     def generate_component_variations(
#         self, component: CodeContent, context: CodeContext
#     ) -> Iterable[CodeContent]:
#         """Attempt to create improved versions of a component."""
#         raise NotImplementedError

#     def run_environment(self, component: CodeContent) -> CodeOutput:
#         """Run the environment with the component."""
#         raise NotImplementedError

#     def evaluate(self, component: CodeContent, output: CodeOutput) -> CodeEvaluation:
#         """Evaluate the results of running the environment with the component."""
#         raise NotImplementedError

#     def ontological_context(self, component: CodeContent) -> CodeContext:
#         """Create the ontological context for the component.
#         The ontological context is the subset of the environment that is needed to understand what the component does in the environment, and would usually be some. For example, if the component is a function, the ontological context could be other functions that use it and variables that are relevant to it.
#         """
#         raise NotImplementedError

#     def select_improvement_candidates(  # type: ignore
#         self, components: CodeComponentPool
#     ) -> CodeComponentPool:
#         """Select items from the component pool that would be improved. Usually these would be components with high ratings."""
#         raise NotImplementedError

#     def sort_and_filter_component_pool(  # type: ignore
#         self, components: CodeComponentPool
#     ) -> CodeComponentPool:
#         """Remove components from the component pool that are not up to standard. Usually these would be components with low ratings."""
#         raise NotImplementedError


# @dataclass
# class CodeEvaluation:
#     """Evaluation of a code component."""

#     feedback: str
#     rating: float | None


# @dataclass
# class CodeComponent:
#     """A code component of some system."""

#     content: CodeContent
#     output: CodeOutput
#     evaluation: CodeEvaluation


# CodeComponentPool = List[CodeComponent]
