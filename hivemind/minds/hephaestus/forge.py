"""Component forge."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, TypeVar, Generic

from hivemind.toolkit.text_formatting import dedent_and_strip


T = TypeVar("T")
NA_STRING = "N/A"


@dataclass
class Component(Generic[T]):
    """A component of some system."""

    # >
    # > some way to convert text representation of component to actual component


class ComponentType(Enum):
    """Type of component."""

    FUNCTION = "function"


ComponentPool = List[Component[T]]


@dataclass
class ComponentForge(Generic[T]):
    """Forge and maintain some component."""

    name: str
    component_type: ComponentType
    description: str
    usage_context: str
    top_k_variations: int
    dependencies: list[Component[T]] = field(default_factory=list)
    variations: list[Component[T]] = field(default_factory=list)

    @property
    def top_variations(self) -> ComponentPool[T]:
        """Get the top variations."""
        return self.variations[: self.top_k_variations]  # type: ignore

    @property
    def top_variations_printout(self) -> str:
        """Get the top variations as a string."""
        return (
            "\n".join(str(variation) for variation in self.top_variations) or NA_STRING
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
        You are a {component_type} forger, able to create and improve upon variations of a {component_type} based on specifications and feedback, given below.

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
        5. Study VARIATION TRAJECTORY, specifically the progression from worst to best variations. Focus on SCORE and FEEDBACK to discern patterns of improvements and persisting weaknesses. Learn from past iterations to avoid repeating errors and to build upon successes.
        6. Synthesize information from steps 1-5 to identify areas of potential enhancement. Cross-reference component's purpose, type, usage, dependencies, and historical trajectory. Look for patterns, gaps, or unexplored avenues that suggest opportunities for improvement.
        7. Generate a new component variation, incorporating insights gained from the reasoning steps. Ensure that this iteration addresses identified weaknesses, leverages strengths, and aligns with the component's purpose and usage context. Balance innovation with feasibility and compatibility with existing systems. Generate the component in the following block:
        ```end_of_reasoning_steps

        In your reply, you must include output from all STEPS of the reasoning process, in this format:
        1. {step_1_output}
        2. {step_2_output}
        [...]
        7. Component Variation:
        ```start_of_component_generation_output
        {component_generation_output}
        ```end_of_action_choice_output
        Any additional comments or thoughts can be added before or after the required output.
        """
        from hivemind.toolkit.models import super_creative_model, query_model
        from langchain.schema import HumanMessage, SystemMessage

        breakpoint()
        # > query

    def validate(self) -> bool:
        """Validate version ."""
        raise NotImplementedError

    def optimize(self) -> bool:
        """Optimize the prompt."""
        raise NotImplementedError


def test_generate_variation() -> None:
    """Test generate variation."""
    name = "process_employee_data"
    description = """
    A Python function that processes and normalizes datasets.
    Accepts a list of dictionaries, each representing a data record.
    Cleans data by removing empty fields and standardizing text (e.g., case normalization).
    Normalizes income data, scaling values to a proportion of 100000.
    Outputs a list of dictionaries with cleaned and normalized data.
    
    Parameters and Return Type:
    - Input: data_records (List[Dict[str, Union[str, int, float]]]) - A list of dictionaries, each containing various data fields.
    - Output: List[Dict[str, Union[str, float]]] - Processed data records.
    """
    usage_context = """
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
    forge = ComponentForge(
        name=name,
        component_type=ComponentType.FUNCTION,
        description=dedent_and_strip(description),
        usage_context=dedent_and_strip(usage_context),
        top_k_variations=3,
    )
    print(forge.generate_variation())


test_generate_variation()

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
