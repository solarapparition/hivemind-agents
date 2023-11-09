"""Interface for optimizing a semantic component of a system (function, prompt, etc.) using a large language model."""

from typing import Sequence, Protocol, Any, NewType, Iterable
from itertools import chain
from dataclasses import dataclass
from abc import ABC, abstractmethod


EnvironmentOutput = NewType("Result", Any)
ComponentContent = NewType("ComponentContent", str)
OntologicalContext = NewType("OntologicalContext", str)


@dataclass
class Evaluation:
    """Evaluation of a semantic component."""

    rating: float
    feedback: str


@dataclass
class SemanticComponent:
    """A semantic component of a system."""

    content: ComponentContent
    output: EnvironmentOutput
    evaluation: Evaluation


ComponentPool = Sequence[SemanticComponent]


class SemanticComponentOptimizer(ABC):
    """Optimizer for a semantic component of a system."""

    @abstractmethod
    def generate_component_variations(
        self, component: ComponentContent, context: OntologicalContext
    ) -> list[ComponentContent]:
        """Attempt to create improved versions of a component."""

    @abstractmethod
    def run_environment(self, component: ComponentContent) -> EnvironmentOutput:
        """Run the environment with the component."""

    @abstractmethod
    def evaluate(
        self, component: ComponentContent, output: EnvironmentOutput
    ) -> Evaluation:
        """Evaluate the results of running the environment with the component."""

    @abstractmethod
    def ontological_context(self, component: ComponentContent) -> OntologicalContext:
        """Create the ontological context for the component."""

    def generate_improved_components(
        self,
        component: ComponentContent,
        rating_threshold: float = 0.0,
    ) -> ComponentPool:
        """Attempt to create improved versions of a component, gated by a threshold."""
        component_candidates = self.generate_component_variations(
            component, self.ontological_context(component)
        )
        outputs = (
            self.run_environment(candidate) for candidate in component_candidates
        )
        evaluations = (
            self.evaluate(candidate, output)
            for candidate, output in zip(component_candidates, outputs)
        )
        return [
            SemanticComponent(content=candidate, output=output, evaluation=evaluation)
            for candidate, output, evaluation in zip(
                component_candidates, outputs, evaluations
            )
            if evaluation.rating > rating_threshold
        ]

    @abstractmethod
    def select_improvement_candidates(self, components: ComponentPool) -> ComponentPool:
        """Select components from the component pool that would be improved."""

    @abstractmethod
    def filter_component_pool(
        self, components: Iterable[SemanticComponent]
    ) -> ComponentPool:
        """Remove components from the component pool that are not up to standard."""

    def improve_component_pool(
        self,
        components: Sequence[SemanticComponent],
    ) -> ComponentPool:
        """Improve a pool of components."""
        improvement_candidates = self.select_improvement_candidates(components)
        improved_components = chain(
            *(
                self.generate_improved_components(candidate.content)
                for candidate in improvement_candidates
            )
        )
        return self.filter_component_pool(chain(components, improved_components))
