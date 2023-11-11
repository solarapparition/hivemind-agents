"""Structure for Aranea agents."""

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import NewType, NamedTuple
import random
import string

from hivemind.toolkit.text_formatting import generate_timestamp_id
from hivemind.toolkit.yaml_tools import yaml

AraneaId = NewType("AraneaId", str)
TaskId = NewType("TaskId", str)


def generate_aranea_id() -> AraneaId:
    """Generate an ID for an agent."""
    timestamp = generate_timestamp_id()
    random_str = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
    return AraneaId(f"aranea_{timestamp}_{random_str}")


class WorkValidation(NamedTuple):
    """Validation of work done by agent."""

    result: bool
    feedback: str


@dataclass
class Aranea:
    """A recursively auto-specializing agent."""

    task: str
    core_instructions: str
    learnings: str
    rank: int
    task_history: list[TaskId]
    id: AraneaId = field(default_factory=generate_aranea_id)
    serialized_attributes: tuple[str, ...] = (
        "id",
        "rank",
        "task_history",
        "core_instructions",
        "learnings",
    )
    serialization_dir: Path = Path(".")

    @property
    def serialization_location(self) -> Path:
        """Return the location where the agent should be serialized."""
        return self.serialization_dir / f"{self.id}.yml"

    def serialize(self) -> None:
        """Serialize the agent to YAML."""
        agent_data = {
            k: v for k, v in asdict(self).items() if k in self.serialized_attributes
        }
        yaml.dump(agent_data, self.serialization_location)

    def validate_work(self, task_data: str) -> WorkValidation:
        """Validate the work done by the agent."""
        print(
            f'Here is the message for the task completion by the agent:\n"{task_data}"'
        )
        while True:
            validation_input: str = (
                input("Was the task successfully completed? (True/False): ")
                .strip()
                .lower()
            )
            if validation_input in {"true", "false"}:
                validation_result: bool = validation_input == "true"
                break
            print("Invalid input. Please enter 'True' or 'False'.")

        feedback: str = input("Provide feedback: ")
        return WorkValidation(validation_result, feedback)


def test_serialize() -> None:
    """Test serialization."""
    test_dir = Path(".data/test/agents")
    agent = Aranea(
        task="task3",
        rank=0,
        task_history=[TaskId("task1"), TaskId("task2")],
        core_instructions="Primary directive here.",
        learnings="Adaptations from past tasks.",
        serialization_dir=test_dir,
    )
    agent.serialize()
    assert agent.serialization_location.exists()


def test() -> None:
    """Run tests."""
    # test_serialize()


if __name__ == "__main__":
    test()

# ....
# > workflow: get task -> ask questions -> extract subtask -> query subagent db -> select subagent -> delegate subtask -> ...
# Subtask Extraction: Broken down based on orthogonality and input/output footprint criteria
# > Includes 'user proxy' actions selection and 'assistant' subagent delegation
# Agent Ranking System: Prevents infinite loops, with higher-ranked agents delegating to lower-ranked ones
# > Vector database used for storing summaries of individual Aranea agents
# New agent initially has rank of their creator's rank minus one. Post first successful task completion, rank adjusts based on the highest rank of its subagents used
# Aranea agents equipped with an `escalate` function to route questions back to their caller during task execution
# If an Aranea agent receives an escalated question, it either responds or propagates the query up, eventually reaching the task owner
