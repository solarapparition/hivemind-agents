"""Structure for Aranea agents."""

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import NewType, NamedTuple, Sequence, Any, Self
import random
import string

from hivemind.toolkit.text_formatting import generate_timestamp_id
from hivemind.toolkit.yaml_tools import yaml

AraneaId = NewType("AraneaId", str)
TaskId = NewType("TaskId", str)
TaskHistory = Sequence[TaskId]


def generate_aranea_id() -> AraneaId:
    """Generate an ID for an agent."""
    timestamp = generate_timestamp_id()
    random_str = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
    return AraneaId(f"aranea_{timestamp}_{random_str}")


class WorkValidation(NamedTuple):
    """Validation of work done by agent."""

    result: bool
    feedback: str


@dataclass(frozen=True)
class Blueprint:
    """A blueprint for an Aranea agent."""

    rank: int
    task_history: TaskHistory
    instructions: str
    learnings: str
    serialization_dir: str
    id: AraneaId = field(default_factory=generate_aranea_id)


@dataclass
class Aranea:
    """A recursively auto-specializing agent."""

    blueprint: Blueprint
    task: str

    @property
    def id(self) -> AraneaId:
        """Id of the agent."""
        return self.blueprint.id

    @property
    def rank(self) -> int:
        """Rank of the agent."""
        return self.blueprint.rank

    @property
    def task_history(self) -> TaskHistory:
        """History of tasks completed by the agent."""
        return self.blueprint.task_history

    @property
    def instructions(self) -> str:
        """Instructions for the agent."""
        return self.blueprint.instructions

    @property
    def learnings(self) -> str:
        """Learnings from past tasks."""
        return self.blueprint.learnings

    @property
    def serialization_dir(self) -> Path:
        """Directory where the agent is serialized."""
        return Path(self.blueprint.serialization_dir)

    @property
    def serialization_location(self) -> Path:
        """Return the location where the agent should be serialized."""
        return self.serialization_dir / f"{self.id}.yml"

    def serialize(self) -> dict[str, Any]:
        """Serialize the agent to a dict."""
        return asdict(self.blueprint)

    def save(self) -> None:
        """Serialize the agent to YAML."""
        yaml.dump(asdict(self.blueprint), self.serialization_location)

    @classmethod
    def load(cls, blueprint_location: Path, task: str) -> Self:
        """Deserialize an Aranea agent from a YAML file."""
        blueprint_data = yaml.load(blueprint_location)
        blueprint_data["task_history"] = tuple(blueprint_data["task_history"])
        return cls(blueprint=Blueprint(**blueprint_data), task=task)

    def __hash__(self) -> int:
        """Hash the agent."""
        return hash(self.blueprint)

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
    test_dir = ".data/test/agents"
    blueprint = Blueprint(
        id=generate_aranea_id(),
        rank=0,
        task_history=(TaskId("task1"), TaskId("task2")),
        instructions="Primary directive here.",
        learnings="Adaptations from past tasks.",
        serialization_dir=test_dir,
    )
    aranea_agent = Aranea(task="task3", blueprint=blueprint)
    aranea_agent.save()
    assert aranea_agent.serialization_location.exists()


def test_deserialize() -> None:
    """Test deserialization."""
    # Setup: Serialize an agent to YAML for testing deserialization
    test_dir = ".data/test/agents"
    blueprint = Blueprint(
        id=generate_aranea_id(),
        rank=0,
        task_history=(TaskId("task1"), TaskId("task2")),
        instructions="Primary directive here.",
        learnings="Adaptations from past tasks.",
        serialization_dir=test_dir,
    )
    aranea_agent = Aranea(task="task3", blueprint=blueprint)
    aranea_agent.save()

    # Test: Deserialize the agent from the YAML file
    deserialized_agent: Aranea = Aranea.load(
        aranea_agent.serialization_location, aranea_agent.task
    )

    # Verify: Deserialized agent matches the original
    assert deserialized_agent.id == aranea_agent.id
    assert deserialized_agent.rank == aranea_agent.rank
    assert deserialized_agent.task_history == aranea_agent.task_history
    assert deserialized_agent.instructions == aranea_agent.instructions
    assert deserialized_agent.learnings == aranea_agent.learnings
    assert deserialized_agent.task == aranea_agent.task


def test_ask_question() -> None:
    """Ask a question to the task owner."""


def test() -> None:
    """Run tests."""
    # test_serialize()
    # test_deserialize()


if __name__ == "__main__":
    test()
