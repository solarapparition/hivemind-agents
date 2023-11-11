"""Structure for Aranea agents."""

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import NewType
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
    return AraneaId(f"{timestamp}-{random_str}")


@dataclass
class Aranea:
    """A recursively specializing agent."""

    rank: int
    task_history: list[TaskId]
    core_instructions: str
    learnings: str
    serialization_location: Path = Path(".")
    id: AraneaId = field(default_factory=generate_aranea_id)
    serialized_attributes: tuple[str, ...] = (
        "id",
        "rank",
        "task_history",
        "core_instructions",
        "learnings",
    )

    def serialize(self) -> str:
        """Serialize the agent to YAML."""
        agent_data = {
            k: v for k, v in asdict(self).items() if k in self.serialized_attributes
        }
        return yaml.dump(agent_data, self.serialization_location)


# Usage
agent = Aranea(
    rank=0,
    task_history=[TaskId("task1"), TaskId("task2")],
    core_instructions="Primary directive here.",
    learnings="Adaptations from past tasks.",
)

agent.serialize()
