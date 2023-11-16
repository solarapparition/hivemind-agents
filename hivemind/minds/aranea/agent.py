"""Structure for Aranea agents."""

from enum import Enum
from dataclasses import dataclass, asdict, field
from functools import cached_property
from pathlib import Path
from typing import NewType, NamedTuple, Sequence, Any, Self, Protocol
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


class TaskValidation(NamedTuple):
    """Validation of work done by agent."""

    result: bool
    feedback: str


@dataclass(frozen=True)
class Blueprint:
    """A blueprint for an Aranea agent."""

    rank: int
    task_history: TaskHistory
    reasoning: str
    knowledge: str
    serialization_dir: str
    id: AraneaId = field(default_factory=generate_aranea_id)


class TaskOwner(Protocol):
    """The owner of a task that you can ask questions to."""

    def answer_question(self, question: str) -> str:
        """Answer a question regarding the task."""
        raise NotImplementedError


class TaskStatus(Enum):
    """Status of a task."""

    NEW = "new"
    IN_PROGRESS = "in progress"
    COMPLETED = "completed"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    IN_VALIDATION = "in validation"


class Event:
    """An event in the event log."""

    ...


class TaskValidator(Protocol):
    """A validator of a task."""

    def __call__(self, task: "Task") -> TaskValidation:
        """Validate the work done by the agent for a task."""
        raise NotImplementedError


class HumanTaskValidator:
    """A human validator of a task."""

    def __call__(self, task: "Task") -> TaskValidation:
        """Validate the work done by the agent."""
        print(f'Please validate the following task:\n"{str(task)}"')
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
        return TaskValidation(validation_result, feedback)


@dataclass
class Task:
    """A task for an Aranea agent."""

    id: TaskId
    description: str
    owner: TaskOwner
    status: TaskStatus = TaskStatus.NEW
    subagent_id: AraneaId | None = None
    status_details: list[str] = field(default_factory=list)
    in_discussion: bool = False
    event_log: list[Event] = field(default_factory=list)
    subtasks: list[Self] = field(default_factory=list)
    validator: TaskValidator = field(default_factory=HumanTaskValidator)

    @property
    def status_printout(self) -> str:
        """Printout of the status of the task."""
        return f"{self.id}: {self.description}"

    def validate(self) -> TaskValidation:
        """Validate the work done by the agent."""
        return self.validator(self)

    def __str__(self) -> str:
        """String representation of the task."""
        print("TODO: STR REPRESENTATION OF TASK")
        breakpoint()
        # return f"{self.id}: {self.description}"


@dataclass
class CoreState:
    """Core runtime state of an agent."""

    knowledge: str
    task_specification: str
    subtask_statuses: list[str]
    task_event_log: list[Event]


@dataclass
class Aranea:
    """A recursively auto-specializing agent."""

    blueprint: Blueprint
    task: Task

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
    def reasoning(self) -> str:
        """Instructions for the agent."""
        return self.blueprint.reasoning

    @property
    def knowledge(self) -> str:
        """Learnings from past tasks."""
        return self.blueprint.knowledge

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
    def load(cls, blueprint_location: Path, task: Task) -> Self:
        """Deserialize an Aranea agent from a YAML file."""
        blueprint_data = yaml.load(blueprint_location)
        blueprint_data["task_history"] = tuple(blueprint_data["task_history"])
        return cls(blueprint=Blueprint(**blueprint_data), task=task)

    def __hash__(self) -> int:
        """Hash the agent."""
        return hash(self.blueprint)

    def ask_question(self, question: str) -> str:
        """Ask a question regarding the task to the owner of the task."""
        return self.task.owner.answer_question(question)

    @property
    def subtask_statuses(self) -> list[str]:
        """Statuses of subtasks."""
        return [subtask.status_printout for subtask in self.task.subtasks]

    @property
    def core_state(self) -> CoreState:
        """Overall state of the agent."""
        return CoreState(
            knowledge=self.knowledge,
            task_specification=self.task.description,
            subtask_statuses=self.subtask_statuses,
            task_event_log=self.task.event_log,
        )


"""
{knowledge} # attribute
{task_specification}
{subtask_statuses}
{action_log}
"""
"""
{reasoning} # attribute
{next_action}
"""
"""
{reasoning}
{action_context}
{action_execution}
"""


"""
subtask_format = {
    "id": "subtask_1",
    "status": "new", # "new", "in progress", "completed", "paused", "cancelled", "blocked", "in validation", "in discussion"
    "discussion_status": "discussion in progress", # "no discussion in progress"
    "event_log": [], # not displayed to agent
    "description": "",
    "blockers": [],
    "subagent_id": "subagent_1",
}
"""

"""actions:
"new_subtask",
"discuss_with_subagent",
    # these are all options for individual discussion messages
    "informational",
    "start_subtask",
    "pause_subtask",
    "cancel_subtask",
    "resume_subtask",
"discuss_with_task_owner", # doesn't send actual message yet, just brings up the context for sending message
    "task_completed",
    "task_blocked",
    "query",
"wait",
"""


# Action Process
# > multistep process
# > step 1: reasoning
# > step 2: next action
# > step 3: action execution


# > difference between blueprint id and runtime id


# > maybe subagents should bid on task? # maybe offered task, then either accept or decline # check if there is theoretical framework for this

# > new: never started; set by agent on creation
# > in progress: currently being done by subagent; set after agent decides to start subtask
# > in validation: currently being validated by validator; set when subagent has reported task completion
# > completed: done by subagent; set by system when subtask has been validated
# > cancelled: will not be done; set by agent
# > blocked: cannot be done by subagent due to some issue; set by subagent
# > paused: temporarily stopped; set by agent
# > in discussion: set automatically when someone starts a discussion; ends when the discussion starts decides to end it


# > status: stopped, in progress, completed, new, paused, subagent id, subtask id
# > message type: "information", "stoppage"
"""
Event Log Analysis
Review event log.
Identify task progress, status of subtasks.
Detect patterns, anomalies.

Information Gathering Necessity Assessment
Evaluate if current information sufficient for decision-making.
If insufficient, assess which aspects require clarification.

Action Identification for Subtask Management
Check status of each subtask.
Determine available actions: extract, start, cancel.
Consider dependencies, impacts of each action.
"""
"""
Communication Needs Evaluation
Assess necessity of information transfer to/from subagents.
Identify specific aspects needing communication.
Evaluate urgency, relevance of communication.

Decision Prioritization
Prioritize actions based on urgency, importance.
Consider task deadlines, constraints.
Balance between information gathering, subtask management, communication.

Outcome Prediction
Anticipate results of each potential action.
Consider best and worst-case scenarios.

Action Selection
Choose action based on analysis, priorities, predictions.
Ensure alignment with overall task objectives.
"""

# > gather information
# > extract/start/stop subtask # depending on status
# > answer question

# questioner: ask unknowns
# decider: decide on next action
# interpreter: give explanations of unknowns

# actions: ask, answer, execute


# user_proxy = autogen.UserProxyAgent(
#    name="User_proxy",
#    system_message="A human admin.",
#    code_execution_config={"last_n_messages": 2, "work_dir": "groupchat"},
#    human_input_mode="TERMINATE",
#    default_auto_reply="Is there anything else left to do?",
# )
# tm = autogen.AssistantAgent(
#     name="Task_manager",
#     system_message="Keeps the group on track by reminding everyone of what needs to be done next, repeating instructions/code if necessary. Reply TERMINATE if the original task is done.",
#     llm_config=llm_config,
# )
# coder = autogen.AssistantAgent(
#     name="Coder",
#     llm_config=llm_config,
# )
# pm = autogen.AssistantAgent(
#     name="Product_manager",
#     system_message="Creative in software product ideas.",
#     llm_config=llm_config,
# )

# groupchat = autogen.GroupChat(agents=[user_proxy, coder, pm, tm], messages=[], max_round=12)
# manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# user_proxy.initiate_chat(manager, message="Find a latest paper about gpt-4 on arxiv and find its potential applications in software.")

# Testing: will need to be converted to Pytest eventually


class TestTaskOwner:
    """Test task owner."""

    def answer_question(self, question: str) -> str:
        """Answer a question regarding the task."""
        return f"Answer to '{question}'"


test_task = Task(
    id=TaskId("task3"),
    description="A task for an Aranea agent.",
    owner=TestTaskOwner(),
)


def test_serialize() -> None:
    """Test serialization."""
    test_dir = ".data/test/agents"
    blueprint = Blueprint(
        id=generate_aranea_id(),
        rank=0,
        task_history=(TaskId("task1"), TaskId("task2")),
        reasoning="Primary directive here.",
        knowledge="Adaptations from past tasks.",
        serialization_dir=test_dir,
    )
    aranea_agent = Aranea(task=test_task, blueprint=blueprint)
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
        reasoning="Primary directive here.",
        knowledge="Adaptations from past tasks.",
        serialization_dir=test_dir,
    )
    aranea_agent = Aranea(task=test_task, blueprint=blueprint)
    aranea_agent.save()

    # Test: Deserialize the agent from the YAML file
    deserialized_agent: Aranea = Aranea.load(
        aranea_agent.serialization_location, aranea_agent.task
    )

    # Verify: Deserialized agent matches the original
    assert deserialized_agent.id == aranea_agent.id
    assert deserialized_agent.rank == aranea_agent.rank
    assert deserialized_agent.task_history == aranea_agent.task_history
    assert deserialized_agent.reasoning == aranea_agent.reasoning
    assert deserialized_agent.knowledge == aranea_agent.knowledge
    assert deserialized_agent.task == aranea_agent.task


def test_ask_question() -> None:
    """Ask a question to the task owner."""


def test() -> None:
    """Run tests."""
    # test_serialize()
    # test_deserialize()


if __name__ == "__main__":
    test()
