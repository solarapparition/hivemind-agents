"""Structure for Aranea agents."""

from enum import Enum
from dataclasses import dataclass, asdict, field
from functools import cached_property
from pathlib import Path
from typing import NewType, NamedTuple, Sequence, Any, Self, Protocol
import random
import string

from hivemind.toolkit.text_formatting import generate_timestamp_id, dedent_and_strip
from hivemind.toolkit.yaml_tools import yaml

BlueprintId = NewType("BlueprintId", str)
TaskId = NewType("TaskId", str)
RuntimeId = NewType("RuntimeId", str)
TaskHistory = Sequence[TaskId]


def generate_blueprint_id() -> BlueprintId:
    """Generate an ID for an agent."""
    timestamp = generate_timestamp_id()
    random_str = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
    return BlueprintId(f"aranea_{timestamp}_{random_str}")


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
    id: BlueprintId = field(default_factory=generate_blueprint_id)


class TaskOwner(Protocol):
    """The owner of a task that you can ask questions to."""

    def answer_question(self, question: str) -> str:
        """Answer a question regarding the task."""
        raise NotImplementedError


class TaskStatus(Enum):
    """Status of a task."""

    NEW = "new"
    DELEGATED = "delegated"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    IN_VALIDATION = "in validation"


class DiscussionStatus(Enum):
    """Status of a discussion."""

    NONE = "none"
    AWAITING_RESPONSE_FROM_AGENT = "awaiting response from agent"
    AWAITING_RESPONSE_FROM_YOU = "awaiting response from you"


@dataclass
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
class TaskList:
    """A list of tasks and their managment functionality."""

    tasks: list["Task"] = field(default_factory=list)

    def __str__(self) -> str:
        """String representation of the task list."""
        return "\n\n".join([str(task) for task in self.tasks])

    def filter_by_status(self, status: TaskStatus) -> Self:
        """Filter the task list by status."""
        return TaskList(tasks=[task for task in self.tasks if task.status == status])


@dataclass
class EventLog:
    """A log of events within a task."""

    events: list[Event] = field(default_factory=list)

    def to_str_with_pov(self, pov_id: RuntimeId) -> str:
        """String representation of the event log with a point of view from a certain agent."""
        print("TODO: EVENT LOG WITH POV")
        breakpoint()

    def recent(self, num_recent: int) -> Self:
        """Recent events."""
        return EventLog(events=self.events[-num_recent:])


@dataclass
class Task:
    """A task for an Aranea agent."""

    id: TaskId
    name: str
    description: str
    owner: TaskOwner
    agent: "Aranea" | None = None
    status: TaskStatus = TaskStatus.NEW
    discussion_status: DiscussionStatus = DiscussionStatus.NONE
    validator: TaskValidator = field(default_factory=HumanTaskValidator)

    @cached_property
    def event_log(self) -> EventLog:
        """Event log for the task."""
        return EventLog()

    @cached_property
    def subtasks(self) -> TaskList:
        """Subtasks of the task."""
        return TaskList()

    # @property
    # def status_printout(self) -> str:
    #     """Printout of the status of the task."""
    #     return f"{self.id}: {self.description}"

    def validate(self) -> TaskValidation:
        """Validate the work done by the agent."""
        return self.validator(self)

    def __str__(self) -> str:
        """String representation of the task."""
        template = """

        """
        print("TODO: STR REPRESENTATION OF TASK")
        breakpoint()
        # return f"{self.id}: {self.description}"


@dataclass
class CoreState:
    """Core runtime state of an agent."""

    agent_id: RuntimeId
    knowledge: str
    main_task: Task
    subtasks: TaskList
    events: EventLog

    def __str__(self) -> str:
        """String representation of the core state."""
        template = """
        ## MISSION:
        You are an expert task manager that specializes in managing the status and delegating the execution of a specific task and its subtasks to AGENTS that can execute those tasks while communicating with the TASK OWNER to gather requirements on the task. Your goal is to complete the task as efficiently as possible.

        ## KNOWLEDGE:
        In addition to the general background knowledge of your language model, you have the following, more specialized knowledge that may be relevant to the task at hand:
        ```start_of_knowledge
        {knowledge}
        ```end_of_knowledge

        ## MAIN TASK DESCRIPTION:
        Here is information about the main task you are currently working on:
        ```start_of_main_task_description
        {task_specification}
        ```end_of_main_task_description

        ## SUBTASK STATUSES:
        Subtasks are tasks that must be completed in order to complete the main task; you do not perform subtasks yourself, but instead delegate them to other agents. This list is NOT exhaustive; you may discover additional subtasks.

        Typically, tasks that are COMPLETED, CANCELLED, DELEGATED, or IN_VALIDATION do not need attention unless you discover information that changes the status of the subtask. In contrast, tasks that are NEW or BLOCKED will need action from you to start/continue execution.

        ### SUBTASKS (COMPLETED):
        These tasks have been validated as complete by agents; use this section as a reference for progress in the main task.
        ```start_of_completed_subtasks
        {completed_subtasks}
        ```end_of_completed_subtasks

        ### SUBTASKS (CANCELLED):
        You have previously cancelled these subtasks for various reason and they will not be done.
        ```start_of_cancelled_subtasks
        {cancelled_subtasks}
        ```end_of_cancelled_subtasks

        ### SUBTASKS (IN VALIDATION):
        These subtasks have been reported as completed by agents, but are still being validated by validators.
        ```start_of_in_validation_subtasks
        {in_validation_subtasks}
        ```end_of_in_validation_subtasks

        ### SUBTASKS (DELEGATED):
        These are subtasks that you have delegated to other agents and that are currently being executed by them.
        ```start_of_delegated_subtasks
        {delegated_subtasks}
        ```end_of_delegated_subtasks

        ### SUBTASKS (NEW):
        These subtasks are newly identified and not yet delegated to any agent:
        ```start_of_new_subtasks
        {new_subtasks}
        ```end_of_new_subtasks

        ### SUBTASKS (BLOCKED):
        These subtasks are blocked by some issue, and execution cannot continue until the issue is resolved, typically by discussing the blocker and/or creating a new subtask to resolve the blocker.
        ```start_of_blocked_subtasks
        {blocked_subtasks}
        ```end_of_blocked_subtasks

        ## RECENT EVENTS LOG:
        This is a log of recent events that have occurred during the execution of the task. This is NOT a complete log—use the task description and subtask statuses to get a complete picture of the current state of the work:
        ```start_of_recent_events_log
        {event_log}
        ```end_of_recent_events_log
        """
        template = dedent_and_strip(template)
        completed_subtasks = str(self.subtasks.filter_by_status(TaskStatus.COMPLETED))
        cancelled_subtasks = str(self.subtasks.filter_by_status(TaskStatus.CANCELLED))
        in_validation_subtasks = str(
            self.subtasks.filter_by_status(TaskStatus.IN_VALIDATION)
        )
        delegated_subtasks = str(self.subtasks.filter_by_status(TaskStatus.DELEGATED))
        new_subtasks = str(self.subtasks.filter_by_status(TaskStatus.NEW))
        blocked_subtasks = str(self.subtasks.filter_by_status(TaskStatus.BLOCKED))
        return template.format(
            knowledge=self.knowledge,
            task_specification=str(self.main_task),
            completed_subtasks=completed_subtasks,
            cancelled_subtasks=cancelled_subtasks,
            in_validation_subtasks=in_validation_subtasks,
            delegated_subtasks=delegated_subtasks,
            new_subtasks=new_subtasks,
            blocked_subtasks=blocked_subtasks,
            event_log=self.events.recent(10).to_str_with_pov(self.agent_id),
        )


@dataclass
class Aranea:
    """A recursively auto-specializing agent."""

    blueprint: Blueprint
    task: Task

    @property
    def blueprint_id(self) -> BlueprintId:
        """Id of the agent."""
        return self.blueprint.id

    @property
    def id(self) -> RuntimeId:
        """Runtime id of the agent."""
        return RuntimeId(f"{self.blueprint_id}_{self.task.id}")

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
    def core_state(self) -> CoreState:
        """Overall state of the agent."""
        return CoreState(
            agent_id=self.id,
            knowledge=self.knowledge,
            main_task=self.task,
            subtasks=self.task.subtasks,
            events=self.task.event_log,
        )

    @property
    def serialization_dir(self) -> Path:
        """Directory where the agent is serialized."""
        return Path(self.blueprint.serialization_dir)

    @property
    def serialization_location(self) -> Path:
        """Return the location where the agent should be serialized."""
        return self.serialization_dir / f"{self.blueprint.id}.yml"

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


# {task_specification} > task must have definition of done
# ....

"""structure
{subtask_statuses} > cancelled subtasks need cancellation reason
{action_log}
"""

"""action choice
{action_choice_reasoning} # attribute > reasoning step: think through what knowledge is relevant
{action_choice}
"extract_next_subtask", # extracts and delegates subtask, but doesn't start it; starting requires discussion with agent first
"discuss_with_agent",
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
# > event log for task also includes agent decisions and thoughts
"""

"""action execution
{action_context}
{action_execution_reasoning}
{action_execution}
"extract_next_subtask", # extracts and delegates subtask, but doesn't start it; starting requires discussion with agent first
> agent_search_strategy
"discuss_with_agent",
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
# > every action has an output
"""


##########

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
        id=generate_blueprint_id(),
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
        id=generate_blueprint_id(),
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
    assert deserialized_agent.blueprint.id == aranea_agent.blueprint.id
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
