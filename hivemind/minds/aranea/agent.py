"""Structure for Aranea agents."""

from enum import Enum
from dataclasses import dataclass, asdict, field
from functools import cached_property
from pathlib import Path
from uuid import uuid4 as generate_uuid
from typing import (
    NewType,
    NamedTuple,
    Any,
    Self,
    Protocol,
    TypeVar,
    Callable,
    Coroutine,
)

from hivemind.toolkit.text_formatting import dedent_and_strip
from hivemind.toolkit.yaml_tools import yaml
from hivemind.toolkit.types import HivemindReply

BlueprintId = NewType("BlueprintId", str)
TaskId = NewType("TaskId", str)
RuntimeId = NewType("RuntimeId", str)
TaskHistory = list[TaskId]
IdTypeT = TypeVar("IdTypeT", BlueprintId, TaskId)


def generate_aranea_id(id_type: type[IdTypeT]) -> IdTypeT:
    """Generate an ID for an agent."""
    return id_type(f"{str(generate_uuid())}")


class TaskValidation(NamedTuple):
    """Validation of work done by agent."""

    result: bool
    feedback: str


class Role(Enum):
    """Role of an agent."""

    ORCHESTRATOR = "orchestrator"
    BOT = "bot"


@dataclass(frozen=True)
class Blueprint:
    """A blueprint for an Aranea agent."""

    name: str
    role: Role
    rank: int | None
    task_history: TaskHistory
    reasoning: str
    knowledge: str
    id: BlueprintId = field(default_factory=lambda: generate_aranea_id(BlueprintId))


class TaskWorkStatus(Enum):
    """Status of a task."""

    NEW = "new"
    DELEGATED = "delegated"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    IN_VALIDATION = "in validation"


class TaskDiscussionStatus(Enum):
    """Status of a discussion."""

    NONE = "none"
    AWAITING_RESPONSE_FROM_AGENT = "awaiting response from agent"
    AWAITING_RESPONSE_FROM_YOU = "awaiting response from you"


def replace_agent_id(
    text_to_replace: str, replace_with: str, agent_id: RuntimeId
) -> str:
    """Replace agent id with some other string."""
    return (
        text_to_replace.replace(f"agent {agent_id}", replace_with)
        .replace(f"Agent {agent_id}", replace_with.title())
        .replace(agent_id, replace_with)
    )


@dataclass
class Event:
    """An event in the event log."""

    timestamp: str
    description: str

    def __str__(self) -> str:
        return f"[{self.timestamp}] {self.description}"

    def to_str_with_pov(self, pov_id: RuntimeId) -> str:
        """String representation of the event with a point of view from a certain agent."""
        return replace_agent_id(str(self), "You", pov_id)


class TaskValidator(Protocol):
    """A validator of a task."""

    def validate(self, task: "Task") -> TaskValidation:
        """Validate the work done by the agent for a task."""
        raise NotImplementedError


class Human:
    """A human part of the hivemind. Can be used as either the owner or validator of a task."""

    name: str = "Human"

    @property
    def id(self) -> RuntimeId:
        """Runtime id of the agent."""
        return RuntimeId(self.name)

    def validate(self, task: "Task") -> TaskValidation:
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
        # if we're printing out the whole task list, assume these are subtasks
        return "\n".join([task.subtask_status_printout for task in self.tasks])

    def filter_by_status(self, status: TaskWorkStatus) -> Self:
        """Filter the task list by status."""
        return TaskList(
            tasks=[task for task in self.tasks if task.work_status == status]
        )


@dataclass
class EventLog:
    """A log of events within a task."""

    events: list[Event] = field(default_factory=list)

    def to_str_with_pov(self, pov_id: RuntimeId) -> str:
        """String representation of the event log with a point of view from a certain agent."""
        return "\n".join([event.to_str_with_pov(pov_id) for event in self.events])

    def recent(self, num_recent: int) -> Self:
        """Recent events."""
        return EventLog(events=self.events[-num_recent:])


@dataclass
class TaskDescription:
    """Description of a task."""

    information: str
    definition_of_done: str = ""

    def __str__(self) -> str:
        """String representation of the task description."""
        template = """
        Information:
        {information}

        Definition of Done:
        {definition_of_done}
        """
        return dedent_and_strip(template).format(
            information=self.information, definition_of_done=self.definition_of_done
        )


class Executor(Protocol):
    """An agent responsible for executing a task."""

    @property
    def id(self) -> RuntimeId:
        """Runtime id of the executor."""
        raise NotImplementedError

    async def execute(self, message: str | None = None) -> str:
        """Execute the subtask. Adds a message to the task's event log if provided, and adds own message to the event log at the end of execution."""
        raise NotImplementedError


@dataclass
class Task:
    """Holds information about a task."""

    description: TaskDescription
    owner_id: RuntimeId
    name: str | None = None
    validator: TaskValidator = field(default_factory=Human)
    id: TaskId = field(default_factory=lambda: generate_aranea_id(TaskId))
    executor: Executor | None = None
    notes: dict[str, str] = field(default_factory=dict)
    work_status: TaskWorkStatus = TaskWorkStatus.NEW
    discussion_status: TaskDiscussionStatus = TaskDiscussionStatus.NONE

    @cached_property
    def event_log(self) -> EventLog:
        """Event log for the task."""
        return EventLog()

    @cached_property
    def subtasks(self) -> TaskList:
        """Subtasks of the task."""
        return TaskList()

    def validate(self) -> TaskValidation:
        """Validate the work done by the agent."""
        return self.validator.validate(self)

    @property
    def main_status_printout(self) -> str:
        """String representation of the task as it would appear as a main task."""
        template = """
        Id: {id}
        Owner: {owner}
        Work Status: {status}
        Discussion Status: {discussion_status}

        Description:
        {description}
        """
        return dedent_and_strip(template).format(
            id=self.id,
            owner=self.owner_id,
            status=self.work_status,
            discussion_status=self.discussion_status,
            description=self.description,
        )

    def __str__(self) -> str:
        """String representation of task status."""
        return self.main_status_printout

    @property
    def executor_id(self) -> RuntimeId | None:
        """Id of the task's executor."""
        return self.executor.id if self.executor else None

    @property
    def subtask_status_printout(self) -> str:
        """String representation of task as it would appear as a subtask."""
        if self.work_status in {TaskWorkStatus.COMPLETED, TaskWorkStatus.CANCELLED}:
            template = """
            Id: {id}
            Name: {name}
            """
            return dedent_and_strip(template).format(
                id=self.id,
                name=self.name,
            )
        template = """
        Id: {id}
        Name: {name}
        Work Status: {status}
        Discussion Status: {discussion_status}
        """
        return dedent_and_strip(template).format(
            id=self.id,
            name=self.name,
            status=self.work_status,
            discussion_status=self.discussion_status,
            agent_id=self.executor_id,
        )


@dataclass
class CoreState:
    """Core runtime state of an agent."""

    agent_id: RuntimeId
    knowledge: str
    main_task: Task
    subtasks: TaskList
    template: str
    events: EventLog

    def __str__(self) -> str:
        """String representation of the core state."""
        completed_subtasks = str(
            self.subtasks.filter_by_status(TaskWorkStatus.COMPLETED)
        )
        cancelled_subtasks = str(
            self.subtasks.filter_by_status(TaskWorkStatus.CANCELLED)
        )
        in_validation_subtasks = str(
            self.subtasks.filter_by_status(TaskWorkStatus.IN_VALIDATION)
        )
        delegated_subtasks = str(
            self.subtasks.filter_by_status(TaskWorkStatus.DELEGATED)
        )
        new_subtasks = str(self.subtasks.filter_by_status(TaskWorkStatus.NEW))
        blocked_subtasks = str(self.subtasks.filter_by_status(TaskWorkStatus.BLOCKED))
        return dedent_and_strip(self.template).format(
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
class Orchestrator:
    """A recursively auto-specializing agent."""

    blueprint: Blueprint
    task: Task
    files_parent_dir: Path

    @property
    def id(self) -> RuntimeId:
        """Runtime id of the agent."""
        return RuntimeId(f"{self.blueprint_id}_{self.task.id}")

    @property
    def blueprint_id(self) -> BlueprintId:
        """Id of the agent."""
        return self.blueprint.id

    @property
    def rank(self) -> int | None:
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
    def role(self) -> Role:
        """Role of the agent."""
        return self.blueprint.role

    @property
    def core_template(self) -> str:
        """Template for the core state."""
        template = """
        ## MISSION:
        You are an expert task manager that specializes in managing the status and delegating the execution of a specific task and its subtasks to AGENTS that can execute those tasks while communicating with the TASK OWNER to gather requirements on the task. Your goal is to use your agents to complete the task as efficiently as possible.

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
        return dedent_and_strip(template)

    @property
    def core_state(self) -> CoreState:
        """Overall state of the agent."""
        return CoreState(
            agent_id=self.id,
            knowledge=self.knowledge,
            main_task=self.task,
            subtasks=self.task.subtasks,
            template=self.core_template,
            events=self.task.event_log,
        )

    @property
    def files_dir(self) -> Path:
        """Directory for files related to the agent."""
        return self.files_parent_dir / self.id

    @property
    def serialization_location(self) -> Path:
        """Return the location where the agent should be serialized."""
        return self.files_dir / "blueprint.yml"

    @property
    def output_dir(self) -> Path:
        """Output directory of the agent."""
        return self.files_dir / "output"

    @property
    def workspace_dir(self) -> Path:
        """Workspace directory of the agent."""
        return self.files_dir / "workspace"

    @property
    def name(self) -> str:
        """Name of the agent."""
        return self.blueprint.name

    def make_files_dirs(self) -> None:
        """Make the files directory for the agent."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def serialize(self) -> dict[str, Any]:
        """Serialize the agent to a dict."""
        assert self.rank is not None, "Rank must not be None when serializing."
        return asdict(self.blueprint)

    def save(self) -> None:
        """Serialize the agent to YAML."""
        yaml.dump(asdict(self.blueprint), self.serialization_location)

    @classmethod
    def load(
        cls,
        blueprint_location: Path,
        task: Task,
        files_parent_dir: Path,
    ) -> Self:
        """Deserialize an Aranea agent from a YAML file."""
        blueprint_data = yaml.load(blueprint_location)
        blueprint_data["task_history"] = tuple(blueprint_data["task_history"])
        return cls(
            blueprint=Blueprint(**blueprint_data),
            task=task,
            files_parent_dir=files_parent_dir,
        )

    async def execute(self, message: str | None = None) -> str:
        """Execute the subtask. Adds a message to the task's event log if provided, and adds own message to the event log at the end of execution."""
        raise NotImplementedError


@dataclass
class Reply:
    """A reply from the main agent."""

    content: str
    continue_func: Callable[[str], Coroutine[Any, Any, str]]

    async def continue_conversation(self, message: str) -> str:
        """Continue the conversation with a message."""
        return await self.continue_func(message)


def delegate(task: Task) -> Executor:
    """Delegate a task to a specific executor, or create a new one to handle the task."""
    raise NotImplementedError


@dataclass
class Aranea:
    """Main interfacing class for the agent."""

    output_dir: Path
    blueprint_dir: Path

    @cached_property
    def id(self) -> RuntimeId:
        """Runtime id of the agent."""
        return RuntimeId(str(generate_uuid()))

    @property
    def name(self) -> str:
        """Name of the agent."""
        return f"Aranea_{self.id}"

    async def run(self, message: str) -> HivemindReply:
        """Run the agent with a message, and a way to continue the conversation. Rerunning this method starts a new conversation."""
        task = Task(
            description=TaskDescription(message),
            owner_id=self.id,
        )
        reply_text = await (executor := delegate(task)).execute()

        async def continue_conversation(message: str) -> str:
            """Continue the conversation with a message."""
            return await executor.execute(message)

        return Reply(
            content=reply_text,
            continue_func=continue_conversation,
        )


# create new agent for when no matching agent
# ....
# > generic bot: coding agent
# > generic bot: oai assistant
# > generic bot: browser
# rank is calculated from blueprint, OR (if not available) from rank of subagents (1+max(subagent ranks))
# ....
# > test
# initial delegation: keep as dummy function for now
# choose action
# ....
# > next action execution > placeholder for `wait` action > every action has an output > event log for task also includes agent decisions and thoughts

"""action choice
{action_choice_reasoning} # attribute > reasoning step: think through what knowledge is relevant
{action_choice}
"extract_next_subtask", # extracts and delegates subtask, but doesn't start it; starting requires discussion with agent first
# > when extracting subtasks, always provide a name
"discuss", # doesn't send actual message yet, just brings up the context for sending message
"""

# > execute_task() must be async
"""action execution
{action_context}
{action_execution_reasoning}
{action_execution}

"extract_next_subtask", # extracts and delegates subtask, but doesn't start it; starting requires discussion with agent first
# > agent retrieval: success_rate/(1+rank/10)
> agent_search_strategy
> rank of none can access all agents
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


class NullTestTaskOwner:
    """Test task owner."""

    def answer_question(self, question: str) -> str:
        """Answer a question regarding the task."""
        return f"Answer to '{question}'"


null_test_task = Task(
    name="Some task",
    description=TaskDescription(
        information="Some information.", definition_of_done="Some definition of done."
    ),
    owner_id=NullTestTaskOwner(),
)

example_test_task = Task(
    name="Reorganize files on a flash drive",
    description=TaskDescription(
        information="The files on the flash drive are currently unorganized.",
        definition_of_done="N/A",
    ),
    owner_id=Human(),
)

TEST_DIR = ".data/test/agents"
test_blueprint = Blueprint(
    name="Test blueprint",
    rank=0,
    task_history=(TaskId("task1"), TaskId("task2")),
    reasoning="Primary directive here.",
    knowledge="Adaptations from past tasks.",
    output_dir=TEST_DIR,
)


def test_serialize() -> None:
    """Test serialization."""
    aranea_agent = Orchestrator(task=null_test_task, blueprint=test_blueprint)
    aranea_agent.save()
    assert aranea_agent.serialization_location.exists()


def test_deserialize() -> None:
    """Test deserialization."""
    # Setup: Serialize an agent to YAML for testing deserialization
    aranea_agent = Orchestrator(task=null_test_task, blueprint=test_blueprint)
    aranea_agent.save()

    # Test: Deserialize the agent from the YAML file
    deserialized_agent: Orchestrator = Orchestrator.load(
        aranea_agent.serialization_location, aranea_agent.task
    )

    # Verify: Deserialized agent matches the original
    assert deserialized_agent.blueprint.id == aranea_agent.blueprint.id
    assert deserialized_agent.rank == aranea_agent.rank
    assert deserialized_agent.task_history == aranea_agent.task_history
    assert deserialized_agent.reasoning == aranea_agent.reasoning
    assert deserialized_agent.knowledge == aranea_agent.knowledge
    assert deserialized_agent.task == aranea_agent.task


def test_id_generation() -> None:
    """Test that ids are generated as UUIDs."""
    blueprint_id = generate_aranea_id(BlueprintId)
    task_id = generate_aranea_id(TaskId)
    assert (
        len(blueprint_id) == 36 == len(str(task_id))
    ), f"{len(blueprint_id)} {len(task_id)}"


def test() -> None:
    """Run tests."""
    test_serialize()
    test_deserialize()
    test_id_generation()


if __name__ == "__main__":
    test()
