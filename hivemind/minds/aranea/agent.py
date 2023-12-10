"""Structure for Aranea agents."""

import shelve
import os
import asyncio

from enum import Enum
from dataclasses import dataclass, asdict, field
from functools import cached_property
from pathlib import Path
from uuid import uuid4 as generate_uuid
from typing import (
    Iterable,
    Iterator,
    MutableMapping,
    NewType,
    NamedTuple,
    Any,
    Self,
    Protocol,
    Sequence,
    Set,
    TypeVar,
    Callable,
    Coroutine,
)

from langchain.schema import SystemMessage, BaseMessage
from colorama import Fore

from hivemind.config import configure_langchain_cache
from hivemind.toolkit.models import super_creative_model, precise_model, query_model
from hivemind.toolkit.text_extraction import ExtractionError, extract_blocks
from hivemind.toolkit.text_formatting import dedent_and_strip
from hivemind.toolkit.yaml_tools import yaml
from hivemind.toolkit.timestamp import utc_timestamp

BlueprintId = NewType("BlueprintId", str)
TaskId = NewType("TaskId", str)
EventId = NewType("EventId", str)
RuntimeId = NewType("RuntimeId", str)
TaskHistory = list[TaskId]
IdTypeT = TypeVar("IdTypeT", BlueprintId, TaskId, EventId)

AGENT_COLOR = Fore.MAGENTA
VERBOSE = True
NONE = "None"


class Concepts(Enum):
    """Concepts for Aranea agents."""

    MAIN_TASK_OWNER = "MAIN TASK OWNER"
    RECENT_EVENTS_LOG = "RECENT EVENTS LOG"


def print_messages(messages: Sequence[BaseMessage]) -> str:
    """Print LangChain messages."""
    return "\n\n---\n\n".join(
        [f"[{message.type.upper()}]:\n\n{message.content}" for message in messages]
    )


def generate_aranea_id(id_type: type[IdTypeT]) -> IdTypeT:
    """Generate an ID for an agent."""
    return id_type(f"{str(generate_uuid())}")


class ValidationResult(NamedTuple):
    """Validation of work done by agent."""

    valid: bool
    feedback: str


class Role(Enum):
    """Role of an agent."""

    ORCHESTRATOR = "orchestrator"
    BOT = "bot"


@dataclass
class Reasoning:
    """Reasoning instructions for an agent."""

    default_action_choice: str | None = None
    subtask_extraction: str | None = None


@dataclass
class Blueprint:
    """A blueprint for an Aranea agent."""

    name: str
    role: Role
    rank: int | None
    task_history: TaskHistory
    reasoning: Reasoning
    knowledge: str
    recent_events_size: int
    id: BlueprintId = field(default_factory=lambda: generate_aranea_id(BlueprintId))


class TaskWorkStatus(Enum):
    """Status of the work for a task."""

    IDENTIFIED = "IDENTIFIED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    BLOCKED = "BLOCKED"
    IN_VALIDATION = "IN_VALIDATION"


class TaskEventStatus(Enum):
    """Status of the events for a task."""

    NONE = "None"
    AWAITING_EXECUTOR = "you are awaiting response from the subtask executor"
    AWAITING_OWNER = "the subtask executor is awaiting a response from you"


def replace_agent_id(
    text_to_replace: str, replace_with: str, agent_id: RuntimeId
) -> str:
    """Replace agent id with some other string."""
    return (
        text_to_replace.replace(f"agent {agent_id}", replace_with)
        .replace(f"Agent {agent_id}", replace_with.title())
        .replace(agent_id, replace_with)
    )


@dataclass(frozen=True)
class MessageData:
    """Data for a message."""

    sender: RuntimeId
    recipient: RuntimeId
    content: str

    def __str__(self) -> str:
        return f"{self.sender} (to {self.recipient}): {self.content}"


@dataclass(frozen=True)
class SubtaskIdentificationData:
    """Data for identifying a subtask."""

    identifier: RuntimeId
    subtask: str
    validation_result: ValidationResult

    def __str__(self) -> str:
        if self.validation_result.valid:
            return (
                f"{self.identifier}: Successfully identified subtask: `{self.subtask}`"
            )
        return f'{self.identifier}: Attempted to identify subtask `{self.subtask}`, but the validator did not approve the subtask, with the following feedback: "{self.validation_result.feedback}"'


EventData = MessageData | SubtaskIdentificationData


@dataclass
class Event:
    """An event in the event log."""

    data: EventData
    timestamp: str = field(default_factory=utc_timestamp)
    id: EventId = field(default_factory=lambda: generate_aranea_id(EventId))

    def __str__(self) -> str:
        # return f"[{self.timestamp}] {self.data}"
        return f"{self.data}"

    def to_str_with_pov(
        self, pov_id: RuntimeId, other_id: RuntimeId, other_name: str
    ) -> str:
        """String representation of the event with a point of view from a certain executor."""
        event_printout = replace_agent_id(str(self), "You", pov_id)
        return replace_agent_id(event_printout, other_name, other_id)

    def serialize(self) -> dict[str, Any]:
        """Serialize the event."""
        return asdict(self)

    def __repr__(self) -> str:
        """String representation of the event."""
        return str(self.serialize())


class WorkValidator(Protocol):
    """A validator of a task."""

    def validate(self, prompt: str) -> ValidationResult:
        """Validate the work done by an executor for a task."""
        raise NotImplementedError


@dataclass
class Human:
    """A human part of the hivemind. Can be slotted into various specialized roles for tasks that the agent can't yet handle autonomously."""

    name: str = "Human"
    reply_cache: MutableMapping[str, str] | None = None
    thread: list[str] = field(default_factory=list)

    @property
    def id(self) -> RuntimeId:
        """Runtime id of the human."""
        return RuntimeId(self.name)

    def respond_manually(self) -> str:
        """Get manual response from the human."""
        return input("Enter your response: ").strip()

    def respond_using_cache(self, reply_cache: MutableMapping[str, str]) -> str:
        """Get cached reply based on thread."""
        if reply := reply_cache.get(str(self.thread)):
            print(f"Cached reply found: {reply}")
            return reply
        if reply := self.respond_manually():
            reply_cache.update({str(self.thread): reply})
        return reply

    def advise(self, prompt: str) -> str:
        """Get input from the human."""
        print(prompt)
        self.thread.append(prompt)
        self.thread.append(
            reply := (
                self.respond_using_cache(self.reply_cache)
                if self.reply_cache is not None
                else self.respond_manually()
            )
        )
        return reply

    def validate(self, prompt: str) -> ValidationResult:
        """Validate the work done by an executor."""
        prompt += "\n\nPlease validate the work as described above (y/n): "
        while True:
            validation_input: str = self.advise(prompt).strip().lower()
            if validation_input in {"y", "n"}:
                validated: bool = validation_input == "y"
                break
            print("Invalid input. Please enter 'y' or 'n'.")
        feedback: str = "" if validated else self.advise("Provide feedback: ")
        return ValidationResult(validated, feedback)


@dataclass
class TaskList:
    """A list of tasks and their managment functionality."""

    items: list["Task"] = field(default_factory=list)

    def __str__(self) -> str:
        """String representation of the task list."""
        # if we're printing out the whole task list, assume these are subtasks
        return "\n".join([task.subtask_status_printout for task in self.items]) or NONE

    def __iter__(self) -> Iterator["Task"]:
        """Iterate over the task list."""
        return iter(self.items)

    def filter_by_status(self, status: TaskWorkStatus) -> Self:
        """Filter the task list by status."""
        return TaskList(
            items=[task for task in self.items if task.work_status == status]
        )


@dataclass
class EventLog:
    """A log of events within a task."""

    events: list[Event] = field(default_factory=list)

    @property
    def last_event(self) -> Event | None:
        """Last event in the event log."""
        return self.events[-1] if self.events else None

    def to_str_with_pov(
        self, pov_id: RuntimeId, other_id: RuntimeId, other_name: str
    ) -> str:
        """String representation of the event log with a point of view from a certain executor."""
        return (
            "\n".join(
                [
                    event.to_str_with_pov(pov_id, other_id, other_name)
                    for event in self.events
                ]
            )
            if self.events
            else NONE
        )

    def recent(self, num_recent: int) -> Self:
        """Recent events."""
        return EventLog(events=self.events[-num_recent:])

    def add(self, *events: Event) -> None:
        """Add events to the event log."""
        self.events.extend(events)


@dataclass
class TaskDescription:
    """Description of a task."""

    information: str
    definition_of_done: str | None = None

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

    @property
    def rank(self) -> int | None:
        """Rank of the executor."""
        raise NotImplementedError

    def accepts(self, task: "Task") -> bool:
        """Decides whether the executor accepts a task."""
        raise NotImplementedError

    async def execute(self, message: str | None = None) -> str:
        """Execute the subtask. Adds a message to the task's event log if provided, and adds own message to the event log at the end of execution."""
        raise NotImplementedError


@dataclass
class Task:
    """Holds information about a task."""

    description: TaskDescription
    owner_id: RuntimeId
    rank_limit: int | None
    validator: WorkValidator
    name: str | None = None
    id: TaskId = field(default_factory=lambda: generate_aranea_id(TaskId))
    executor: Executor | None = None
    notes: dict[str, str] = field(default_factory=dict)
    work_status: TaskWorkStatus = TaskWorkStatus.IDENTIFIED
    event_status: TaskEventStatus = TaskEventStatus.NONE

    @cached_property
    def event_log(self) -> EventLog:
        """Event log for the task."""
        return EventLog()

    @cached_property
    def subtasks(self) -> TaskList:
        """Subtasks of the task."""
        return TaskList()

    @property
    def main_status_printout(self) -> str:
        """String representation of the task as it would appear as a main task."""
        # Id: {id}
        # Owner: {owner}
        # Work Status: {status}
        # Event Status: {event_status}

        # template = """
        # {description}
        # """
        # return dedent_and_strip(template).format(
        # id=self.id,
        # owner=self.owner_id,
        # status=self.work_status.value,
        # event_status=self.event_status.value,
        # description=self.description,
        # )
        return str(self.description)

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
        Work Status: {work_status}
        Event Status: {event_status}
        """
        return dedent_and_strip(template).format(
            id=self.id,
            name=self.name,
            work_status=self.work_status.value,
            event_status=self.event_status.value,
        )


@dataclass
class CoreState:
    """Core runtime state of an Aranea subagent."""

    id: RuntimeId
    knowledge: str
    main_task: Task
    subtasks: TaskList
    template: str

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
            self.subtasks.filter_by_status(TaskWorkStatus.IN_PROGRESS)
        )
        blocked_subtasks = str(self.subtasks.filter_by_status(TaskWorkStatus.BLOCKED))
        return dedent_and_strip(self.template).format(
            knowledge=self.knowledge,
            task_specification=str(self.main_task),
            completed_subtasks=completed_subtasks,
            cancelled_subtasks=cancelled_subtasks,
            in_validation_subtasks=in_validation_subtasks,
            delegated_subtasks=delegated_subtasks,
            # identified_subtasks=identified_subtasks,
            blocked_subtasks=blocked_subtasks,
        )


class ActionName(Enum):
    """Names of actions available to the orchestrator."""

    IDENTIFY_NEW_SUBTASK = "IDENTIFY_NEW_SUBTASK"
    START_DISCUSSION_FOR_SUBTASK = "START_DISCUSSION_FOR_SUBTASK"
    MESSAGE_TASK_OWNER = "ASK_MAIN_TASK_OWNER"
    REPORT_MAIN_TASK_COMPLETE = "REPORT_MAIN_TASK_COMPLETE"
    WAIT = "WAIT"


ORCHESTRATOR_CONCEPTS = f"""
- ORCHESTRATOR: the agent that is responsible for managing the execution of a main task and managing the statuses of its subtasks, while communicating with the task's owner to gather required information for the task. The orchestrator must communicate with both the task owner and subtask executors to complete the main task as efficiently as possible.
- MAIN TASK: the main task that the orchestrator is responsible for managing, which it does by identifying subtasks and providing support for specialized executor agents for the subtasks.
- SUBTASK: a task that must be executed in order to complete the main task. The orchestrator does NOT execute subtasks itself; instead, it facilitates the resolution of subtasks by making high-level decisions regarding each subtask in the context of the overall task and providing support for the subtask executors.
- SUBTASK STATUS: the status of subtasks that have already been identified. The status of a subtask can be one of the following:
  - {TaskWorkStatus.BLOCKED.value}: the subtask is blocked by some issue, and execution cannot continue until the issue is resolved, typically by discussing the blocker and/or identifying a new subtask to resolve the blocker.
  - {TaskWorkStatus.IN_PROGRESS.value}: the subtask is currently being executed by a subtask executor.
  - {TaskWorkStatus.IN_VALIDATION.value}: the subtask has been reported as completed by its executor, but is still being validated by a validator. Validation happens automatically and does not require or action from the orchestrator.
  - {TaskWorkStatus.COMPLETED.value}: the subtask has been validated as complete by a validator. Completed subtasks provide a record of overall successful progress for the main task.
  - {TaskWorkStatus.CANCELLED.value}: the subtask has been cancelled for various reason and will not be done.
- SUBTASK EXECUTOR: an agent that is responsible for executing a subtask. Subtask executors specialize in executing certain types of tasks; whenever a subtask is identified, an executor is automatically assigned to it without any action required from the orchestrator.
- {Concepts.MAIN_TASK_OWNER.value}: the one who requested the main task to be done. The orchestrator must communicate with the task owner to gather background information required to complete the main task.
""".strip()
#   - {TaskWorkStatus.IDENTIFIED.value}: the subtask has been newly identified and has not started execution yet.

ORCHESTRATOR_INFORMATION_SECTIONS = f"""
- KNOWLEDGE: background knowledge relating to the orchestrator's area of specialization. The information may or may not be relevant to the specific main task, but is provided as support for the orchestrator's decisionmaking.
- MAIN TASK DESCRIPTION: a description of information about the main task that the orchestrator has learned so far from the {Concepts.MAIN_TASK_OWNER.value}. This may NOT be a complete description of the main task, so the orchestrator must always take into account if there is enough information for performing its actions. Additional information may also be in the {Concepts.RECENT_EVENTS_LOG.value}, as messages from the main task owner.
- SUBTASKS: a list of all subtasks that have been identified by the orchestrator so far; for each one, there is a high-level description of what must be done, as well as the subtask's status. This is not an exhaustive list of all required subtasks for the main task; there may be additional subtasks that are required. This list is automatically maintained and updated by a background process.
- {Concepts.RECENT_EVENTS_LOG.value}: a log of recent events that have occurred during the execution of the task. This can include status updates for subtasks, messages from the main task owner, and the orchestrator's own previous thoughts/decisions.
""".strip()


def query_and_extract_reasoning(
    messages: Sequence[SystemMessage], preamble: str, printout: bool
) -> str:
    """Query the model and extract the reasoning process."""
    if printout:
        result = query_model(
            model=super_creative_model,
            messages=messages,
            preamble=preamble,
            color=AGENT_COLOR,
            printout=printout,
        )
    else:
        result = query_model(
            model=super_creative_model,
            messages=messages,
            printout=printout,
        )
    if not (extracted_result := extract_blocks(result, "start_of_reasoning_process")):
        raise ExtractionError("Could not extract reasoning process from the result.")
    return extracted_result[0]


BASE_ORCHESTRATOR_INFO = f"""
## CONCEPTS:
{ORCHESTRATOR_CONCEPTS}

## ORCHESTRATOR INFORMATION SECTIONS:
By default, the orchestrator has access to the following information. Note that all information here is read-only; while identifying new subtasks, the orchestrator cannot modify any of the information here.
{ORCHESTRATOR_INFORMATION_SECTIONS}
""".strip()


class ActionModeName(Enum):
    """States of an action."""

    DEFAULT = "DEFAULT"
    SUBTASK_DISCUSSION = "SUBTASK DISCUSSION"


def generate_action_reasoning(
    role: Role, state: ActionModeName, actions: str, printout: bool = False
) -> str:
    """Generate reasoning for choosing an action."""
    if role == Role.ORCHESTRATOR and state == ActionModeName.DEFAULT:
        context = """
        ## MISSION:
        You are the instructor for an AI task orchestration agent. Your purpose is to provide step-by-step guidance for the agent to think through what it must do next.

        {base_info}

        ## ORCHESTRATOR ACTIONS:
        In its default state, the orchestrator can perform the following actions:
        {actions}
        """
        request = f"""
        ## REQUEST FOR YOU:
        Provide a step-by-step, robust reasoning process for the orchestrator to sequentially think through the information it has access to so that it has the appropriate mental context for deciding what to do next. These steps provide the internal thinking that an intelligent agent must go through so that they have all the relevant information on top of mind. Some things to note:
        - The final action that the orchestrator decides on MUST be one of the ORCHESTRATOR ACTIONS described above. The orchestrator cannot perform any other actions.
        - Assume that the orchestrator has access to the information described above, but no other information, except for general world knowledge that is available to a standard LLM like GPT-3.
        - The orchestrator requires precise references to information it's been given, and it may need a reminder to check for specific parts; it's best to be explicit and use the _exact_ capitalized terminology to refer to concepts or information sections (e.g. "MAIN TASK" or "KNOWLEDGE section"); however, only capitalize terms that are capitalized in the information sections—don't use capitalization as emphasis.
        - Typically, tasks that are {TaskWorkStatus.COMPLETED.value}, {TaskWorkStatus.CANCELLED.value}, {TaskWorkStatus.IN_PROGRESS.value}, or {TaskWorkStatus.IN_VALIDATION.value} do not need immediate attention unless the orchestrator discovers information that changes the status of the subtask. Tasks that are {TaskWorkStatus.BLOCKED} will need action from the orchestrator to start or resume execution respectively.
        - The reasoning process should be written in second person and be around 5-7 steps, though you can add substeps within a step (a, b, c, etc.) if it is complex.
        - The reasoning steps can refer to the results of previous steps, and it may be effective to build up the orchestrator's mental context step by step, starting from basic information available, similar to writing a procedural script for a program but in natural language instead of code.

        Provide the reasoning process in the following format:
        ```start_of_reasoning_process
        1. {{reasoning step 1}}
        2. {{reasoning step 2}}
        3. [... etc.]
        ```end_of_reasoning_process
        You may add comments or thoughts before or after the reasoning process, but the reasoning process block itself must only contain the reasoning steps, directed at the orchestrator.
        """
        messages = [
            SystemMessage(
                content=dedent_and_strip(context).format(
                    base_info=BASE_ORCHESTRATOR_INFO,
                    actions=actions,
                )
            ),
            SystemMessage(content=dedent_and_strip(request)),
        ]
        return query_and_extract_reasoning(
            messages,
            preamble=f"Generating reasoning for {role.value} in {state.value} state...\n{print_messages(messages)}",
            printout=printout,
        )
    raise NotImplementedError


MODULAR_SUBTASK_IDENTIFICATION = """
"Modular Subtask Identification" (MSI) is a philosophy for identifying a required subtask from a main task that emphasizes two principles:
- orthogonality: the identified subtask is as independent from the rest of the uncompleted main task as possible. This allows it to be executed in isolation without requiring any other subtasks to be completed first.
- small input/output footprint: the identified subtask has a small input and output footprint, meaning that it requires little information to be provided to it, and provides compact output. This reduces the amount of context needed to understand the subtask and its results.
""".strip()


def generate_subtask_extraction_reasoning(printout: bool = False) -> str:
    """Generate reasoning for choosing an action."""
    context = """
    ## MISSION:
    You are the instructor for an AI task orchestration agent. Your purpose is to provide step-by-step guidance for the agent to think through how to identify the next subtask from the main task description.

    {base_info}

    ## MODULAR SUBTASK INDENTIFICATION PHILOSOPHY:
    {msi}

    """
    request = """
    ## REQUEST FOR YOU:
    Provide a step-by-step, robust reasoning process for the orchestrator to a) sequentially process the information in the information sections it has access to so that it can identify a new subtask that is not yet identified, and b) understand what MSI is and follow its principles. These steps provide the internal thinking that an intelligent agent must go through so that they have all the relevant information on top of mind before they perform subtask identification. Some things to note:
    - Assume that the orchestrator has access to the information described above, but no other information, except for general world knowledge that is available to a standard LLM like GPT-3.
    - The orchestrator requires precise references to information in its information sections, and it may need a reminder to check for specific parts; it's best to be explicit and use the _exact_ capitalized terminology to refer to concepts or information sections (e.g. "MAIN TASK" or "KNOWLEDGE section").
    - In its current state, the orchestrator is not able to perform any other actions besides subtask identification and the reasoning preceeding it.
    - The reasoning process should be written in second person and be around 5-7 steps, though you can add substeps (a, b, c, etc.) within a step if it is complex.
    - The reasoning steps can refer to the results of previous steps, and it may be effective to build up the orchestrator's mental context step by step, starting from basic information available, similar to writing a procedural script for a program but in natural language instead of code.
    - The orchestrator should only perform the subtask identification on the _last_ step, after it has considered _all_ the information it needs. No other actions need to be performed after subtask identification.
    Provide the reasoning process in the following format:
    ```start_of_reasoning_process
    1. {reasoning step 1}
    2. {reasoning step 2}
    3. [... etc.]
    ```end_of_reasoning_process
    You may add comments or thoughts before or after the reasoning process, but the reasoning process block itself must only contain the reasoning steps, directed at the orchestrator.
    """
    messages = [
        SystemMessage(
            content=dedent_and_strip(context).format(
                base_info=BASE_ORCHESTRATOR_INFO,
                msi=MODULAR_SUBTASK_IDENTIFICATION,
            )
        ),
        SystemMessage(content=dedent_and_strip(request)),
    ]
    return query_and_extract_reasoning(
        messages,
        preamble=f"Generating subtask extraction reasoning...\n{print_messages(messages)}",
        printout=printout,
    )


@dataclass(frozen=True)
class ActionDecision:
    """Decision for an action."""

    action_choice: str
    justifications: str

    @cached_property
    def action_name(self) -> str:
        """Name of the action chosen."""
        return (
            self.action_choice.split(":")[0]
            if ":" in self.action_choice
            else self.action_choice
        )

    @cached_property
    def action_args(self) -> dict[str, str]:
        """Arguments of the action chosen."""
        action_args: dict[str, str] = {}
        if self.action_name == ActionName.MESSAGE_TASK_OWNER.value:
            action_args["message"] = self.action_choice.split(":")[1].strip().strip('"')
        return action_args

    @classmethod
    def from_yaml_str(cls, yaml_str: str) -> Self:
        """Create an action decision from a YAML string."""
        return cls(**yaml.load(yaml_str))

    def validate_action(self, valid_actions: Iterable[str]) -> None:
        """Validate that the action is allowed."""
        for allowed_action in valid_actions:
            if self.action_choice.startswith(allowed_action):
                return
        raise ValueError(
            "Action choice validation failed.\n"
            f"{valid_actions=}\n"
            f"{self.action_choice=}\n"
        )


PauseExecution = NewType("PauseExecution", bool)


@dataclass
class ActionResult:
    """Result of an action."""

    pause_execution: PauseExecution
    new_events: list[Event]
    new_work_status: TaskWorkStatus | None = None
    new_event_status: TaskEventStatus | None = None


@dataclass
class Orchestrator:
    """A recursively auto-specializing Aranea subagent."""

    blueprint: Blueprint
    task: Task
    files_parent_dir: Path
    delegator: "Delegator"

    _action_mode: ActionModeName = ActionModeName.DEFAULT
    _focused_subtask: Task | None = None

    @classmethod
    @property
    def default_recent_events_size(cls) -> int:
        """Default size of recent events."""
        return 10

    @property
    def id(self) -> RuntimeId:
        """Runtime id of the orchestrator."""
        return RuntimeId(f"{self.blueprint_id}_{self.task.id}")

    @property
    def blueprint_id(self) -> BlueprintId:
        """Id of the orchestrator."""
        return self.blueprint.id

    @property
    def executor_max_rank(self) -> int | None:
        """Maximum rank of the orchestrator's task executors."""
        executors = [subtask.executor for subtask in self.task.subtasks]
        ranks = [
            executor.rank
            for executor in executors
            if executor is not None and executor.rank is not None
        ]
        # if filtered ranks is less than num executors, it means some task have either not been delegated or its executor is unranked
        return None if len(ranks) != len(executors) else max(ranks)

    @property
    def rank_limit(self) -> int | None:
        """Limit of how high the orchestrator can be ranked."""
        return self.task.rank_limit

    @property
    def rank(self) -> int | None:
        """Rank of the orchestrator."""
        # we always go with existing rank if available b/c executor_max_rank varies and could be < existing rank between runs
        if self.blueprint.rank is not None:
            return self.blueprint.rank
        if (
            rank := None
            if self.executor_max_rank is None
            else 1 + self.executor_max_rank
        ) is not None:
            assert (
                rank >= 1
            ), f"Orchestrator rank must be >= 1. For {self.id}, rank={rank}."
            if self.rank_limit is not None:
                rank = min(rank, self.rank_limit)
        return rank

    @property
    def task_history(self) -> TaskHistory:
        """History of tasks completed by the orchestrator."""
        return self.blueprint.task_history

    @property
    def reasoning(self) -> Reasoning:
        """Instructions for the orchestrator for various tasks."""
        return self.blueprint.reasoning

    @property
    def knowledge(self) -> str:
        """Learnings from past tasks."""
        return self.blueprint.knowledge or NONE

    @property
    def role(self) -> Role:
        """Role of the orchestrator."""
        return self.blueprint.role

    @property
    def core_template(self) -> str:
        """Template for the core state."""
        template = f"""
        ## MISSION:
        You are an advanced task orchestrator that specializes in managing the execution of a MAIN TASK and delegating its SUBTASKS to EXECUTORS that can execute those tasks, while communicating with the MAIN TASK OWNER to gather required information for the task. Your overall purpose is to facilitate task execution by communicating with both the MAIN TASK OWNER and SUBTASK EXECUTORS to complete the MAIN TASK as efficiently as possible.

        ## KNOWLEDGE:
        In addition to the general background knowledge of your language model, you have the following, more specialized knowledge that may be relevant to the task at hand:
        ```start_of_knowledge
        {{knowledge}}
        ```end_of_knowledge

        ## MAIN TASK DESCRIPTION:
        Here is information about the main task you are currently working on:
        ```start_of_main_task_description
        {{task_specification}}
        ```end_of_main_task_description
        More recent information may be available in the RECENT EVENTS LOG below. These will be automatically integrated into the main task description when they are no longer recent.

        ## SUBTASKS:
        - SUBTASKS are tasks that must be executed in order to complete the MAIN TASK.
        - You do NOT execute subtasks yourself, but instead delegate them to SUBTASK EXECUTORS.
        - Typically, tasks that are COMPLETED, CANCELLED, IN_PROGRESS, or IN_VALIDATION do not need attention unless you discover information that changes the status of the subtask.
        - In contrast, tasks that are NEW or BLOCKED will need action from you to start/continue execution.
        - This is not an exhaustive list of all required subtasks for the main task; you may discover additional subtasks that must be done to complete the main task.

        ### SUBTASKS ({TaskWorkStatus.COMPLETED.value}):
        These tasks have been reported as completed, and validated as such by the validator; use this section as a reference for progress in the main task.
        ```start_of_completed_subtasks
        {{completed_subtasks}}
        ```end_of_completed_subtasks

        ### SUBTASKS ({TaskWorkStatus.CANCELLED.value}):
        You have previously cancelled these subtasks for various reason and they will not be done.
        ```start_of_cancelled_subtasks
        {{cancelled_subtasks}}
        ```end_of_cancelled_subtasks

        ### SUBTASKS ({TaskWorkStatus.IN_VALIDATION.value}):
        These subtasks have been reported as completed by executors, but are still being validated by validators.
        ```start_of_in_validation_subtasks
        {{in_validation_subtasks}}
        ```end_of_in_validation_subtasks

        ### SUBTASKS ({TaskWorkStatus.IN_PROGRESS.value}):
        These are subtasks that you have delegated to other executors and that are currently being executed by them.
        ```start_of_delegated_subtasks
        {{delegated_subtasks}}
        ```end_of_delegated_subtasks

        ### SUBTASKS ({TaskWorkStatus.BLOCKED.value}):
        These subtasks are blocked by some issue, and execution cannot continue until the issue is resolved, typically by discussing the blocker and/or creating a new subtask to resolve the blocker.
        ```start_of_blocked_subtasks
        {{blocked_subtasks}}
        ```end_of_blocked_subtasks
        """
        # ### SUBTASKS ({TaskWorkStatus.IDENTIFIED.value}):
        # These subtasks are newly identified and not yet begun.
        # ```start_of_identified_subtasks
        # {{identified_subtasks}}
        # ```end_of_identified_subtasks
        return dedent_and_strip(template)

    @property
    def core_state(self) -> CoreState:
        """Overall state of the orchestrator."""
        return CoreState(
            id=self.id,
            knowledge=self.knowledge,
            main_task=self.task,
            subtasks=self.task.subtasks,
            template=self.core_template,
        )

    @property
    def files_dir(self) -> Path:
        """Directory for files related to the orchestrator."""
        return self.files_parent_dir / self.id

    @property
    def serialization_location(self) -> Path:
        """Return the location where the orchestrator should be serialized."""
        return self.files_dir / "blueprint.yml"

    @property
    def output_dir(self) -> Path:
        """Output directory of the orchestrator."""
        return self.files_dir / "output"

    @property
    def workspace_dir(self) -> Path:
        """Workspace directory of the orchestrator."""
        return self.files_dir / "workspace"

    @property
    def name(self) -> str:
        """Name of the orchestrator."""
        return self.blueprint.name

    def make_files_dirs(self) -> None:
        """Make the files directory for the orchestrator."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def serialize(self) -> dict[str, Any]:
        """Serialize the orchestrator to a dict."""
        return asdict(self.blueprint)

    def save(self, update_blueprint: bool = True) -> None:
        """Serialize the orchestrator to YAML."""
        if update_blueprint:
            self.blueprint.rank = self.rank
        # assume that at the point of saving, all executors have been saved and so would have a rank
        assert self.blueprint.rank is not None, "Rank must not be None when saving."
        yaml.dump(self.serialize(), self.serialization_location)

    def accepts(self, task: Task) -> bool:
        """Decides whether the orchestrator accepts a task."""
        raise NotImplementedError

    @property
    def recent_events_size(self) -> int:
        """Number of recent events to display."""
        return self.blueprint.recent_events_size

    @property
    def state_update_frequency(self) -> int:
        """How often to update the state of the task, in terms of new events."""
        return max(1, int(self.recent_events_size / 2))

    @property
    def recent_event_status(self) -> str:
        """Status of recent events."""
        template = """
        ## RECENT EVENTS LOG:
        This is a log of recent events that have occurred during the execution of the main task. This is NOT a complete log—use the main task description and subtask statuses to get a complete picture of the current state of the work:
        ```start_of_recent_events_log
        {event_log}
        ```end_of_recent_events_log
        """
        return dedent_and_strip(template).format(
            event_log=self.task.event_log.recent(
                self.recent_events_size
            ).to_str_with_pov(
                self.id, self.task.owner_id, Concepts.MAIN_TASK_OWNER.value
            ),
        )

    @property
    def default_status(self) -> str:
        """Default status of the orchestrator."""
        template = """
        {core_state}

        {recent_event_status}
        """
        return dedent_and_strip(template).format(
            core_state=str(self.core_state),
            recent_event_status=self.recent_event_status,
        )

    @property
    def default_action_names(self) -> Set[str]:
        """Names of actions available in the default state."""
        return {
            ActionName.IDENTIFY_NEW_SUBTASK.value,
            ActionName.MESSAGE_TASK_OWNER.value,
            ActionName.START_DISCUSSION_FOR_SUBTASK.value,
            ActionName.REPORT_MAIN_TASK_COMPLETE.value,
        }

    @property
    def default_actions(self) -> str:
        """Actions available in the default mode."""
        actions = """
        - `{IDENTIFY_NEW_SUBTASK}`: identify a new subtask from the MAIN TASK that is not yet on the existing subtask list. This adds the subtask to the list and begins a discussion thread with the subtask's executor to start work on the task.
        - `{START_DISCUSSION_FOR_SUBTASK}: "{{id}}"`: open a discussion thread with a subtask's executor, which allows you to exchange information about the subtask, and then optionally updating its status at the end of the discussion—starting, pausing, resuming, cancelling, etc. {{id}} must be replaced with the id of the subtask to be discussed.
        - `{MESSAGE_TASK_OWNER}: "{{message}}"`: send a message to the MAIN TASK OWNER to gather or clarify information about the task. {{message}} must be replaced with the message you want to send.
        - `{REPORT_MAIN_TASK_COMPLETE}`: report the main task as complete.
        - `{WAIT}`: do nothing until the next event from an executor or the MAIN TASK OWNER.
        """
        return dedent_and_strip(
            actions.format(
                IDENTIFY_NEW_SUBTASK=ActionName.IDENTIFY_NEW_SUBTASK.value,
                # IDENTIFIED=TaskWorkStatus.IDENTIFIED.value,
                START_DISCUSSION_FOR_SUBTASK=ActionName.START_DISCUSSION_FOR_SUBTASK.value,
                MESSAGE_TASK_OWNER=ActionName.MESSAGE_TASK_OWNER.value,
                REPORT_MAIN_TASK_COMPLETE=ActionName.REPORT_MAIN_TASK_COMPLETE.value,
                WAIT=ActionName.WAIT.value,
            )
        )

    @property
    def default_action_reasoning(self) -> str:
        """Prompt for choosing an action in the default mode."""
        if not self.blueprint.reasoning.default_action_choice:
            self.blueprint.reasoning.default_action_choice = generate_action_reasoning(
                self.role,
                ActionModeName.DEFAULT,
                self.default_actions,
                printout=VERBOSE,
            )
        action_choice_core = self.blueprint.reasoning.default_action_choice
        template = """
        Use the following reasoning process to decide what to do next:
        ```start_of_reasoning_steps
        {action_choice_core}
        ```end_of_reasoning_steps

        In your reply, you must include output from all steps of the reasoning process, in this block format:
        ```start_of_action_reasoning_output
        1. {{step_1_output}}
        2. {{step_2_output}}
        3. [... etc.]
        ```end_of_action_reasoning_output
        After this block, you must include the action you have decided on, in this format:
        ```start_of_action_choice_output
        justifications: |-
          {{justifications}}
        action_choice: |-
          {{action_choice}} # must be one of the actions listed above, in the same format
        ```end_of_action_choice_output
        Any additional comments or thoughts can be added before or after the output blocks.
        """
        return dedent_and_strip(template).format(action_choice_core=action_choice_core)

    @property
    def default_action_context(self) -> str:
        """Prompt for choosing an action in the default state."""
        template = """
        {default_status}

        ## ACTION CHOICES:
        These are the actions you can currently perform.
        {default_actions}
        """
        return dedent_and_strip(template).format(
            default_status=self.default_status,
            default_actions=self.default_actions,
        )

    def validate_action_mode_value(self, action_mode: ActionModeName) -> None:
        """Validate the value of the action mode."""
        if self._focused_subtask is None:
            assert (
                action_mode == ActionModeName.DEFAULT
            ), f"Action mode must be {ActionModeName.DEFAULT} when focused subtask is None. {action_mode=}, {self._focused_subtask=}"
        else:
            assert (
                action_mode == ActionModeName.SUBTASK_DISCUSSION
            ), f"Action mode must be {ActionModeName.SUBTASK_DISCUSSION} when focused subtask is not None. {action_mode=}, {self._focused_subtask=}"

    @property
    def action_mode(self) -> ActionModeName:
        """What action state the orchestrator is in."""
        self.validate_action_mode_value(self._action_mode)
        return self._action_mode

    @action_mode.setter
    def action_mode(self, value: ActionModeName) -> None:
        """Set the action state of the orchestrator."""
        self._action_mode = value

    def validate_focused_subtask_value(self, focused_subtask: Task | None) -> None:
        """Validate the value of the focused subtask."""
        assert (
            focused_subtask is None or focused_subtask in self.subtasks
        ), f"Focused subtask must be None or in subtasks. {focused_subtask=}, {self.subtasks=}"
        if self._action_mode == ActionModeName.DEFAULT:
            assert (
                focused_subtask is None
            ), f"Focused subtask must be None in default mode. {focused_subtask=}"
        if self._action_mode == ActionModeName.SUBTASK_DISCUSSION:
            assert isinstance(
                focused_subtask, Task
            ), f"Focused subtask must be a Task when in subtask discussion mode. {focused_subtask=}"

    @property
    def focused_subtask(self) -> Task | None:
        """Subtask that the orchestrator is currently focused on."""
        self.validate_focused_subtask_value(self._focused_subtask)
        return self._focused_subtask

    @focused_subtask.setter
    def focused_subtask(self, value: Task | None) -> None:
        """Set the focused subtask of the orchestrator."""
        self.validate_focused_subtask_value(value)
        self._focused_subtask = value

    @property
    def action_choice_context(self) -> str:
        """Context for choosing an action."""
        if self.action_mode == ActionModeName.DEFAULT:
            return self.default_action_context
        raise NotImplementedError

    @property
    def action_choice_reasoning(self) -> str:
        """Prompt for choosing an action."""
        if self.action_mode == ActionModeName.DEFAULT:
            return self.default_action_reasoning
        raise NotImplementedError

    def choose_action(self) -> ActionDecision:
        """Choose an action to perform."""
        messages = [
            SystemMessage(content=self.action_choice_context),
            SystemMessage(content=self.action_choice_reasoning),
        ]
        action_choice = query_model(
            model=precise_model,
            messages=messages,
            preamble=f"Choosing next action...\n{print_messages(messages)}",
            color=AGENT_COLOR,
        )
        if not (
            extracted_result := extract_blocks(
                action_choice, "start_of_action_choice_output"
            )
        ):
            raise ExtractionError("Could not extract action choice from the result.")
        return ActionDecision.from_yaml_str(extracted_result[0])

    @property
    def events(self) -> list[Event]:
        """Events that have occurred during the execution of the task."""
        return self.task.event_log.events

    def message_task_owner(self, message: str) -> ActionResult:
        """Send message to main task owner. Main task is blocked until there is a reply."""
        return ActionResult(
            new_events=[
                Event(
                    data=MessageData(
                        sender=self.id, recipient=self.task.owner_id, content=message
                    )
                ),
            ],
            pause_execution=PauseExecution(True),
            new_work_status=TaskWorkStatus.BLOCKED,
            new_event_status=TaskEventStatus.AWAITING_OWNER,
        )

    @property
    def subtask_extraction_context(self) -> str:
        """Context for extracting a subtask."""
        template = """
        {default_status}

        ## MODULAR SUBTASK IDENTIFICATION PHILOSOPHY (MSI):
        {msi}
        """
        return dedent_and_strip(template).format(
            default_status=self.default_status,
            msi=MODULAR_SUBTASK_IDENTIFICATION,
        )

    @property
    def subtask_extraction_reasoning(self) -> str:
        """Prompt for extracting a subtask."""
        if not self.blueprint.reasoning.subtask_extraction:
            self.blueprint.reasoning.subtask_extraction = (
                generate_subtask_extraction_reasoning(printout=VERBOSE)
            )
        template = """
        Use the following reasoning process to decide what to do next:
        ```start_of_reasoning_steps
        {subtask_extraction_core}
        ```end_of_reasoning_steps

        In your reply, you must include output from all steps of the reasoning process, in this block format:
        ```start_of_reasoning_output
        1. {{step_1_output}}
        2. {{step_2_output}}
        3. [... etc.]
        ```end_of_reasoning_output
        After this block, you must include the subtask you have identified for its executor. To the executor, the identified subtask becomes its own MAIN TASK, and you are the MAIN TASK OWNER of the subtask. The executor knows nothing about the your original MAIN TASK. The subtask must be described in the following format:
        ```start_of_subtask_identification_output
        subtask_identified: |- # high-level, single-sentence description of the subtask
          {{subtask_identified}}
        ```end_of_subtask_identification_output
        Any additional comments or thoughts can be added before or after the output blocks.
        """
        return dedent_and_strip(template).format(
            subtask_extraction_core=self.blueprint.reasoning.subtask_extraction
        )

    @property
    def subtasks(self) -> TaskList:
        """Subtasks of the orchestrator."""
        return self.task.subtasks

    @property
    def validator_state(self) -> str:
        """State sent to the validator."""
        template = """
        ## MAIN TASK DESCRIPTION:
        Here is information about the main task being worked on:
        ```start_of_main_task_description
        {task_specification}
        ```end_of_main_task_description

        ## SUBTASKS:
        Here are the subtasks that have been identified so far:

        ### SUBTASKS (COMPLETED):
        ```start_of_completed_subtasks
        {completed_subtasks}
        ```end_of_completed_subtasks

        ### SUBTASKS (CANCELLED):
        ```start_of_cancelled_subtasks
        {cancelled_subtasks}
        ```end_of_cancelled_subtasks

        ### SUBTASKS (IN_VALIDATION):
        ```start_of_in_validation_subtasks
        {in_validation_subtasks}
        ```end_of_in_validation_subtasks

        ### SUBTASKS (IN_PROGRESS):
        ```start_of_delegated_subtasks
        {delegated_subtasks}
        ```end_of_delegated_subtasks

        ### SUBTASKS (BLOCKED):
        ```start_of_blocked_subtasks
        {blocked_subtasks}
        ```end_of_blocked_subtasks
        """
        # ### SUBTASKS (NEW):
        # ```start_of_new_subtasks
        # {new_subtasks}
        # ```end_of_new_subtasks

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
            self.subtasks.filter_by_status(TaskWorkStatus.IN_PROGRESS)
        )
        # new_subtasks = str(self.subtasks.filter_by_status(TaskWorkStatus.IDENTIFIED))
        blocked_subtasks = str(self.subtasks.filter_by_status(TaskWorkStatus.BLOCKED))
        return dedent_and_strip(template).format(
            task_specification=str(self.task),
            completed_subtasks=completed_subtasks,
            cancelled_subtasks=cancelled_subtasks,
            in_validation_subtasks=in_validation_subtasks,
            delegated_subtasks=delegated_subtasks,
            # new_subtasks=new_subtasks,
            blocked_subtasks=blocked_subtasks,
        )

    def validate_subtask_identification(self, subtask: str) -> ValidationResult:
        """Validate some work."""
        instructions = """
        {validator_state}

        ## REQUEST FOR YOU:
        Please check that the subtask identification is correct:
        - Subtask: {subtask}
        """
        instructions = dedent_and_strip(instructions).format(
            validator_state=self.validator_state,
            subtask=subtask,
        )
        return self.task.validator.validate(instructions)

    def add_subtask(self, subtask: Task) -> None:
        """Add a subtask to the orchestrator."""
        self.task.subtasks.items.append(subtask)

    def focus_on_subtask(self, subtask: Task) -> None:
        """Activate subtask mode."""
        self.action_mode = ActionModeName.SUBTASK_DISCUSSION
        self.focused_subtask = subtask

    def subtask_message(self, subtask: Task, message: str) -> Event:
        """Format a message to a subtask."""
        assert (
            subtask.executor_id is not None
        ), "Cannot post message to subtask without an executor."
        return Event(
            data=MessageData(
                sender=self.id,
                recipient=subtask.executor_id,
                content=message,
            )
        )

    def send_subtask_message(self, message: str) -> None:
        """Post a message to the focused subtask."""
        assert (
            self.focused_subtask is not None
        ), "Subtask message can only be sent when a subtask is focused."
        self.focused_subtask.event_log.add(
            self.subtask_message(self.focused_subtask, message)
        )
        self.focused_subtask.work_status = TaskWorkStatus.IN_PROGRESS

    def initiate_subtask(self, subtask: Task) -> None:
        """Initiate a subtask."""
        self.focus_on_subtask(subtask)
        initiation_message = "Hi, please feel free to ask me any questions about the context of this task—I've only given you a brief description to start with, but I can provide more information if you need it."
        self.send_subtask_message(initiation_message)

    def identify_new_subtask(self) -> ActionResult:
        """Identify a new subtask."""
        messages = [
            SystemMessage(content=self.subtask_extraction_context),
            SystemMessage(content=self.subtask_extraction_reasoning),
        ]
        new_subtask = query_model(
            model=precise_model,
            messages=messages,
            preamble=f"Extracting subtask...\n{print_messages(messages)}",
            color=AGENT_COLOR,
        )
        extracted_subtask = extract_blocks(
            new_subtask, "start_of_subtask_identification_output"
        )
        if not extracted_subtask:
            raise ExtractionError(
                f"Could not extract subtask from the result:\n{new_subtask}"
            )
        extracted_subtask = str(yaml.load(extracted_subtask[-1])["subtask_identified"])
        subtask_validation = self.validate_subtask_identification(extracted_subtask)
        subtask_identification_event = Event(
            data=SubtaskIdentificationData(
                identifier=self.id,
                subtask=extracted_subtask,
                validation_result=subtask_validation,
            )
        )
        if not subtask_validation.valid:
            return ActionResult(
                pause_execution=PauseExecution(False),
                new_events=[subtask_identification_event],
            )
        subtask = Task(
            name=extracted_subtask,
            owner_id=self.id,
            rank_limit=None if self.rank_limit is None else self.rank_limit - 1,
            description=TaskDescription(information=extracted_subtask),
            validator=self.task.validator,
        )
        self.delegator.assign_executor(subtask)
        assert subtask.executor is not None, "Task executor assignment failed."
        self.add_subtask(subtask)
        self.initiate_subtask(subtask)

        breakpoint()
        # task_status: remove task event status—should be wholly accounted for by regular status
        # task_status: update wait action to wait for a specific task
        return ActionResult(
            pause_execution=PauseExecution(False),
            new_events=[subtask_identification_event],
        )

    def act(self, decision: ActionDecision) -> ActionResult:
        """Act on a decision."""
        decision.validate_action(valid_actions=self.default_action_names)
        if decision.action_name == ActionName.MESSAGE_TASK_OWNER.value:
            return self.message_task_owner(decision.action_args["message"])
        if decision.action_name == ActionName.IDENTIFY_NEW_SUBTASK.value:
            return self.identify_new_subtask()
        if decision.action_name == ActionName.START_DISCUSSION_FOR_SUBTASK.value:
            raise NotImplementedError
        if decision.action_name == ActionName.REPORT_MAIN_TASK_COMPLETE.value:
            raise NotImplementedError
        if decision.action_name == ActionName.WAIT.value:
            raise NotImplementedError

        breakpoint()
        # > update main task description to make it clear that additional info may be in recent events
        # > if AUTO_AWAIT_EXECUTOR_ACTION, automatically execute waiting action if there are subtasks awaiting executor reply > otherwise unimplemented
        # > subtask mode instruction generation: orchestrator doesn't always understand that the executor doesn't have the same information as the orchestrator
        # > always refer to subtask as "this task"
        # > remove all extraneous references that aren't relevant
        # > subtask mode action: send message
        # > `{MESSAGE_TASK_OWNER}: "{{message}}"`: send a message to the MAIN TASK OWNER to gather or clarify information about the task. {{message}} must be replaced with the message you want to send.
        # > `{WAIT}`: do nothing until the next event from an executor or the MAIN TASK OWNER.
        # > subtask mode action: close subtask
        # discuss_subtask > open up subtask discussion mode > preset message if subtask is a newly identified one # warn of missing context and have executor ask questions > subtask discussion mode doesn't have other subtasks

    def message_from_owner(self, message: str) -> Event:
        """Create a message from the task owner."""
        return Event(
            data=MessageData(
                sender=self.task.owner_id,
                recipient=self.id,
                content=message,
            )
        )

    def message_to_owner(self, message: str) -> Event:
        """Create a message to the task owner."""
        return Event(
            data=MessageData(
                sender=self.id,
                recipient=self.task.owner_id,
                content=message,
            )
        )

    _new_event_count = 0

    def update_state_from_new_events(self) -> None:
        """Update the state of the task from new events."""
        raise NotImplementedError(
            "This is where we update the main task description based on new events (such as info from main task owner."
        )

    def add_to_event_log(self, events: Sequence[Event]) -> None:
        """Add events to the event log."""
        self.task.event_log.add(*events)
        self._new_event_count += len(events)
        if self._new_event_count >= self.state_update_frequency:
            self.update_state_from_new_events()
            self._new_event_count = 0

    async def execute(self, message: str | None = None) -> str:
        """Execute the task. Adds a message (if provided) to the task's event log, and adds own message to the event log at the end of execution."""
        if message is not None:
            self.add_to_event_log([self.message_from_owner(message)])
        while True:
            action_decision = self.choose_action()
            action_result = self.act(action_decision)
            if action_result.new_events:
                self.add_to_event_log(action_result.new_events)
            if action_result.new_work_status:
                self.task.work_status = action_result.new_work_status
            if action_result.new_event_status:
                self.task.event_status = action_result.new_event_status
            if action_result.pause_execution:
                break
        if not (last_event := self.task.event_log.last_event):
            raise NotImplementedError
        if not isinstance(last_event.data, MessageData):  # type: ignore
            raise NotImplementedError
        if (
            last_event.data.sender != self.id
            and last_event.data.recipient != self.task.owner_id
        ):
            raise NotImplementedError
        return last_event.data.content

    @classmethod
    def load(
        cls,
        blueprint_location: Path,
        task: Task,
        files_parent_dir: Path,
        delegator: "Delegator",
    ) -> Self:
        """Deserialize an Aranea orchestrator from a YAML file."""
        blueprint_data = yaml.load(blueprint_location)
        blueprint_data["task_history"] = tuple(blueprint_data["task_history"])
        return cls(
            blueprint=Blueprint(**blueprint_data),
            task=task,
            files_parent_dir=files_parent_dir,
            delegator=delegator,
        )


@dataclass
class Reply:
    """A reply from the main agent."""

    content: str
    continue_func: Callable[[str], Coroutine[Any, Any, str]]

    async def continue_conversation(self, message: str) -> str:
        """Continue the conversation with a message."""
        return await self.continue_func(message)


def load_executor(blueprint: Blueprint) -> Executor:
    """Factory function for loading an executor from a blueprint."""
    raise NotImplementedError


class Advisor(Protocol):
    """A single-reply advisor for some issue."""

    def advise(self, prompt: str) -> str:
        """Advise on some issue."""
        raise NotImplementedError


class CapabilityMappingError(Exception):
    """Error raised when a task cannot be mapped to a capability."""


@dataclass
class BaseCapability:
    """A base capability of an Aranea agent."""


def automap_base_capability(
    task: Task, base_capabilities: Sequence[BaseCapability]
) -> BaseCapability | None:
    """Automatically map a task to a base capability if possible."""
    if not base_capabilities:
        return None
    raise NotImplementedError
    return BaseCapability()


def get_choice(prompt: str, allowed_choices: Set[Any], advisor: Advisor) -> Any:
    """Get a choice from the advisor."""
    while True:
        if (choice := advisor.advise(prompt)) in allowed_choices:
            return choice
        prompt = f"Invalid input. Valid choices: {allowed_choices}."


@dataclass
class BlueprintSearchResult:
    """Result of a blueprint search."""

    blueprint: Blueprint
    score: float


DelegationSuccessful = NewType("DelegationSuccessful", bool)


@dataclass
class Delegator:
    """Delegates tasks to executors, creating new ones if needed."""

    executors_dir: Path
    base_capabilities: Sequence[BaseCapability]
    advisor: Advisor

    def search_blueprints(
        self,
        task: Task,
        executor_files_dir: Path,
        rank_limit: int | None = None,
    ) -> list[BlueprintSearchResult]:
        """Search for blueprints of executors that can handle a task."""
        if not os.listdir(executor_files_dir):
            return []

        raise NotImplementedError

        # > agent retrieval: success_rate/(1 + completion_time) # completion time is squished down to 0-1 scale via x/(1+x)
        # > rank of none can access all agents
        # > use strategy from self
        # > remember to rerank
        # > filter out executors that have more than 2 tasks and don't must have at least >50% success rate to be a candidate

    def evaluate(
        self,
        candidates: list[BlueprintSearchResult],
        task: Task,
    ) -> Iterable[BlueprintSearchResult]:
        """Evaluate candidates for a task."""
        raise NotImplementedError

    def make_executor(self, task: Task) -> Executor:
        """Factory for creating a new executor for a task."""
        if base_capability := self.map_base_capability(task):
            raise NotImplementedError
            return create_base_capability(base_capability)

        # now we know it's not a basic task that a simple bot can handle, so we must create an orchestrator
        blueprint = Blueprint(
            name=f"aranea_orchestrator_{task.id}",
            role=Role.ORCHESTRATOR,
            rank=None,
            task_history=[task.id],
            reasoning=Reasoning(),
            knowledge="",
            recent_events_size=Orchestrator.default_recent_events_size,
        )
        return Orchestrator(
            blueprint=blueprint,
            task=task,
            files_parent_dir=self.executors_dir,
            delegator=self,
        )

    def delegate(
        self,
        task: Task,
        max_candidates: int = 10,
    ) -> DelegationSuccessful:
        """Find an executor to delegate the task to."""
        candidates = self.search_blueprints(task, self.executors_dir, task.rank_limit)
        if not candidates:
            return DelegationSuccessful(False)
        candidates = sorted(candidates, key=lambda result: result.score, reverse=True)[
            :max_candidates
        ]
        for candidate in self.evaluate(candidates, task):
            candidate = load_executor(candidate.blueprint)
            if candidate.accepts(task):
                task.executor = candidate
                task.rank_limit = candidate.rank
                return DelegationSuccessful(True)
        return DelegationSuccessful(False)

    def assign_executor(self, task: Task) -> None:
        """Assign an existing or new executor to a task."""
        delegation_successful = self.delegate(task)
        # blueprints represent known capabilities; so, failure means we must create a new executor
        if not delegation_successful:
            task.executor = self.make_executor(task)

    def map_base_capability(self, task: Task) -> BaseCapability | None:
        """Map a task to a base capability if possible."""
        base_capability_question = dedent_and_strip(
            """
            Evaluate this task:
            ```
            {task}
            ```
            
            Is this task a base capability, i.e. something that one of our bots can handle, OR a human can do in a few minutes? (y/n)
            """
        ).format(task=task.description)
        is_base_capability = (
            get_choice(
                base_capability_question,
                allowed_choices={"y", "n"},
                advisor=self.advisor,
            )
            == "y"
        )

        if not is_base_capability:
            return

        # now we know it's a base capability
        automapped_capability = automap_base_capability(task, self.base_capabilities)
        capability_validation_question = dedent_and_strip(
            """
            I have identified the following base capability for this task:
            "{automapped_capability}"

            Please confirm whether this is correct (y/n):
            """
        ).format(automapped_capability=automapped_capability)
        if not automapped_capability:
            capability_validation_question = dedent_and_strip(
                """
                I have not identified any base capabilities for this task.

                Please confirm whether this is correct (y/n):
                """
            )
        is_correct = (
            get_choice(
                capability_validation_question,
                allowed_choices={"y", "n"},
                advisor=self.advisor,
            )
            == "y"
        )
        if is_correct:
            return automapped_capability

        # now we know automapped capability didn't work
        manual_capability_prompt = "Automated capability mapping failed. Please choose a base capability for this task:"
        raise NotImplementedError
        manual_capability_choice = get_choice(
            manual_capability_prompt,
            allowed_choices=set(range(len(self.all_base_capabilities))),
            advisor=human,
        )

        # TODO: basic coding task case: 20 lines or less of base python > coding bot will be equipped with function it wrote
        # TODO: basic search task case: search for basic info about a concept
        # TODO: basic file reading/writing task case
        # TODO: basic browser task case


def default_bot_base_capabilities() -> list["BaseCapability"]:
    """Default base capabilities for bots."""
    return []


@dataclass
class Aranea:
    """Main interfacing class for the agent."""

    files_dir: Path = Path(".data/aranea")
    """Directory for files related to the agent and any subagents."""
    base_capabilities: Sequence[BaseCapability] = field(
        default_factory=default_bot_base_capabilities
    )
    """Base automated capabilities of the agent."""
    base_capability_advisor: Advisor = field(
        default_factory=lambda: Human(name="Human Advisor")
    )
    """Advisor for determining if some task is doable via a base capabilities."""
    work_validator: WorkValidator = field(
        default_factory=lambda: Human(name="Human Validator")
    )
    """Agent that approves or rejects work."""

    def __post_init__(self) -> None:
        """Post-initialization hook."""
        configure_langchain_cache(self.cache_dir)

    @property
    def cache_dir(self) -> Path:
        """Directory for the LLM cache."""
        return self.files_dir / ".cache"

    @property
    def executors_dir(self):
        """Directory for executors."""
        return self.files_dir / "executors"

    @cached_property
    def delegator(self) -> Delegator:
        """Delegator for assigning tasks to executors."""
        return Delegator(
            self.executors_dir,
            self.base_capabilities,
            advisor=self.base_capability_advisor,
        )

    @cached_property
    def id(self) -> RuntimeId:
        """Runtime id of the agent."""
        return RuntimeId(str(generate_uuid()))

    @property
    def name(self) -> str:
        """Name of the agent."""
        return f"Aranea_{self.id}"

    async def run(self, message: str) -> Reply:
        """Run the agent with a message, and a way to continue the conversation. Rerunning this method starts a new conversation."""
        if not self.executors_dir.exists():
            self.executors_dir.mkdir(parents=True, exist_ok=True)
        task = Task(
            description=TaskDescription(information=message),
            owner_id=self.id,
            rank_limit=None,
            validator=self.work_validator,
        )
        self.delegator.assign_executor(task)
        assert task.executor is not None, "Task executor assignment failed."
        reply_text = await task.executor.execute()

        async def continue_conversation(message: str) -> str:
            """Continue the conversation with a message."""
            assert (
                task.executor is not None
            ), "Task executor must exist in order to be executed."
            return await task.executor.execute(message)

        return Reply(
            content=reply_text,
            continue_func=continue_conversation,
        )


# > need to add cancellation reason for cancelled tasks
# > when subtask extraction fails, update extraction script (perhaps with trajectory of extraction history)
# > add success record to reasoning processes
# > main aranea agent: create name "Main Task" for task when initializing task
# > check that when printing as main task, name is actually being
# next action execution > placeholder for `wait` action > event log for task also includes agent decisions and thoughts
# ....
# > if a task is set to be complete, trigger validation agent automatically
# knowledge learning: level of confidence about the knowledge # when updating knowledge, can add, subtract, update, or promote/demote knowledge
# knowledge learning: must define terms
# each time agent is rerun, its modified blueprint is saved separately


"""
"discuss_with_agent",
    "cancel_subtask",
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

    id = RuntimeId("test_task_owner")

    def answer_question(self, question: str) -> str:
        """Answer a question regarding the task."""
        return f"Answer to '{question}'"


# null_test_task = Task(
#     name="Some task",
#     description=TaskDescription(
#         information="Some information.", definition_of_done="Some definition of done."
#     ),
#     rank_limit=None,
#     owner_id=NullTestTaskOwner().id,
# )

# example_test_task = Task(
#     name="Reorganize files on a flash drive",
#     description=TaskDescription(
#         information="The files on the flash drive are currently unorganized.",
#         definition_of_done="N/A",
#     ),
#     rank_limit=None,
#     owner_id=Human().id,
# )

# task: learn how to create langchain agent
# task: full flow of learning how to perform some skill from a tutorial
# task: create an oai assistant agent using only documentation # need to set up virtual environment for it
# task: buy herbal tea from amazon

TEST_DIR = Path(".data/test/agents")
# test_blueprint = Blueprint(
#     name="Test blueprint",
#     rank=0,
#     task_history=(TaskId("task1"), TaskId("task2")),
#     reasoning="Primary directive here.",
#     knowledge="Adaptations from past tasks.",
#     output_dir=TEST_DIR,
# )


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


def test_generate_reasoning() -> None:
    """Test generate_reasoning()."""
    assert generate_action_reasoning(
        Role.ORCHESTRATOR, ActionModeName.DEFAULT, printout=True
    )


def test_default_action_reasoning() -> None:
    """Test default_action_reasoning()."""
    orchestrator = Orchestrator(
        blueprint=Blueprint(
            name="Test blueprint",
            role=Role.ORCHESTRATOR,
            rank=0,
            task_history=TaskHistory(),
            reasoning=Reasoning(),
            knowledge="Adaptations from past tasks.",
            delegator=Delegator(
                executors_dir=TEST_DIR,
                base_capabilities=[],
            ),
        ),
        files_parent_dir=TEST_DIR,
        task=null_test_task,
    )
    assert (output := orchestrator.default_action_reasoning)
    print(output)


def test_generate_extraction_reasoning() -> None:
    """Run test."""
    generate_subtask_extraction_reasoning(printout=True)


def test_human_cache_response():
    """Test human response."""

    def ask_questions():
        with shelve.open(str(cache_path), writeback=True) as cache:
            human = Human(reply_cache=cache)
            human.advise("What is your name?")
            human.advise("What is your age?")

    cache_path = Path(".data/test/test_human_reply_cache")
    cache_path.unlink(missing_ok=True)
    ask_questions()
    ask_questions()
    cache_path.unlink(missing_ok=True)


async def test_full() -> None:
    """Run a full flow on an example task."""
    with shelve.open(".data/test/aranea_human_reply_cache", writeback=True) as cache:
        human_tester = Human(reply_cache=cache)
        aranea = Aranea(
            files_dir=Path(".data/test/aranea"),
            base_capability_advisor=human_tester,
            work_validator=human_tester,
        )
        task = "Create an OpenAI assistant agent."
        reply = (result := await aranea.run(task)).content
        while human_reply := human_tester.advise(reply):
            reply = await result.continue_conversation(human_reply)


def test() -> None:
    """Run tests."""
    configure_langchain_cache(Path(".cache"))
    # test_serialize()
    # test_deserialize()
    # test_id_generation()
    # test_generate_reasoning()
    # test_default_action_reasoning()
    # test_generate_extraction_reasoning()
    # test_human_cache_response()
    asyncio.run(test_full())


if __name__ == "__main__":
    test()

# SUBTASK_DESCRIPTION_REQUIREMENTS = """
# - Term Definitions: every term related to the main task is defined within the subtask description. No assumptions of prior knowledge.
# - Contextual Independence: subtask description stands independently of the main task description. Should be understandable without main task details.
# - Objective Clarity: Clearly state the subtask's objective. Objective should be specific, measurable, and achievable within the scope of the subtask.
# """
