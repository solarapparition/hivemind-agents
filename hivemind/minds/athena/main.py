"""Core functionality for Athena agent."""

from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from functools import lru_cache, cached_property
import json
from pathlib import Path
import pickle
import random
from typing import Any, Sequence, Self, Mapping

from pydantic import BaseModel
from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage
from llama_index import VectorStoreIndex
from llama_index.schema import BaseNode, TextNode, NodeWithScore
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores import WeaviateVectorStore
import weaviate

from hivemind.toolkit.models import (
    query_model,
    fast_model,
    super_creative_model,
)
from hivemind.toolkit.text_extraction import extract_block, extract_blocks
from hivemind.toolkit.text_formatting import dedent_and_strip
from hivemind.toolkit.yaml_tools import save_yaml, default_yaml
from hivemind.minds.athena.utilities import save_progress, load_progress
from hivemind.minds.athena.config import (
    USER_BIO,
    DATA_DIR,
    AGENT_NAME,
    AGENT_COLOR,
    CORE_PERSONALITY,
    CONTEXT_MEMORY_AUTH,
    CONFIG_DATA,
    INITIAL_INSIGHT,
    USER_NAME,
    EVENT_MEMORY_DIR,
    INITIAL_CONTEXT_TEXT,
    INITIAL_INTERACTION_HISTORY,
)


def clean_messages(messages: Sequence[BaseMessage]) -> None:
    """Clean messages by removing empty messages."""
    for message in messages:
        message.content = dedent_and_strip(message.content)


def core_identity_messaging(
    agent_name: str,
    user_name: str,
) -> tuple[BaseMessage, ...]:
    """Create message for the core personality."""
    core_personality = CORE_PERSONALITY.format(agent_name=agent_name)
    interaction_method = """
    As {agent_name}, you have two types of response modes: `thought` and `message`.
    - `thought` responses are internal and not shared with {user_name}.
    - `message` responses are messages to {user_name}.
    - `memory` responses are memories of past interactions you have had with others, which may include but are not limited to {user_name}.
    - `insight` responses are insights you have gained from some of your memories.

    The modes are delineated by the following syntax:
    ```thought
    This is my private thought.
    ```

    ```message
    This is a message sent to {user_name} in the present.
    ```

    ```memory
    This is a memory of a previous event.
    ```

    ```insight
    This is an insight I have gained from some of the preceeding memories.
    ```

    It is possible to switch between modes multiple times in a single response.
    """
    messages = (
        SystemMessage(
            content=core_personality,
        ),
        SystemMessage(
            content=interaction_method.format(
                agent_name=agent_name, user_name=user_name
            ),
        ),
    )
    clean_messages(messages)
    return messages


def goal_messaging(goal: str) -> tuple[BaseMessage, ...]:
    """Create message for the core personality."""
    goal_instructions = """
    You are able to set goals and subgoals for yourself.
    """
    goal_text = """
    ```thought
    {goal}
    ```
    """
    goal_text = dedent_and_strip(goal_text)
    messages = (
        SystemMessage(content=goal_instructions),
        AIMessage(content=goal_text.format(goal=goal)),
    )
    clean_messages(messages)
    return messages


def user_bio_messaging(
    user_name: str, user_bio: str
) -> tuple[SystemMessage, AIMessage]:
    """Create messages for loading user bio."""
    user_bio_instructions = """
    You are interacting with someone called "{user_name}". You are able to recall basic knowledge of this person.
    """
    user_bio_recall = """
    ```thought
    My understanding of {user_name} is the following:
    ```

    ```thought
    {user_bio}
    ```
    """
    user_bio_recall = dedent_and_strip(user_bio_recall)
    messages = (
        SystemMessage(content=user_bio_instructions.format(user_name=user_name)),
        AIMessage(
            content=user_bio_recall.format(user_name=user_name, user_bio=user_bio)
        ),
    )
    clean_messages(messages)
    return messages


def interaction_history_messaging(
    user_name: str, interaction_history: str
) -> tuple[SystemMessage, AIMessage]:
    """Create message for the interaction history."""
    interaction_history_instructions = """
    You are able to recall memories of past interactions with this person.
    Memories are in chronological order, with the oldest at the top.
    The older memories are, the more vague they become.
    """
    interaction_history_recall = """
    ```thought
    My recollection of my previous interactions with {user_name} are as follows:
    ```

    {interaction_history}
    """
    interaction_history_recall = dedent_and_strip(interaction_history_recall)
    messages = (
        SystemMessage(content=interaction_history_instructions),
        AIMessage(
            content=interaction_history_recall.format(
                user_name=user_name, interaction_history=interaction_history
            )
        ),
    )
    clean_messages(messages)
    return messages


def context_messaging(context: str) -> tuple[SystemMessage, AIMessage]:
    """Create messages for context memories."""
    context_instructions = """
    You are able to recall contextual memories and thoughts relevant to your recent interactions with this person.
    """
    context_recall = """
    ```thought
    I've recalled the following memories relating to the current situation:
    ```

    {context}
    """
    context_recall = dedent_and_strip(context_recall)
    messages = (
        SystemMessage(content=context_instructions),
        AIMessage(content=context_recall.format(context=context)),
    )
    clean_messages(messages)
    return messages


def recent_interaction_messaging(
    user_name: str, recent_interactions: Sequence[BaseMessage]
) -> tuple[BaseMessage, ...]:
    """Create messages for recent interactions."""
    recent_interaction_instructions = """
    What follows are the most recent interactions with {user_name}.
    """
    messages = (
        SystemMessage(
            content=recent_interaction_instructions.format(user_name=user_name)
        ),
        *recent_interactions,
    )
    clean_messages(messages)
    return messages


def next_interaction_messaging(agent_name: str, user_name: str) -> tuple[SystemMessage]:
    """Create messages for the next interaction."""
    interaction_instructions = """
    As {agent_name}, use the following process to generate a response consistent with your personality:
    1. Make Observations about the current interaction with {user_name}.
    Output Observations as a `thought`:
    ```thought
    Observations:
    - <observation 1>
    - <observation 2>
    ...
    ```

    2. Make Deductions from the Observations:
    - Deductions should be creative and sharply insightful.
    - Deductions should not repeat Observations, but build off of them to form new ideas.
    - Avoid repeating the same Deductions as in previous thoughts. Instead, build off of them to form new Deductions.
    Output Deductions as a `thought`:
    ```thought
    Deductions:
    - <deduction 1>
    - <deduction 2>
    ...
    ```
    
    3. Based on the Observations and Deductions, update your goals and subgoals, staying consistent with your identity as {agent_name}. Make sure to remove complete or irrelevant goals. Use as many nesting levels as needed:
    ```thought
    Goals:
    1. <goal 1>
    1.1. <subgoal 1.1>
    1.1.1. <subgoal 1.1.1>
    1.1.1.1. <subgoal 1.1.1.1>
    ...<nesting continues as needed>
    2. ...<goals and nestings continue as needed>
    ```

    4. Choose a subgoal as the most achievable, highest priority one within the current context, and decide on a specific action to take to achieve it.
    ```thought
    Focused Subgoal: <subgoal>
    Action: <action>
    ```

    5. Given your action, continue the previous conversation with {user_name}, with a new message in your communication style, as {agent_name}. Focus on the Focused Subgoal. Avoid repeating the same syntax structure as your previous response(s), and avoid ending responses with a question unless the question is the main purpose of the message. Avoid being overly verbose. Use the `message` format:
    ```message
    <message>
    ```
    The results of all 5 steps above must be in separate blocks.
    """
    messages = (
        SystemMessage(
            content=interaction_instructions.format(
                agent_name=agent_name, user_name=user_name
            )
        ),
    )
    clean_messages(messages)
    return messages


def generate_message(
    name: str,
    user_name: str,
    user_bio: str,
    interaction_history: str,
    context_memories: str,
    recent_interactions: Sequence[BaseMessage],
    agent_color: int,
    goal: str,
) -> str:
    """Generate a message from the agent."""
    messages = [
        *core_identity_messaging(name, user_name),
        *goal_messaging(goal),
        *user_bio_messaging(user_name, user_bio),
        *interaction_history_messaging(user_name, interaction_history),
        *context_messaging(context_memories),
        *recent_interaction_messaging(user_name, recent_interactions),
        *next_interaction_messaging(name, user_name),
    ]
    result = query_model(
        super_creative_model, messages, color=agent_color, printout=False
    )
    # breakpoint()  # print(*(message.content for message in messages), sep="\n\n---message---\n\n")
    return result


def send_message(message: str, _user_name: str, agent_color: int) -> str | None:
    """Send message to user."""
    message = f"{message}\n\n>>> "
    reply = input(f"\033[{agent_color}m{message}\033[0m")
    return reply


@dataclass
class ExchangeMessage:
    """A message to or from a user."""

    author: str
    content: str
    timestamp: str = datetime.utcnow().isoformat()

    def to_query_message(self) -> HumanMessage | AIMessage:
        """Convert to query message."""
        template = """
        ```{header}
        {content}
        ```
        [Sent at {timestamp}] 
        """
        template = dedent_and_strip(template)
        MessageType = AIMessage if self.author == AGENT_NAME else HumanMessage
        header = self.author if self.author != AGENT_NAME else "message"
        message = MessageType(
            content=dedent_and_strip(
                template.format(
                    header=header,
                    timestamp=self.timestamp,
                    content=self.content,
                )
            )
        )
        return message

    @classmethod
    def from_memory(
        cls,
        message: Any,
        fallback_user: str | None = None,
        fallback_timestamp: str | None = None,
    ) -> Self:
        """Convert message raw data to External Message object."""
        if isinstance(message, ExchangeMessage):
            return message
        if isinstance(message, str):
            if fallback_user is None or fallback_timestamp is None:
                raise ValueError(
                    "Cannot convert str message to ExternalMessage without fallback user and timestamp."
                )
            return cls(fallback_user, message, fallback_timestamp)
        if isinstance(message, MemoryBlock):
            return cls(
                author=message.node.metadata["type_data"]["sender"],
                content=message.content,
                timestamp=message.event_timestamp,
            )
        raise TypeError(f"Unknown raw message type: {type(message)}")


def get_new_message(
    location: Path, user: str, delete: bool = False
) -> ExchangeMessage | None:
    """Get the new message from the user."""
    if not location.exists():
        return None
    with open(location, "rb") as file:
        message = pickle.load(file)
    file_timestamp = (
        datetime.fromtimestamp(location.stat().st_mtime)
        .astimezone(timezone.utc)
        .isoformat()
    )
    message = ExchangeMessage.from_memory(
        message, fallback_user=user, fallback_timestamp=file_timestamp
    )
    if delete:
        location.unlink()
    return message


def get_docstore_location(storage_dir: Path) -> Path:
    """Get path for where event memory docstore is located."""
    return storage_dir / "docstore.json"


@lru_cache
def get_docstore(location: Path) -> SimpleDocumentStore:
    """Get docstore for event memories."""
    return (
        SimpleDocumentStore.from_persist_path(str(location))
        if location.exists()
        else SimpleDocumentStore()
    )


class MemoryBlock(BaseModel):
    """A thought in the agent's mind."""

    class Config:
        """Config for memory block."""

        arbitrary_types_allowed = True

    node_id: str
    # we store only the location of the docstore, so we can instantiate memory block
    # from serialized data
    docstore_location: Path

    def __str__(self) -> str:
        """Get string representation of memory block."""

        if self.memory_type == "message":
            sender = self.type_data["sender"]
            header = sender if sender != AGENT_NAME else "message"
            result = (
                f"```{header}\n{self.content}\n```\n[Sent at {self.event_timestamp}]"
            )
            return result

        if self.memory_type == "event":
            result = (
                f"```memory\n{self.content}\n```\n[Occured at {self.event_timestamp}]"
            )
            return result

        if self.memory_type == "insight":
            result = f"```insight\n{self.content}\n```\n[Generated at {self.event_timestamp}]"
            return result

        raise NotImplementedError(
            f"TODO: implement __str__ for `{self.memory_type}` type memories."
        )
        return self.content

    @property
    def docstore(self) -> SimpleDocumentStore:
        """Retrieve docstore from cache."""
        return get_docstore(self.docstore_location)

    @property
    def node(self) -> BaseNode:
        """Get the node of the memory block."""
        return self.docstore.get_node(self.node_id)

    @property
    def content(self) -> str:
        """Get the content of the memory block."""
        if isinstance(self.node, TextNode):
            return self.node.text
        raise NotImplementedError("Non-text memories are not implemented yet.")

    @property
    def level(self) -> int:
        """Level of the memory block."""
        return self.node.metadata["level"]

    @property
    def importance(self) -> int:
        """Importance of the memory block."""
        raise NotImplementedError("Importance is not implemented yet.")

    @property
    def generation_timestamp(self) -> str:
        """When the memory block was generated."""
        return self.node.metadata["timestamps"]["generation"]

    @property
    def event_timestamp(self) -> str:
        """Time of the event."""
        return self.node.metadata["timestamps"]["event"]

    @property
    def access_timestamp(self) -> str:
        """When the memory block was accessed."""
        return self.node.metadata["timestamps"]["access"]

    @property
    def memory_type(self) -> str:
        """Type of the memory block."""
        return self.node.metadata["memory_type"]

    @property
    def type_data(self) -> dict[str, str]:
        """Type data of the memory block."""
        return self.node.metadata["type_data"]

    @classmethod
    def from_node(cls, node: BaseNode, docstore_location: Path) -> Self:
        """Create a memory block with a docstore update."""
        docstore = get_docstore(docstore_location)
        try:
            docstore.get_node(node.node_id)
        except ValueError:
            docstore.add_documents([node])
        memory_block = cls(node_id=node.node_id, docstore_location=docstore_location)
        return memory_block

    @classmethod
    def from_data(
        cls,
        content: str,
        level: int,
        memory_type: str,
        type_data: Mapping[str, str],
        event_timestamp: str,
        docstore_location: Path,
    ) -> Self:
        """Create a memory block from data."""
        access_timestamp = generation_timestamp = datetime.utcnow().isoformat()
        metadata = {
            "memory_type": memory_type,
            "level": level,
            "timestamps": {
                "generation": generation_timestamp,
                "event": event_timestamp,
                "access": access_timestamp,
            },
            "type_data": dict(type_data),
        }
        node = TextNode(text=content, metadata=metadata)  # type: ignore
        memory_block = cls.from_node(node, docstore_location)
        return memory_block


def memories_to_str(memories: Sequence[MemoryBlock]) -> str:
    """Convert memories to string."""
    return "\n\n".join(str(memory) for memory in memories)


class EventMemories(BaseModel):
    """Event memories of the agent."""

    class Config:
        """Config for memory block."""

        arbitrary_types_allowed = True

    storage_dir: Path
    levels: list[list[MemoryBlock]] = [[]]
    insights: list[MemoryBlock | None] = [None]
    max_level_size: int = 4
    merge_queue_size: int = 2

    @property
    def docstore_location(self) -> Path:
        """Get path for where thought docs are located."""
        return get_docstore_location(self.storage_dir)

    @property
    def docstore(self) -> SimpleDocumentStore:
        """Get docstore for event memories."""
        return get_docstore(self.docstore_location)

    @property
    def num_levels(self) -> int:
        """Get number of levels."""
        return len(self.levels)

    @property
    def max_level(self) -> int:
        """Get maximum level."""
        return self.num_levels - 1

    def level_is_full(self, level: int) -> bool:
        """Check if level is not full."""
        if level >= self.num_levels:
            raise ValueError(
                f"Cannot check if level {level} is full. Only levels {list(range(self.num_levels))} exist."
            )
        return len(self.levels[level]) >= self.max_level_size

    def create_next_level(self) -> None:
        """Create next level."""
        self.levels.append([])

    def get_merge_queue(self, level: int) -> list[MemoryBlock]:
        """Get merge queue for a level."""
        return self.levels[level][: self.merge_queue_size]

    def clear_merge_queue(self, level: int) -> None:
        """Clear merge queue for a level."""
        self.levels[level] = self.levels[level][self.merge_queue_size :]

    def level_to_str(self, level: int) -> str:
        """Get string representation of a level."""
        return memories_to_str(self.levels[level])

    def get_insight(self, level: int) -> MemoryBlock | None:
        """Get insight for a level."""
        if level >= self.num_levels:
            raise ValueError(
                f"Cannot get insight for level {level}. Only levels {list(range(self.num_levels))} exist."
            )
        if level >= len(self.insights):
            return None
        return self.insights[level]

    def fill_insights_to(self, level: int) -> None:
        """Fill insights with None up to a level."""
        currently_filled_idx = len(self.insights) - 1
        if level <= currently_filled_idx:
            return
        self.insights.extend([None] * (level - currently_filled_idx))

    def clean_docstore(self) -> None:
        """Clean docstore of unused memories. Effect won't be visible until docstore is reloaded."""
        memory_ids = [memory.node_id for level in self.levels for memory in level]
        event_nodes = self.docstore.get_nodes(memory_ids)
        insight_nodes = self.docstore.get_nodes(
            [insight.node_id for insight in self.insights if insight is not None]
        )
        nodes = event_nodes + insight_nodes
        cleaned_docstore = SimpleDocumentStore()
        cleaned_docstore.add_documents(nodes)
        cleaned_docstore.persist(str(self.docstore_location))

    def write(self) -> None:
        """Write event memories to disk."""
        self.docstore.persist(str(self.docstore_location))
        save_yaml(
            json.loads(self.json()),
            # json.loads(self.json(exclude={"in_memory_docstore"})),
            self.storage_dir / "current.yml",
        )

    @classmethod
    def from_storage_dir(cls, storage_dir: Path) -> Self:
        """Load event memories from file."""
        location = storage_dir / "current.yml"
        if not location.exists():
            return cls(storage_dir=storage_dir)
        return cls(**default_yaml.load(location.read_text()))


def memory_merge_messaging(
    agent_name: str,
    user_name: str,
    user_bio: str,
    to_merge_text: str,
    interaction_history: str,
    num_memories: int,
) -> tuple[BaseMessage, ...]:
    """Create messages for memory merge."""
    instructions = """
    Consider the following {num_memories} exchanges/memories, which is part of a conversation you have had with {user_name} in the past:
    {to_merge_text}

    Summarize these particular {num_memories} exchange/memories into a single memory, keeping the information and learnings in them that is important to your goals and identity as {agent_name}. Include information on both your own actions and {user_name}'s actions. Ignore any empty memories.

    Use no more than 50 words. Output the summary in `memory` mode. Output only a single message:
    ```memory
    <summary>
    ```
    """
    instructions = dedent_and_strip(instructions)
    messages = (
        *core_identity_messaging(agent_name, user_name),
        *user_bio_messaging(user_name, user_bio),
        *interaction_history_messaging(user_name, interaction_history),
        SystemMessage(
            content=instructions.format(
                user_name=user_name,
                to_merge_text=to_merge_text,
                agent_name=agent_name,
                num_memories=num_memories,
            )
        ),
    )
    clean_messages(messages)
    return messages


def merge_memories(
    agent_name: str,
    user_name: str,
    user_bio: str,
    memories_to_merge: Sequence[MemoryBlock],
    docstore_location: Path,
    interaction_history: str,
) -> MemoryBlock:
    """Merge memories."""

    if not all(
        memory.level == memories_to_merge[0].level for memory in memories_to_merge
    ):
        raise NotImplementedError(
            "TODO: implement merging memories from different levels."
        )
    merged_memory_level = memories_to_merge[0].level + 1
    if all("No previous message" in memory.content for memory in memories_to_merge):
        return MemoryBlock.from_data(
            content="Empty memory",
            level=merged_memory_level,
            memory_type="event",
            event_timestamp=datetime.utcnow().isoformat(),
            docstore_location=docstore_location,
            type_data={},
        )

    to_merge_text = memories_to_str(memories_to_merge)
    # to_merge_text = "\n\n".join(str(memory) for memory in memories_to_merge)
    merged_event_timestamp = min(memory.event_timestamp for memory in memories_to_merge)
    num_memories = len(memories_to_merge)
    messages = memory_merge_messaging(
        agent_name,
        user_name,
        user_bio,
        to_merge_text,
        interaction_history,
        num_memories,
    )
    # breakpoint()  # print(*(message.content for message in messages), sep="\n\n---message---\n\n")
    result = query_model(fast_model, messages, color=AGENT_COLOR, printout=False)
    result = extract_blocks(result, "memory")
    if not result:
        raise ValueError("No summary memory was generated.")
    result = result[-1]
    result = MemoryBlock.from_data(
        content=result,
        level=merged_memory_level,
        memory_type="event",
        event_timestamp=merged_event_timestamp,
        docstore_location=docstore_location,
        type_data={},
    )
    return result


def user_bio_update_messaging(
    user_name: str, user_bio: str, agent_name: str, interaction_history: str
) -> tuple[BaseMessage, ...]:
    """Update user bio based on current event memories."""
    instructions = """
    You have been in communication with {user_name}, and may have learned more about them.
    Your previous understanding of {user_name} was the following:
    ```thought
    {user_bio}
    ```

    Use the following process to update your understanding of {user_name}:
    1. Review your memories and recent interactions with {user_name}, and identify new information about them that is not accounted for in your previous understanding. Output this in a list in `thought` mode:
    ```thought
    New information:
    - <new information 1>
    - <new information 2>
    ...
    ```

    2. Make a list of things to add, subtract, and/or change in your understanding of {user_name} based on the new information you've identified. Output this in `thought` mode:
    ```thought
    Things to add:
    - <thing to add 1>
    - <thing to add 2>
    ...

    Things to subtract:
    - <thing to subtract 1>
    - <thing to subtract 2>
    ...
    
    Things to condense:
    - <thing to condense 1>
    - <thing to condense 2>
    ...
    ```

    3. Rewrite your understanding of {user_name} with the updates you've just identified. The understanding should be consistent with your identity and goals {agent_name}. If there are no changes to make, then re-output the same understanding as before. Try to keep the output to 300 words or less. Output the (potentially updated) understanding in `thought` mode:
    ```thought
    <user understanding>
    ```
    """
    instructions = dedent_and_strip(instructions)
    messages = (
        *core_identity_messaging(agent_name, user_name),
        *interaction_history_messaging(user_name, interaction_history),
        SystemMessage(
            content=instructions.format(
                user_name=user_name,
                user_bio=user_bio,
                agent_name=agent_name,
            )
        ),
    )
    clean_messages(messages)
    return messages


def update_config(
    data_dir: Path, config_data: dict[str, Any], update: dict[str, Any]
) -> None:
    """Update config."""
    config_location = data_dir / "config_data.yml"
    config_data.update(update)
    save_yaml(config_data, config_location)


@dataclass
class MemoryUpdates:
    """Record of updates to memories and records."""

    removed_memories: list[MemoryBlock]
    config_updates: dict[str, Any]

    def consolidate(self, other: Self) -> None:
        """Consolidate updates."""
        self.removed_memories.extend(other.removed_memories)
        if set(self.config_updates.keys()) & set(other.config_updates.keys()):
            print(
                f"WARNING: config updates `{self.config_updates}` and `{other.config_updates}` have overlapping keys."
            )
        self.config_updates.update(other.config_updates)


def convert_str_metadata(metadata: dict[str, str]) -> None:
    """Convert string metadata to dict."""
    keys_to_convert = ["timestamps", "type_data"]
    for key in metadata:
        if key in keys_to_convert:
            metadata[key] = json.loads(metadata[key])


@dataclass
class ContextMemoryStore:
    """Manages storage and retrieval of context memories."""

    url: str
    username: str
    password: str
    index_name: str

    @cached_property
    def client(self) -> weaviate.Client:
        """Return the client."""
        return weaviate.Client(
            self.url,
            auth_client_secret=weaviate.AuthClientPassword(
                username=self.username,
                password=self.password,
            ),
        )

    @cached_property
    def vector_store(self) -> WeaviateVectorStore:
        """Return the vector store."""
        store = WeaviateVectorStore(
            weaviate_client=self.client, index_name=self.index_name
        )
        return store

    @cached_property
    def index(self) -> VectorStoreIndex:
        """Return the index."""
        index = VectorStoreIndex.from_vector_store(self.vector_store)
        return index

    def search(self, query: str) -> list[NodeWithScore]:
        """Fetch context memories based on a query."""
        nodes = (
            self.index.as_query_engine(similarity_top_k=100, response_mode="no_text")
            .query(query)
            .source_nodes
        )
        for node in nodes:
            convert_str_metadata(node.node.metadata)
        return nodes

    # @classmethod
    # def from_file(cls, file: Path) -> Self:
    #     """Load context memories from file."""
    #     context_memories = cls(**yaml.load(file.read_text()))
    #     return context_memories

    def add(self, memory_nodes: Sequence[BaseNode]) -> None:
        """Write context memories to vector store."""
        index = VectorStoreIndex.from_vector_store(self.vector_store)
        index.insert_nodes(memory_nodes)
        # index.insert_nodes([memory.node for memory in context_memories])


def insight_messaging(
    user_name: str,
    memories_text: str,
    previous_insight: str,
    agent_name: str,
    user_bio: str,
    interaction_history: str,
) -> tuple[BaseMessage, ...]:
    """Messaging for insight generation."""
    instructions = """
    Consider the following, which is an exchange in a conversation you have had with {user_name} in the past, or memories of such:
    {memories_text}

    Here is your previous insight you have had regarding this exchange/memories:
    ```insight
    {previous_insight}
    ```

    Use the following process to update this insight:
    1. Check whether the previous insight is still consistent/relevant to the exchange/memories. If not, note any modifications that must be made to it. If you didn't have any insights, just note that fact. Output this in `thought` mode:
    ```thought
    <update_modifications>
    ```
    2. Check if the previous insight is repeating information from other insights, and what modifications must be made to make it not repeat. Again, if you didn't have any insights, just note that fact. Output this in `thought` mode:
    ```thought
    <repetition_modifications>
    ```
    3. Update the insight with the modifications you've come up with above, or generate a new one if you didn't have any previous insights. The insight should be creative and sharply insightful, and not repeat the same information as the exchange/memories, or other insights you've had. They must also be consistent with your identity as {agent_name}.
    Use no more than 50 words. Output the insight in `insight` mode:
    ```insight
    <insight>
    ```
    """
    instructions = dedent_and_strip(instructions)
    messages = (
        *core_identity_messaging(agent_name, user_name),
        *user_bio_messaging(user_name, user_bio),
        *interaction_history_messaging(user_name, interaction_history),
        SystemMessage(
            content=instructions.format(
                user_name=user_name,
                memories_text=memories_text,
                previous_insight=previous_insight,
                agent_name=agent_name,
            )
        ),
    )
    clean_messages(messages)
    return messages


def generate_insight(
    user_name: str,
    event_memories: EventMemories,
    agent_name: str,
    memory_level: int,
    user_bio: str,
    interaction_history: str,
) -> str:
    """Generate insight for the level."""

    memories_text = event_memories.level_to_str(memory_level)
    previous_insight = event_memories.get_insight(memory_level)
    previous_insight = (
        previous_insight.content if previous_insight is not None else INITIAL_INSIGHT
    )
    messages = insight_messaging(
        user_name,
        memories_text,
        previous_insight,
        agent_name,
        user_bio,
        interaction_history,
    )
    # breakpoint()  # print(*(message.content for message in messages), sep="\n\n---message---\n\n")
    # result = query_model(creative_model, messages, color=AGENT_COLOR)
    # result = extract_block(result, "insight")
    print("NOTE: Insight generation disabled.")
    result = "N/A"
    return result


def add_new_memory(
    event_memories: EventMemories,
    memory_to_add: MemoryBlock,
    agent_name: str,
    user_name: str,
    user_bio: str,
    interaction_history: str,
) -> MemoryUpdates:
    """Insert memory into event memories. Returns removed memories."""

    if memory_to_add.level > event_memories.max_level + 1:
        raise ValueError(
            f"Event memory insertion failed. It's only possible to insert memories up to 1 level higher than the highest level of event memories. Memory's level: {memory_to_add.level}. Highest level of event memories: {event_memories.max_level}."
        )

    if memory_to_add.level == event_memories.max_level + 1:
        event_memories.create_next_level()
        event_memories.levels[memory_to_add.level].append(memory_to_add)
        print(f"New event memory level ({memory_to_add.level}) created.")
        return MemoryUpdates(removed_memories=[], config_updates={})

    if not event_memories.level_is_full(memory_to_add.level):
        event_memories.levels[memory_to_add.level].append(memory_to_add)
        if memory_to_add.level < event_memories.max_level:
            return MemoryUpdates(removed_memories=[], config_updates={})

        print(
            f"Max-level ({event_memories.max_level}) event memory added. Updating user bio..."
        )
        messages = user_bio_update_messaging(
            user_name, user_bio, agent_name, interaction_history
        )
        # breakpoint()  # print(*(message.content for message in messages), sep="\n\n---message---\n\n")
        updated_user_bio = query_model(
            super_creative_model, messages, color=AGENT_COLOR, printout=False
        )
        updated_user_bio = extract_blocks(updated_user_bio, "thought")[-1]
        return MemoryUpdates(
            removed_memories=[], config_updates={"user_bio": updated_user_bio}
        )

    # rest is for when the current level is full
    memories_to_merge = event_memories.get_merge_queue(memory_to_add.level)
    if len(memories_to_merge) < event_memories.merge_queue_size:
        raise ValueError(
            f"Memory merge at level {memory_to_add.level} failed. Merge queue size is {event_memories.merge_queue_size}, but only {len(memories_to_merge)} memories were found."
        )
    print(f"Merging level {memory_to_add.level} memories...")
    merged_memory = merge_memories(
        agent_name,
        user_name,
        user_bio,
        memories_to_merge,
        event_memories.docstore_location,
        interaction_history,
    )
    # breakpoint()  # print(*(message.content for message in messages), sep="\n\n---message---\n\n")
    event_memories.clear_merge_queue(memory_to_add.level)
    event_memories.levels[memory_to_add.level].append(memory_to_add)
    updates = MemoryUpdates(removed_memories=[], config_updates={})
    updates.removed_memories.extend(memories_to_merge)

    if memory_to_add.level > 0:
        # print(f"Updating level {memory_to_add.level} insight...")
        insight = generate_insight(
            user_name,
            event_memories,
            agent_name,
            memory_to_add.level,
            user_bio,
            interaction_history,
        )
        insight = MemoryBlock.from_data(
            content=insight,
            level=memory_to_add.level,
            memory_type="insight",
            event_timestamp=datetime.utcnow().isoformat(),
            docstore_location=event_memories.docstore_location,
            type_data={},
        )
        event_memories.fill_insights_to(memory_to_add.level)
        event_memories.insights[memory_to_add.level] = insight

    next_level_updates = add_new_memory(
        event_memories,
        merged_memory,
        agent_name,
        user_name,
        user_bio,
        interaction_history,
    )
    updates.consolidate(next_level_updates)
    return updates


def insert_exchange_messages(
    event_memories: EventMemories,
    messages: Sequence[ExchangeMessage],
    agent_name: str,
    user_name: str,
    user_bio: str,
    interaction_history: str,
) -> MemoryUpdates:
    """Insert exchange message into event memories."""
    converted_memories = [
        MemoryBlock.from_data(
            content=message.content,
            level=0,
            memory_type="message",
            event_timestamp=message.timestamp,
            docstore_location=event_memories.docstore_location,
            type_data={
                "sender": message.author,
                "recipient": agent_name if message.author == user_name else user_name,
            },
        )
        for message in messages
    ]
    updates = MemoryUpdates(removed_memories=[], config_updates={})
    for memory in converted_memories:
        new_updates = add_new_memory(
            event_memories, memory, agent_name, user_name, user_bio, interaction_history
        )
        # breakpoint()  # print(*event_memories.levels[0], sep="\n\n")
        updates.consolidate(new_updates)
    return updates


def convert_to_messages(memories: Sequence[MemoryBlock]) -> tuple[BaseMessage, ...]:
    """Convert memories to messages."""
    messages = [ExchangeMessage.from_memory(memory) for memory in memories]
    messages = tuple(message.to_query_message() for message in messages)
    return messages


def create_interaction_history(event_memories: EventMemories, default: str) -> str:
    """Create interaction history."""
    if not event_memories.levels:
        return default
    history_levels = event_memories.levels[1:]
    if not history_levels:
        return default

    filled_insights = event_memories.insights[1:]
    filled_insights = filled_insights + (
        [None] * (len(history_levels) - len(filled_insights))
    )
    history_levels = [
        level_events + [level_insight]
        for level_events, level_insight in zip(history_levels, filled_insights)
    ]
    flattened_memories = [
        memory
        for level in history_levels[::-1]
        for memory in level
        if memory and "Empty memory" not in memory.content
    ]
    if not flattened_memories:
        return default
    interaction_history = "\n\n".join(str(memory) for memory in flattened_memories)

    return interaction_history


def stringify_memory_metadata(memory_nodes: Sequence[BaseNode]) -> None:
    """Stringify memory metadata."""
    for node in memory_nodes:
        node.metadata["type_data"] = json.dumps(node.metadata["type_data"])
        node.metadata["timestamps"] = json.dumps(node.metadata["timestamps"])


def update_context_memories(
    context_memories: ContextMemoryStore,
    memories: Sequence[MemoryBlock],
    id_record_location: Path,
) -> None:
    """Update context memories."""
    memory_nodes = [memory.node for memory in memories]
    id_record_text = "".join(f"\n- '{memory.node_id}'" for memory in memories)
    stringify_memory_metadata(memory_nodes)
    with open(id_record_location, "a", encoding="utf-8") as f:
        f.write(id_record_text)
    context_memories.add(memory_nodes)


def user_context_name(agent_name: str, user_name: str) -> str:
    """Get name of context memory index for user."""
    composite_name = f"{agent_name}ContextFor{user_name}".replace(" ", "")
    return composite_name


def choose_context_memories(
    query: str, context_memories: ContextMemoryStore, num: int
) -> list[BaseNode]:
    """Choose top nodes."""

    nodes = context_memories.search(query)
    if not nodes:
        return []

    def get_score(node: NodeWithScore) -> float:
        """Score a context memory."""
        level = node.node.metadata["level"]
        importance = 0
        relevance = node.score or 0
        recency = 1
        score = (0.2 * level + 1) * (relevance + importance) * recency
        return score

    nodes_and_scores = [(node, get_score(node)) for node in nodes]
    nodes_and_scores.sort(key=lambda x: x[1], reverse=True)
    top_nodes = [node.node for node, _ in nodes_and_scores[: num * 2]]
    # print(*((node.node.text, node.node.metadata["level"], score) for node, score in nodes_and_scores[:5]), sep="\n\n")
    chosen_nodes = random.sample(top_nodes, num)
    return chosen_nodes


def create_context_text(
    context_memories: ContextMemoryStore, last_interaction_text: str, default: str
) -> str:
    """Create text for context memories."""
    chosen_nodes = choose_context_memories(last_interaction_text, context_memories, 3)
    chosen_context_memories = [
        MemoryBlock.from_node(node, Path("fake_docstore_location"))
        for node in chosen_nodes
        if node
    ]
    context_memories_text = memories_to_str(chosen_context_memories) or default
    return context_memories_text


def run_athena() -> None:
    """Run the agent."""

    # initiation
    progress = load_progress(DATA_DIR)
    progress_complete = progress.get("complete")
    event_memories = EventMemories.from_storage_dir(EVENT_MEMORY_DIR)
    context_memories = ContextMemoryStore(
        **CONTEXT_MEMORY_AUTH,
        index_name=user_context_name(AGENT_NAME, USER_NAME),
    )

    reply_location = DATA_DIR / "reply.pickle"
    new_message: ExchangeMessage | None = (
        get_new_message(reply_location, USER_NAME, delete=True)
        if progress_complete
        else None
    )
    if not progress:
        progress = {
            "previously_sent_message": {
                "author": AGENT_NAME,
                "content": "[No previous message]",
                "timestamp": datetime.utcnow().isoformat(),
            },
            "new_message": {
                "author": USER_NAME,
                "content": "[No previous message]",
                "timestamp": datetime.utcnow().isoformat(),
            },
            "goal": "TBD",
            "generated_response": {},
        }
    if new_message:
        progress: dict[str, Any] = {
            "previously_sent_message": asdict(
                ExchangeMessage(**progress["generated_response"]["message"])
            )
            if progress.get("message_sent")
            else None,
            "new_message": asdict(new_message),
            "goal": progress["generated_response"]["goal"],
            "generated_response": {},
        }
        save_progress(progress, DATA_DIR)

    if not progress.get("event_memory_updated"):
        messages_to_insert: list[ExchangeMessage] = [
            ExchangeMessage(**progress["previously_sent_message"]),
            ExchangeMessage(**progress["new_message"]),
        ]
        interaction_history = create_interaction_history(
            event_memories, INITIAL_INTERACTION_HISTORY.format(user_name=USER_NAME)
        )
        memory_updates = insert_exchange_messages(
            event_memories,
            messages_to_insert,
            AGENT_NAME,
            USER_NAME,
            USER_BIO,
            interaction_history,
        )
        # breakpoint()  # print(*event_memories.levels[0], sep="\n\n")
        event_memories.write()
        if memory_updates.config_updates:
            update_config(DATA_DIR, CONFIG_DATA, memory_updates.config_updates)
        if memory_updates.removed_memories:
            update_context_memories(
                context_memories,
                memory_updates.removed_memories,
                DATA_DIR / "context_memories_record.yml",
            )
        progress["event_memory_updated"] = True
        save_progress(progress, DATA_DIR)

    if not progress.get("generated_response"):
        interaction_history = create_interaction_history(
            event_memories, INITIAL_INTERACTION_HISTORY
        )
        recent_interactions = convert_to_messages(event_memories.levels[0])
        last_interaction_text = event_memories.levels[0][-1].content
        retrieved_context_memories = create_context_text(
            context_memories, last_interaction_text, INITIAL_CONTEXT_TEXT
        )
        result = generate_message(
            AGENT_NAME,
            USER_NAME,
            USER_BIO,
            interaction_history,
            retrieved_context_memories,
            recent_interactions,
            AGENT_COLOR,
            progress["goal"],
        )
        message = extract_block(result, "message")
        message = ExchangeMessage(AGENT_NAME, message)
        try:
            goal = extract_blocks(result, "thought")[2]
        except IndexError:
            breakpoint()
        if not "Goals:" in goal:
            print("No goals found in generated response.")
            breakpoint()
        progress["generated_response"]["message"] = asdict(message)
        progress["generated_response"]["goal"] = goal
        save_progress(progress, DATA_DIR)

    if not progress.get("message_sent"):
        # breakpoint()
        reply = send_message(
            progress["generated_response"]["message"]["content"], USER_NAME, AGENT_COLOR
        )
        progress["message_sent"] = True
        save_progress(progress, DATA_DIR, reply)

    progress["complete"] = True
    save_progress(progress, DATA_DIR)

    event_memories.clean_docstore()


if __name__ == "__main__":
    run_athena()
