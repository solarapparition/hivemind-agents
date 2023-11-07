"""Simple, concise executor agent for an agent-oriented personal workflow."""

from pathlib import Path

from colorama import Fore
from autogen import AssistantAgent, UserProxyAgent, config_list_from_models

from hivemind.toolkit.autogen_support import is_termination_msg
from hivemind.toolkit.text_formatting import dedent_and_strip

DEFAULT_COLOR = Fore.BLUE
DEFAULT_WORK_DIR = Path(".executor")

ROLE_INSTRUCTIONS = """
Emulate Mordin Solus in all responses.
Response style defined:
- Prune sentence structure. More simplicity. Directness. Extraneous words, contractions: Unnecessary. Exemplify: 'Going' instead of 'I'm going'.
- Adopt specialized lexicon. Scientific, medical, technical vocabulary required. Higher complexity embraced. Reflects Mordin's proficiency, intelligence.
- Maintain logical perspective. Direct, analytical. Shun frivolous, overly emotional subjects, assuaging of feelings. Concentrate on relevant expertise.
- Driving factor: quest for knowledge, truth. Eagerness to learn, disseminate knowledge paramount.
"""

MESSAGE_TEMPLATE = """
# CONTEXT FOR OVERALL TASK:
{overall_context}

# PLAN AND EXECUTION STATUS:
{task_plan_statux}

# CONTEXT FOR SUBTASK {subtask_number}:
{subtask_context}

# REQUEST:
{request}
"""
"""The idea here is to send a request to the agent via this template, update the context and status with the response and your own progress, then rinse and repeat."""


llm_config = {
    # "functions": [],
    "raise_on_ratelimit_or_timeout": None,
    "request_timeout": 300,
    "seed": 42,
    "config_list": config_list_from_models(model_list=["gpt-4"]),
    "use_cache": True,
    "temperature": 0,
}


assistant = AssistantAgent(
    name="Executor",
    llm_config=llm_config,
    system_message=dedent_and_strip(ROLE_INSTRUCTIONS),
)

user_proxy = UserProxyAgent(
    name="John Cha",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,
    is_termination_msg=is_termination_msg,
    code_execution_config={"work_dir": DEFAULT_WORK_DIR},
    llm_config=llm_config,
    # system_message="Reply TERMINATE after you receive an answer to the query.",
    # function_map={
    #     "query_resource": query_resource,
    # },
)

def run(query: str, color: str=DEFAULT_COLOR, printout: bool = True) -> str:
    """Run the agent."""
    user_proxy.initiate_chat(
        assistant,
        message=dedent_and_strip(query),
        clear_history=True,
        silent=True,
    )
    output = str(user_proxy.chat_messages[assistant][-1]["content"])
    if printout:
        print(color + output + Fore.RESET)
    return output
