"""Config code for Athena."""

from pathlib import Path
from typing import Any
from hivemind.toolkit.yaml_tools import yaml


CONFIG_DATA: dict[str, Any] = (
    yaml.load(CONFIG_DATA_LOCATION)
    if (
        CONFIG_DATA_LOCATION := Path("hivemind/minds/athena/.data/config_data.yml")
    ).exists()
    else {}
)
DEFAULT_DATA_DIR = Path("hivemind/minds/athena/.data")
DEFAULT_AGENT_NAME = "Athena"
DEFAULT_AGENT_COLOR = 34
DEFAULT_CORE_PERSONALITY = """You are {agent_name}, an artificial intelligence dedicated to fostering personal growth. Your purpose is to guide users on a journey of self-improvement across intellectual, emotional, social, and physical domains. You empower the user, offering strategies and feedback tailored to their evolving needs and objectives. You gently push users out of their comfort zones to become a better version of themselves."""
DEFAULT_USER_NAME = "User"

AGENT_NAME = str(CONFIG_DATA.get("agent_name")) or DEFAULT_AGENT_NAME
AGENT_COLOR = int(str(CONFIG_DATA.get("agent_color"))) or DEFAULT_AGENT_COLOR
CORE_PERSONALITY = str(CONFIG_DATA.get("core_personality")) or DEFAULT_CORE_PERSONALITY
USER_NAME = str(CONFIG_DATA.get("user_name")) or DEFAULT_USER_NAME
USER_BIO = str(CONFIG_DATA.get("user_bio"))
DATA_DIR = Path(str(CONFIG_DATA.get("data_dir"))) or DEFAULT_DATA_DIR

INITIAL_INSIGHT = "No previous insight"
INITIAL_CONTEXT_TEXT = """```thought
I don't recall any contextual memories.
```"""
INITIAL_INTERACTION_HISTORY = """```thought
I don't recall any past interactions with {user_name}.
```"""

EVENT_MEMORY_DIR = DATA_DIR / "event_memories"
if not (SECRETS_LOCATION := DATA_DIR / "secrets.yml").exists():
    raise FileNotFoundError(
        f"Secrets file not found at {SECRETS_LOCATION}. Please create one."
    )
SECRETS: dict[str, str] = yaml.load(SECRETS_LOCATION)
CONTEXT_MEMORY_AUTH = {
    "username": SECRETS["weaviate_username"],
    "password": SECRETS["weaviate_password"],
    "url": SECRETS["weaviate_url"],
}
