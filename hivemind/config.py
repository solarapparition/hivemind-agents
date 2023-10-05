"""Configuration process for Hivemind."""
from pathlib import Path

from hivemind.toolkit.yaml_tools import yaml

BASE_WORK_DIR = Path(".data/shared_workspace")
SECRETS_LOCATION = Path(".data/secrets.yaml")
secrets: dict[str, str] = yaml.load(SECRETS_LOCATION)
BROWSERLESS_API_KEY = secrets["BROWSERLESS_API_KEY"]
SERPER_API_KEY = secrets["SERPER_API_KEY"]
