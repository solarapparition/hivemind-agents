"""Configuration process for Hivemind."""
from pathlib import Path
from os import makedirs

from hivemind.toolkit.yaml_tools import yaml

DATA_DIR = Path(".data")
BASE_WORK_DIR = Path(".data/shared_workspace")
makedirs(BASE_WORK_DIR, exist_ok=True)
EMBEDCHAIN_DATA_DIR = Path(".data/embedchain")
makedirs(EMBEDCHAIN_DATA_DIR, exist_ok=True)
SECRETS_LOCATION = Path(".data/secrets.yaml")
makedirs(SECRETS_LOCATION.parent, exist_ok=True)
LANGCHAIN_CACHE_DIR = Path(".data/cache")
makedirs(LANGCHAIN_CACHE_DIR, exist_ok=True)
TEST_DIR = DATA_DIR / "test"
makedirs(TEST_DIR, exist_ok=True)
BROWSERPILOT_DATA_DIR = DATA_DIR / "browserpilot"
makedirs(BROWSERPILOT_DATA_DIR, exist_ok=True)
CHROMEDRIVER_LOCATION = DATA_DIR / "drivers" / "chromedriver" / "chromedriver"
makedirs(CHROMEDRIVER_LOCATION.parent, exist_ok=True)

secrets: dict[str, str] = yaml.load(SECRETS_LOCATION)
BROWSERLESS_API_KEY = secrets["BROWSERLESS_API_KEY"]
SERPER_API_KEY = secrets["SERPER_API_KEY"]
TO_MARKDOWN_API_KEY = secrets["TO_MARKDOWN_API_KEY"]
