"""YAML tools for Hivemind."""

import os
from pathlib import Path
from typing import Mapping, Any
from ruamel.yaml import YAML
from ruamel.yaml.compat import StringIO

yaml_safe = YAML(typ="safe")
yaml_safe.default_flow_style = False
yaml_safe.default_style = "|"  # type: ignore
yaml_safe.allow_unicode = True

yaml = YAML()
yaml.default_flow_style = False
yaml.default_style = "|"  # type: ignore
yaml.allow_unicode = True


def save_yaml(data: Mapping[str, Any], location: Path) -> None:
    """Save YAML to a file, making sure the directory exists."""
    if not location.exists():
        os.makedirs(location.parent, exist_ok=True)
    yaml.dump(data, location)


def dump_yaml_str(data: Mapping[str, Any] | list[Any]) -> str:
    """Dump yaml as a string."""
    stream = StringIO()
    yaml.dump(data, stream)
    value = stream.getvalue().strip()
    return value
