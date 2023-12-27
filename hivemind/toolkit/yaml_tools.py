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

default_yaml = YAML()
default_yaml.default_flow_style = False
default_yaml.default_style = "|"  # type: ignore
default_yaml.allow_unicode = True


def save_yaml(data: Mapping[str, Any], location: Path) -> None:
    """Save YAML to a file, making sure the directory exists."""
    if not location.exists():
        os.makedirs(location.parent, exist_ok=True)
    default_yaml.dump(data, location)


def as_yaml_str(
    data: Mapping[str, Any] | list[Any], yaml: YAML = default_yaml
) -> str:
    """Dump yaml as a string."""
    yaml.dump(data, stream := StringIO())
    return stream.getvalue().strip()
