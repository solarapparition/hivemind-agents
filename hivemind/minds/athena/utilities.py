"""Utilities for companion agent."""

from pathlib import Path
import pickle
from typing import Mapping, Any
from hivemind.toolkit.yaml_tools import save_yaml, default_yaml


def save_progress(
    progress: Mapping[str, Any], progress_dir: Path, reply: Any = None
) -> None:
    """Save progress to a file."""
    # reply is from external party and might not be serializable
    reply_location = progress_dir / "reply.pickle"
    if reply is not None:
        with open(reply_location, "wb") as file:
            pickle.dump(reply, file)
    location = progress_dir / "progress.yaml"
    save_yaml(progress, location)


def load_progress(progress_dir: Path) -> dict[str, Any]:
    """Load progress from disk."""
    location = progress_dir / "progress.yaml"
    progress = default_yaml.load(location) or {} if location.exists() else {}
    return progress
