"""Extract blocks from text."""

import re


class ExtractionError(Exception):
    """Raised when an extraction fails."""


def extract_block(text: str, block_type: str) -> str | None:
    """Extract a code block from the text."""
    pattern = (
        r"```{block_type}\n(.*?)```".format(  # pylint:disable=consider-using-f-string
            block_type=block_type
        )
    )
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    extracted_string = match.group(1).strip()
    return extracted_string


def extract_blocks(text: str, block_type: str) -> list[str] | None:
    """Extracts specially formatted blocks of text from the LLM's output. `block_type` corresponds to a label for a markdown code block such as `yaml` or `python`."""

    pattern = (
        r"```{block_type}\n(.*?)```".format(  # pylint:disable=consider-using-f-string
            block_type=block_type
        )
    )
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None
    extracted_strings = [match.strip() for match in matches]
    return extracted_strings
