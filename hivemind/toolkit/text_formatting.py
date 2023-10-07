"""Text formatting utilities."""

from os import getenv
from textwrap import dedent

import requests


def dedent_and_strip(text: str) -> str:
    """Dedent and strip text."""
    return dedent(text).strip()


def webpage_to_markdown(url: str, api_key: str) -> None:
    """Convert a web page to markdown."""
    res = requests.post(
        "https://2markdown.com/api/2md",
        json={"url": url},
        headers={"X-Api-Key": api_key},
        timeout=10,
    )
    return res.json()["article"]



def test() -> None:
    """Run test."""
    print(webpage_to_markdown("https://2markdown.com", getenv("TO_MARKDOWN_API_KEY")))


if __name__ == "__main__":
    test()
