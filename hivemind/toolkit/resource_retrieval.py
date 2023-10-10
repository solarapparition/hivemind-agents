"""Functions for retrieving resources from the web."""

import json

from bs4 import BeautifulSoup
import requests

from hivemind.config import BROWSERLESS_API_KEY


def scrape(url: str, printout: bool = False) -> str | None:
    """Scrape text from webpage."""
    headers = {
        "Cache-Control": "no-cache",
        "Content-Type": "application/json",
    }
    data = {"url": url}
    data_json = json.dumps(data)
    response = requests.post(
        f"https://chrome.browserless.io/content?token={BROWSERLESS_API_KEY}",
        headers=headers,
        data=data_json,
        timeout=60,
    )
    if response.status_code != 200 and printout:
        print(
            f"Scraping of `{url} failed: HTTP request failed with status code {response.status_code}"
        )
        return None
    soup = BeautifulSoup(response.content, "html.parser")
    return soup.get_text()


