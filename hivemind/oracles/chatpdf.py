"""ChatPDF integration."""

import os
import urllib.parse

import requests


def check_url(string: str) -> bool:
    """Check if a string is a valid URL."""
    try:
        result = urllib.parse.urlparse(string)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def fetch_api_key() -> str:
    """
    Fetch the ChatPDF API key from the environmental variables.
    """
    api_key = os.getenv("CHATPDF_API_KEY")
    if not api_key:
        raise ValueError("CHAT_PDF_API_KEY env variable is not set.")
    return api_key


def validate_query(query: str) -> None:
    """
    Check if the query is an empty string and raises an error if it is.
    """
    if not query:
        raise ValueError("Query cannot be an empty string.")


def upload_pdf(pdf_url: str, api_key: str) -> str:
    """
    Upload a PDF file to the ChatPDF service and returns the sourceId.
    """
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
    }
    data = {
        "url": pdf_url,
    }
    response = requests.post(
        "https://api.chatpdf.com/v1/sources/add-url",
        json=data,
        headers=headers,
        timeout=20,
    )
    return response.json()["sourceId"]


def send_query(source_id: str, query: str, api_key: str) -> str:
    """
    Send a query to a PDF file and returns the service's reply.
    """
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
    }
    data = {
        "referenceSources": True,
        "sourceId": source_id,
        "messages": [
            {
                "role": "user",
                "content": query,
            },
        ],
    }
    response = requests.post(
        "https://api.chatpdf.com/v1/chats/message",
        json=data,
        headers=headers,
        timeout=20,
    )
    return response.json()["content"]


def chat_with_pdf(pdf_url: str, query: str) -> str:
    """
    Send a query to a PDF file and returns the service's reply.
    """
    if not check_url(pdf_url):
        raise ValueError("Invalid URL provided.")

    api_key = fetch_api_key()
    validate_query(query)
    source_id = upload_pdf(pdf_url, api_key)
    reply = send_query(source_id, query, api_key)
    return reply


def test() -> None:
    """
    Test the chat_with_pdf function.
    """
    test_url = "https://arxiv.org/pdf/2305.10601.pdf"
    print(f"Expected Result: a summary about the paper at {test_url} ")
    print(f"Actual Result: '{chat_with_pdf(test_url, 'Summarize this paper.')}'")


if __name__ == "__main__":
    test()
