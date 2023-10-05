"""Agents for performing specific, simple tasks."""

# Adapted from AI Jason's original code here: https://github.com/JayZeeDesign/microsoft-autogen-experiments/blob/main/content_agent.py

from pathlib import Path
from typing import Any
from dataclasses import dataclass
import json

from bs4 import BeautifulSoup
import requests
from autogen import (
    AssistantAgent,
    UserProxyAgent,
)
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate

from hivemind.toolkit.autogen import (
    is_termination_msg,
    ConfigDict,
    DEFAULT_CONFIG_LIST as config_list,
    DEFAULT_LLM_CONFIG as llm_config,
)
from hivemind.config import BROWSERLESS_API_KEY, SERPER_API_KEY, BASE_WORK_DIR


def search(query: str) -> dict[str, Any]:
    """Search a query on Google."""
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json()


def scrape(url: str) -> str | None:
    """Scrape website, and also will summarize the content based on objective if the content is too large, objective is the original objective & task that user give to the agent, url is the url of the website to be scraped"""

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        "Cache-Control": "no-cache",
        "Content-Type": "application/json",
    }

    # Define the data to be sent in the request
    data = {"url": url}

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    response = requests.post(
        f"https://chrome.browserless.io/content?token={BROWSERLESS_API_KEY}",
        headers=headers,
        data=data_json,
        timeout=60,
    )

    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENT:", text)
        if len(text) > 8000:
            output = summarize(text)
            return output
        return text
    print(f"HTTP request failed with status code {response.status_code}")
    return None


def summarize(content: str) -> str:
    """Summarize scraped content."""
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500
    )
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a detailed summary of the following text for a research purpose:
    '''
    {text}
    '''
    SUMMARY:"""
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True,
    )
    output = summary_chain.run(
        input_documents=docs,
    )
    return output


def research(query: str) -> str:
    """Research a query, and return the research report."""
    llm_config_researcher = {
        "functions": [
            {
                "name": "search",
                "description": "google search for relevant information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Google search query",
                        }
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "scrape",
                "description": "Scraping website content based on url",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Website url to scrape",
                        }
                    },
                    "required": ["url"],
                },
            },
        ],
        "config_list": config_list,
    }

    researcher = AssistantAgent(
        name="research_function_assistant",
        system_message="Research information about a given query, collect as much information as possible, and generate detailed research results with technical details with all reference links attached; Add TERMINATE to the end of the research report.",
        llm_config=llm_config_researcher,
    )
    user_proxy = UserProxyAgent(
        name="research_function_user_proxy",
        code_execution_config={"last_n_messages": 2, "work_dir": "coding"},
        is_termination_msg=lambda x: x.get("content", "")
        and x.get("content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="NEVER",
        function_map={
            "search": search,
            "scrape": scrape,
        },
    )
    user_proxy.initiate_chat(researcher, message=query)
    user_proxy.stop_reply_at_receive(researcher) # stops the autoreply
    return user_proxy.last_message()["content"]


@dataclass
class ResearchDaemon:
    """Daemon to research detailed info about a topic."""

    @property
    def name(self) -> str:
        """Name of the daemon."""
        return "research_daemon"

    @property
    def llm_config(self) -> ConfigDict:
        """Return the config for the LLM."""

        return {
            "functions": [
                {
                    "name": "research",
                    "description": "research about a given topic, returning the research material including reference links",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The topic to be researched about",
                            }
                        },
                        "required": ["query"],
                    },
                },
            ],
            "raise_on_ratelimit_or_timeout": None,
            "request_timeout": 600,
            "seed": 42,
            "config_list": config_list,
            "temperature": 0,
        }

    @property
    def work_dir(self) -> str:
        """Return the working directory for the daemon."""
        return str(BASE_WORK_DIR / "research_daemon")

    def run(
        self,
        message: str,
    ) -> None:
        """Run the daemon."""

        user_proxy = UserProxyAgent(
            name=f"{self.name}_user_proxy",
            human_input_mode="TERMINATE",
            max_consecutive_auto_reply=10,
            is_termination_msg=is_termination_msg,
            code_execution_config={"work_dir": self.work_dir},
            llm_config=llm_config,
            system_message="Reply TERMINATE if the task has been solved at full satisfaction. Otherwise, reply CONTINUE, or the reason why the task is not solved yet.",
            function_map={
                "research": research,
            },
        )
        assistant = AssistantAgent(
            name=f"{self.name}_assistant",
            llm_config=self.llm_config,
            system_message="Convert the user's request to a search engine query, and use that to create arguments for your search function.",
        )
        user_proxy.initiate_chat(
            assistant,
            message=message,
        )


def test() -> None:
    """Test the daemon."""
    daemon = ResearchDaemon()
    daemon.run("Find information on the 'autogen' framework for llms.")


if __name__ == "__main__":
    test()
