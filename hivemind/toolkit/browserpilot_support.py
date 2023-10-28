"""Supporting functions for browserpilot."""

from browserpilot.agents.gpt_selenium_agent import GPTSeleniumAgent

def run_with_instructions(agent: GPTSeleniumAgent, instructions: str) -> None:
    """Run browserpilot agent with instructions."""
    agent.set_instructions(instructions)
    agent.run()
