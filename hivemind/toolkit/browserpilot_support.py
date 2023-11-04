"""Supporting functions for browserpilot."""

from browserpilot.agents.gpt_selenium_agent import GPTSeleniumAgent


def run_browserpilot_with_instructions(
    agent: GPTSeleniumAgent, instructions: str, handle_error: bool = True
) -> str | None:
    """Run browserpilot agent with instructions."""
    agent.set_instructions(instructions)
    try:
        agent.run()
    except Exception as error:
        if not handle_error or "Failed to execute" not in str(error):
            raise error
        return str(error)
    return None
