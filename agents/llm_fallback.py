# This file manages AI interactions using CrewAI framework and handles rate limits
# KEY FILE: All agents connect through CrewAI's Agent, Task, and Crew classes here
from __future__ import annotations

import os
import re
from contextvars import ContextVar
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from crewai.tools import BaseTool


load_dotenv()

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

# Share the retry message across stages of the current analysis.
_shared_limit_message: ContextVar[str | None] = ContextVar("limit_message", default=None)


def get_groq_model(*, for_crew: bool = False) -> str:
    model = os.getenv("MODEL_NAME", "").strip()
    if for_crew:
        model = os.getenv("CREW_MODEL", "").strip() or model
    model = model.removeprefix("groq/")

    # Existing deployments can still carry the retired model in their environment.
    if not model or model == "llama-3.3-70b-versatile":
        return DEFAULT_GROQ_MODEL
    return model


# Function to detect if an error is a rate limit error
def is_rate_limit_error(err: Exception) -> bool:
    # Check if error message contains rate limit keywords
    message = str(err).lower()
    rate_signals = [
        "rate limit",
        "rate_limit",
        "ratelimit",
        "resource_exhausted",
        "quota exceeded",
        "exceeded your current quota",
        "too many requests",
        "429",
    ]
    return any(signal in message for signal in rate_signals)


    # Extract wait time from error message
def extract_retry_wait(err_text: str) -> str:
    lowered = str(err_text).lower()
    patterns = [
        r"try again in\s*([0-9hms\.\s]+)",
        r"retry in\s*([0-9hms\.\s]+)",
        r"retrydelay[^0-9]*([0-9.]+)s?",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            value = match.group(1).strip().rstrip(".,;")
            return value
    return ""


    # Convert time string to seconds
def _wait_to_seconds(wait: str) -> int:
    if not wait:
        return 0

    token = wait.lower().strip().rstrip(".,;")

    h_match = re.search(r"(\d+(?:\.\d+)?)\s*h", token)
    m_match = re.search(r"(\d+(?:\.\d+)?)\s*m", token)
    s_match = re.search(r"(\d+(?:\.\d+)?)\s*s", token)

    if h_match or m_match or s_match:
        hours = float(h_match.group(1)) if h_match else 0.0
        minutes = float(m_match.group(1)) if m_match else 0.0
        seconds = float(s_match.group(1)) if s_match else 0.0
        total = int(round(hours * 3600 + minutes * 60 + seconds))
        return max(total, 0)

    number_match = re.search(r"\d+(?:\.\d+)?", token)
    if number_match:
        return max(int(round(float(number_match.group(0)))), 0)
    return 0


    # Format seconds back to readable hours, minutes, seconds format
def _format_hms(total_seconds: int) -> str:
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours}h {minutes}m {seconds}s"


    # Build a friendly error message with wait time
def build_limit_exceeded_message(err: Exception) -> str:
    wait = extract_retry_wait(str(err))
    if wait:
        seconds = _wait_to_seconds(wait)
        if seconds > 0:
            return f"Groq limit exceeded. Use after {_format_hms(seconds)}."
    return "Groq limit exceeded. Please try again later."


# Store and retrieve shared limit message across the app
def get_shared_limit_message(err: Exception) -> str:
    message = _shared_limit_message.get()
    if message is None:
        message = build_limit_exceeded_message(err)
        _shared_limit_message.set(message)
    return message


# Clear the stored limit message when starting a new pipeline
def reset_shared_limit_message() -> None:
    _shared_limit_message.set(None)


# *** MAIN FUNCTION: WHERE ALL AGENTS CONNECT THROUGH CREWAI ***
# This function creates a CrewAI agent and runs it to process tasks
def kickoff_with_llm_fallback(
    role: str,
    goal: str,
    backstory: str,
    prompt: str,
    expected_output: str,
    *,
    tools: list[BaseTool] | None = None,
) -> tuple[str, str]:
    from crewai import Agent, Crew, LLM, Task

    # Workaround: CrewAI injects an Anthropic-only 'cache_breakpoint' field into
    # every provider's messages, which Groq's API rejects with a 400 error.
    # https://github.com/crewAIInc/crewAI/issues/5886
    try:
        import crewai.llms.cache as _crewai_cache

        _crewai_cache.mark_cache_breakpoint = lambda msg, *args, **kwargs: msg
    except ImportError:
        pass

    load_dotenv()

    # Get API key from environment
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY is missing.")

    llm_options = {}
    agent_options = {}
    task_options = {}
    crew_options = {}
    if tools:
        llm_options = {"parallel_tool_calls": False, "timeout": 25, "max_retries": 0}
        agent_options = {
            "tools": tools, "cache": False, "allow_delegation": False,
            "max_iter": 12, "max_retry_limit": 0, "max_execution_time": 90,
        }
        task_options = {"tool_failure_policy": "raise"}
        crew_options = {"cache": False, "tracing": False}

    # CrewAI strips the routing prefix and sends the full model ID to Groq.
    groq_llm = LLM(
        model=f"openai/{get_groq_model(for_crew=True)}",
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_api_key,
        custom_openai=True,
        **llm_options,
    )

    # CREATE AGENT: This is where each agent (cleaning, eda, viz, etc.) is defined
    primary_agent = Agent(
        role=role,  # Role like "Data Cleaning Specialist" or "EDA Specialist"
        goal=goal,  # Goal like "Clean and prepare data for analysis"
        backstory=backstory,  # Background to give context to the AI
        llm=groq_llm,
        verbose=False,
        **agent_options,
    )
    # CREATE TASK: What the agent should do
    primary_task = Task(
        description=prompt,  # Detailed task description
        agent=primary_agent,  # Assign task to the agent
        expected_output=expected_output,  # What we expect as output
        **task_options,
    )
    # CREATE CREW: Combine agent and tasks into a crew
    primary_crew = Crew(agents=[primary_agent], tasks=[primary_task], verbose=False, **crew_options)

    # RUN THE CREW: Execute the task using kickoff() method
    result = primary_crew.kickoff()
    return str(result), "groq"
