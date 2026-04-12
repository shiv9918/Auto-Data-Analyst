import os
import re
from crewai import Agent, Task, Crew
from dotenv import load_dotenv


load_dotenv()

_shared_limit_message = None


def is_rate_limit_error(err: Exception) -> bool:
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


def _format_hms(total_seconds: int) -> str:
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours}h {minutes}m {seconds}s"


def build_limit_exceeded_message(err: Exception) -> str:
    wait = extract_retry_wait(str(err))
    if wait:
        seconds = _wait_to_seconds(wait)
        if seconds > 0:
            return f"Groq limit exceeded. Use after {_format_hms(seconds)}."
    return "Groq limit exceeded. Please try again later."


def get_shared_limit_message(err: Exception) -> str:
    global _shared_limit_message
    if _shared_limit_message:
        return _shared_limit_message
    _shared_limit_message = build_limit_exceeded_message(err)
    return _shared_limit_message


def reset_shared_limit_message() -> None:
    global _shared_limit_message
    _shared_limit_message = None


def kickoff_with_llm_fallback(
    role: str,
    goal: str,
    backstory: str,
    prompt: str,
    expected_output: str,
) -> tuple[str, str]:
    load_dotenv()
    
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY is missing.")
    
    # Use model string - LiteLLM will handle it via environment variables
    model_string = "groq/llama-3.3-70b-versatile"

    primary_agent = Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=model_string,
        verbose=False,
    )
    primary_task = Task(
        description=prompt,
        agent=primary_agent,
        expected_output=expected_output,
    )
    primary_crew = Crew(agents=[primary_agent], tasks=[primary_task], verbose=False)

    try:
        result = primary_crew.kickoff()
        return str(result), "groq"
    except Exception:
        raise
