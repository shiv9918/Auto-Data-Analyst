# This file manages AI interactions using CrewAI framework and handles rate limits
# KEY FILE: All agents connect through CrewAI's Agent, Task, and Crew classes here
import os
import re
from dotenv import load_dotenv


load_dotenv()

# Store rate limit error messages to show users
_shared_limit_message = None

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
    global _shared_limit_message
    if _shared_limit_message:
        return _shared_limit_message
    _shared_limit_message = build_limit_exceeded_message(err)
    return _shared_limit_message


# Clear the stored limit message when starting a new pipeline
def reset_shared_limit_message() -> None:
    global _shared_limit_message
    _shared_limit_message = None


# *** MAIN FUNCTION: WHERE ALL AGENTS CONNECT THROUGH CREWAI ***
# This function creates a CrewAI agent and runs it to process tasks
def kickoff_with_llm_fallback(
    role: str,
    goal: str,
    backstory: str,
    prompt: str,
    expected_output: str,
) -> tuple[str, str]:
    # Import CrewAI components (Agent, Task, Crew)
    from crewai import Agent, Task, Crew

    load_dotenv()

    # Get API key from environment
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY is missing.")

    # Use Groq's LLM model for faster processing
    model_string = "groq/llama-3.3-70b-versatile"

    # CREATE AGENT: This is where each agent (cleaning, eda, viz, etc.) is defined
    primary_agent = Agent(
        role=role,  # Role like "Data Cleaning Specialist" or "EDA Specialist"
        goal=goal,  # Goal like "Clean and prepare data for analysis"
        backstory=backstory,  # Background to give context to the AI
        llm=model_string,  # Which LLM model to use
        verbose=False,
    )
    # CREATE TASK: What the agent should do
    primary_task = Task(
        description=prompt,  # Detailed task description
        agent=primary_agent,  # Assign task to the agent
        expected_output=expected_output,  # What we expect as output
    )
    # CREATE CREW: Combine agent and tasks into a crew
    primary_crew = Crew(agents=[primary_agent], tasks=[primary_task], verbose=False)

    # RUN THE CREW: Execute the task using kickoff() method
    try:
        result = primary_crew.kickoff()
        return str(result), "groq"
    except Exception:
        raise
