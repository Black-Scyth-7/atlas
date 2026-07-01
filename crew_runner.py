"""Standalone CrewAI runner — executed inside the isolated .venv-crew.

Atlas (in its own venv) shells out to this script to run a coding task on a
CrewAI agent. They communicate only over a pipe, so CrewAI's heavy dependency
tree never touches Atlas's env.

The agent is defined here from a role/goal/backstory (no CrewAI login or
published repository needed) and runs on Gemini via litellm.

Protocol:
  stdin  : {"task": "<coding task>", "context": "<optional>"}
  stdout : {"ok": true, "result": "<text>"}  or  {"ok": false, "error": "<msg>"}

Environment (passed through by the parent, from Atlas's .env):
  GEMINI_API_KEY            - Google AI Studio key for the Gemini model
  ATLAS_CREW_MODEL          - litellm model string (e.g. gemini/gemini-2.0-flash)
  ATLAS_CREW_AGENT_ROLE     - the agent's role
  ATLAS_CREW_AGENT_GOAL     - the agent's goal
  ATLAS_CREW_AGENT_BACKSTORY- the agent's backstory

Run manually to test (uses your Gemini key):
  echo {"task":"reverse a string in python"} | .venv-crew\\Scripts\\python crew_runner.py
"""

import json
import os
import sys

# CrewAI prints banners/warnings/telemetry to stdout; keep the REAL stdout aside
# for our JSON result and send everything else to stderr, so the parent can
# parse stdout cleanly.
_REAL_STDOUT = sys.stdout
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")


def _emit(obj: dict) -> None:
    _REAL_STDOUT.write(json.dumps(obj))
    _REAL_STDOUT.flush()


def main() -> None:
    sys.stdout = sys.stderr  # route all library chatter away from our JSON
    try:
        raw = sys.stdin.read()
        req = json.loads(raw) if raw.strip() else {}
    except Exception as e:
        _emit({"ok": False, "error": f"bad request JSON: {e}"})
        return

    task_text = str(req.get("task", "")).strip()
    context = str(req.get("context", "")).strip()
    if not task_text:
        _emit({"ok": False, "error": "no task provided"})
        return

    if not os.environ.get("GEMINI_API_KEY"):
        _emit({"ok": False, "error": "GEMINI_API_KEY is not set"})
        return

    try:
        from crewai import Agent, Crew, Task
    except Exception as e:
        _emit({"ok": False, "error": f"crewai not installed in this env ({e})"})
        return

    try:
        model = os.environ.get("ATLAS_CREW_MODEL", "gemini/gemini-2.0-flash")
        agent = Agent(
            role=os.environ.get("ATLAS_CREW_AGENT_ROLE", "Senior Software Engineer"),
            goal=os.environ.get("ATLAS_CREW_AGENT_GOAL",
                                "Write correct, maintainable code."),
            backstory=os.environ.get("ATLAS_CREW_AGENT_BACKSTORY",
                                     "An experienced software engineer."),
            llm=model,
            verbose=False,
        )
        description = task_text
        if context:
            description += f"\n\nContext:\n{context}"
        task = Task(
            description=description,
            expected_output=("A complete, correct, well-structured solution. "
                             "Include the full code in fenced blocks and a brief "
                             "explanation of key decisions."),
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = crew.kickoff()
        _emit({"ok": True, "result": str(result).strip()})
    except Exception as e:
        # Surface CrewAI's real error (API key, model name, quota, etc.).
        _emit({"ok": False, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
