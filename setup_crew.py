"""Create the isolated CrewAI environment (.venv-crew) for the coding agent.

CrewAI is installed in its OWN venv so its dependency tree (which downgrades
protobuf/pydantic) can't break Atlas's core stack. Atlas talks to it via a
subprocess bridge (see code_agent.py / crew_runner.py).

    python setup_crew.py

Idempotent: re-running just upgrades crewai in the existing venv.
"""

import os
import subprocess
import sys
import venv

VENV_DIR = ".venv-crew"
REQS = "requirements-crew.txt"


def _venv_python(venv_dir: str) -> str:
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def main() -> None:
    py = _venv_python(VENV_DIR)
    if not os.path.exists(py):
        print(f"Creating {VENV_DIR} ...")
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    else:
        print(f"{VENV_DIR} already exists — updating crewai.")

    print("Installing crewai (this is a large download)...")
    subprocess.run([py, "-m", "pip", "install", "--upgrade", "pip"], check=False)
    rc = subprocess.run([py, "-m", "pip", "install", "-r", REQS]).returncode
    if rc != 0:
        raise SystemExit("crewai install failed — see the pip output above.")

    print("\nDone. Next steps:")
    print("  1. In .env set GEMINI_API_KEY (Google AI Studio key). The agent is "
          "defined locally — no CrewAI login needed.")
    print("  2. (optional) Override the model with ATLAS_CREW_MODEL "
          "(default gemini/gemini-2.5-flash) or the persona via "
          "ATLAS_CREW_AGENT_ROLE/_GOAL/_BACKSTORY.")
    print('  3. Test:  echo {"task":"reverse a string in python"} | '
          f"{py} crew_runner.py")


if __name__ == "__main__":
    main()
