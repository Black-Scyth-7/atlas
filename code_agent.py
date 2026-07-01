"""Bridge from Atlas to the CrewAI coding agent in the isolated .venv-crew.

Atlas delegates programming tasks here; this runs crew_runner.py inside the
separate CrewAI environment as a subprocess and returns the result. CrewAI's
dependencies never enter Atlas's own venv.

Best-effort, like vision.py / face_id.py: if the isolated venv isn't set up or
no cloud provider key is present, `available` is False and Atlas just keeps using
its local coding tools.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import subprocess

# The agent emits project files between these markers so Atlas can write them to
# disk (delimiters avoid the escaping problems of putting code inside JSON).
_FILE_BLOCK = re.compile(r"<<<FILE:\s*(.+?)\s*>>>\r?\n(.*?)<<<ENDFILE>>>", re.DOTALL)
_BUILD_INSTRUCTIONS = (
    "\n\nIf this requires creating one or more project files, output EACH file "
    "in EXACTLY this format, with nothing between the blocks:\n"
    "<<<FILE: relative/path>>>\n<full file content>\n<<<ENDFILE>>>\n"
    "Put every file under ONE clearly-named project subfolder (include it in each "
    "path, e.g. my_project/app.py, my_project/templates/index.html). Include a "
    "README.md with install and run steps. After the blocks, add one short "
    "sentence summary.")

from config import CodeAgentConfig

class CodeAgent:
    def __init__(self, cfg: CodeAgentConfig):
        self.cfg = cfg
        self._py = self._venv_python(cfg.crew_venv)
        self._runner = os.path.abspath(cfg.runner)
        if not cfg.enable_code_agent:
            self.available, self.reason = False, "disabled in config"
        elif not os.path.exists(self._py):
            self.available, self.reason = False, (
                f"{cfg.crew_venv} not set up — run `python setup_crew.py`")
        elif not os.path.exists(self._runner):
            self.available, self.reason = False, f"{cfg.runner} missing"
        elif not os.environ.get("GEMINI_API_KEY"):
            self.available, self.reason = False, "GEMINI_API_KEY not set in .env"
        else:
            self.available, self.reason = True, ""

    @staticmethod
    def _venv_python(venv_dir: str) -> str:
        if os.name == "nt":
            return os.path.join(venv_dir, "Scripts", "python.exe")
        return os.path.join(venv_dir, "bin", "python")

    def _invoke(self, task: str, context: str = "") -> tuple[bool, str]:
        """Run the CrewAI subprocess. Returns (ok, result-or-error-text)."""
        env = os.environ.copy()
        env["ATLAS_CREW_MODEL"] = self.cfg.model
        env["ATLAS_CREW_AGENT_ROLE"] = self.cfg.agent_role
        env["ATLAS_CREW_AGENT_GOAL"] = self.cfg.agent_goal
        env["ATLAS_CREW_AGENT_BACKSTORY"] = self.cfg.agent_backstory
        payload = json.dumps({"task": task, "context": context})
        try:
            proc = subprocess.run(
                [self._py, self._runner], input=payload, capture_output=True,
                text=True, timeout=self.cfg.timeout, env=env,
                cwd=os.path.dirname(self._runner) or None)
        except subprocess.TimeoutExpired:
            return False, f"timed out after {self.cfg.timeout}s"
        except Exception as e:
            return False, f"couldn't run the coding agent: {e}"
        out = (proc.stdout or "").strip()
        if not out:
            return False, ((proc.stderr or "").strip()[-300:] or "no output")
        try:
            data = json.loads(out.splitlines()[-1])
        except Exception:
            return False, f"unreadable reply: {out[:300]}"
        if not data.get("ok"):
            return False, str(data.get("error", "unknown error"))
        return True, str(data.get("result", "")).strip()

    def run(self, task: str, context: str = "") -> str:
        """User-facing: run a coding task, save the result, speak a summary."""
        if not self.available:
            return f"The CrewAI coding agent isn't available ({self.reason})."
        task = (task or "").strip()
        if not task:
            return "What coding task should I hand to the engineer?"
        base = self._dest_base(task) if self.cfg.build_projects else ""
        full_task = task + _BUILD_INSTRUCTIONS if base else task
        ok, text = self._invoke(full_task, context)
        if not ok:
            return f"The coding agent failed: {text}"
        if base:
            built = self._write_project(base, text)
            if built:
                tops = sorted({w.split(os.sep)[0] for w in built})
                where = os.path.join(base, tops[0]) if len(tops) == 1 else base
                return (f"Built the project — wrote {len(built)} files to {where}. "
                        "Check the README to run it.")
        return self._deliver(task, text)

    @staticmethod
    def _dest_base(task: str) -> str:
        """Resolve where to build: a folder named in the task (desktop/documents/
        downloads), else the Desktop. Handles OneDrive redirection."""
        low = (task or "").lower()
        home = os.path.expanduser("~")

        def folder(name: str) -> str:
            plain = os.path.join(home, name)
            onedrive = os.path.join(home, "OneDrive", name)
            return (onedrive if (not os.path.isdir(plain)
                                 and os.path.isdir(onedrive)) else plain)

        if "document" in low:
            base = folder("Documents")
        elif "download" in low:
            base = folder("Downloads")
        else:
            base = folder("Desktop")
        return os.path.abspath(base)

    def _write_project(self, base: str, result: str):
        """Write the agent's <<<FILE>>> blocks under `base` (sandboxed). Returns
        the list of files written (relative to base), or None if no blocks."""
        blocks = _FILE_BLOCK.findall(result)
        if not blocks:
            return None
        written = []
        for rel, content in blocks:
            rel = rel.strip().lstrip("/\\").replace("\\", "/")
            dest = os.path.normpath(os.path.join(base, rel))
            if dest != base and not dest.startswith(base + os.sep):
                continue  # sandbox: never write outside the base folder
            try:
                os.makedirs(os.path.dirname(dest) or base, exist_ok=True)
                with open(dest, "w", encoding="utf-8", newline="\n") as f:
                    f.write(content.rstrip("\n") + "\n")
                written.append(os.path.relpath(dest, base))
            except Exception:
                pass
        return written or None

    def complete(self, task: str, context: str = "") -> str:
        """Programmatic: return the agent's RAW output (e.g. generated code),
        or '' on failure. Used by create_tool to have CrewAI write tool code."""
        if not self.available:
            return ""
        ok, text = self._invoke((task or "").strip(), context)
        return text if ok else ""

    def _deliver(self, task: str, result: str) -> str:
        """Save the full result; speak a short confirmation + excerpt."""
        if not result:
            return "The coding agent returned an empty result."
        path = ""
        try:
            os.makedirs(self.cfg.output_dir, exist_ok=True)
            name = "code_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".md"
            path = os.path.join(self.cfg.output_dir, name)
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(f"# Task\n{task}\n\n# Result\n{result}\n")
        except Exception:
            path = ""
        # Keep the spoken part short — the full code lives in the file.
        first_line = next((ln.strip() for ln in result.splitlines()
                           if ln.strip() and not ln.strip().startswith("```")),
                          "")
        summary = first_line[:200]
        if path:
            return (f"The senior engineer finished — I saved the full solution to "
                    f"{path}." + (f" {summary}" if summary else ""))
        return result[:1500]


if __name__ == "__main__":
    cfg = CodeAgentConfig()
    ca = CodeAgent(cfg)
    print("available:", ca.available, "|", ca.reason or "ok")
    if ca.available:
        import sys
        task = " ".join(sys.argv[1:]) or "Write a Python function that reverses a string."
        print(ca.run(task))
