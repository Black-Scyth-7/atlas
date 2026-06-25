"""Stage 13/15: iterative ReAct task agent.

One resident Qwen3 acts as a tool-using agent that completes multi-step tasks:

    decide -> call a tool -> observe the result -> decide again -> ... -> finish

Each turn the model either emits a tool call (a JSON object) or a plain-text
final answer. After each tool runs, its result is fed back as an "Observation:"
so the model can adapt — chaining as many steps as the task needs (up to
`max_iterations`).

Risky / irreversible actions (close an app, create a GitHub repo, shut down)
are not executed silently: the agent pauses, speaks a confirmation question, and
waits for the user's spoken "yes" on the next turn before carrying them out. The
pending action + the loop's message history are stashed so the task resumes
seamlessly once confirmed.

Everything runs on the single in-process model via primitives on the LLM
instance (`raw_complete`) and the shared `Tools` registry (`parse` / `execute`).
"""

from __future__ import annotations

import re
from typing import Optional

_CONTINUE = ("If the task is now complete, reply to the user in one short "
             "plain-text sentence (no JSON). Otherwise call the next tool.")

# Spoken yes / no for confirming a risky action.
_YES_RE = re.compile(
    r"\b(yes|yeah|yep|yup|sure|ok|okay|go ahead|do it|please do|confirm|"
    r"affirmative|proceed|continue|of course|absolutely|fine)\b", re.IGNORECASE)
_NO_RE = re.compile(
    r"\b(no|nope|nah|don'?t|do not|cancel|stop|never ?mind|abort|forget it|"
    r"negative|leave it)\b", re.IGNORECASE)


def _is_affirmative(text: str) -> bool:
    return bool(_YES_RE.search(text)) and not _NO_RE.search(text)


def _is_negative(text: str) -> bool:
    return bool(_NO_RE.search(text))


def _short_args(call: dict) -> str:
    args = call.get("arguments") or {}
    parts = []
    for k, v in args.items():
        s = str(v)
        parts.append(f"{k}={s[:40]}")
    return ", ".join(parts)


class Orchestrator:
    def __init__(self, llm, tools, cfg):
        self.llm = llm
        self.tools = tools
        self.cfg = cfg
        # When a risky action awaits the user's spoken confirmation:
        # {"messages": [...react history...], "call": {...tool call...}}.
        self._pending: Optional[dict] = None

    # ---- prompt assembly -------------------------------------------------
    def _system_prompt(self) -> str:
        from llm import _date_note  # lazy: llm imports agents

        sys = self.cfg.react_system_prompt
        if self.tools is not None:
            sys += "\n\n" + self.tools.prompt_block(instructions=False)
        sys += _date_note()
        if getattr(self.llm.cfg, "disable_thinking", False):
            sys += " /no_think"
        return sys

    # ---- risk / confirmation --------------------------------------------
    def _needs_confirm(self, call: dict) -> bool:
        if not self.cfg.confirm_risky:
            return False
        name = call["name"]
        args = call.get("arguments") or {}
        if name == "system_power":
            return str(args.get("action", "")).lower() in ("shutdown", "restart")
        if name == "write_file":
            # Creating a new file is fine; overwriting an existing one isn't.
            import os
            path = os.path.expanduser(str(args.get("path", "")).strip())
            return bool(path) and os.path.exists(path)
        return name in set(self.cfg.risky_tools)

    def _describe(self, call: dict) -> str:
        name = call["name"]
        args = call.get("arguments") or {}
        if name == "create_github_repo":
            vis = "private" if args.get("private", True) else "public"
            return f"create the {vis} GitHub repository '{args.get('name', '')}'"
        if name == "system_power":
            return f"{str(args.get('action', '')).lower()} the computer"
        if name == "close_app":
            return f"close {args.get('name', 'that app')}"
        if name == "run_command":
            return f"run the command: {args.get('command', '')}"
        if name == "debug_python":
            return f"run {args.get('path', 'the script')} to debug it"
        if name == "write_file":
            return f"overwrite the existing file {args.get('path', '')}"
        if name == "reset_all":
            return ("factory-reset everything — erase all memory, conversation "
                    "history, the cache, and your enrolled voice and face")
        return f"run {name}"

    def _confirm_question(self, call: dict) -> str:
        return f"I'm about to {self._describe(call)}. Should I go ahead?"

    # ---- the ReAct loop --------------------------------------------------
    def _loop(self, messages: list[dict]) -> str:
        for _ in range(self.cfg.max_iterations):
            text = self.llm.raw_complete(messages)
            call = self.tools.parse(text) if self.tools is not None else None
            if call is None:
                return text.strip() or "Done."
            messages.append({"role": "assistant", "content": text})
            if self._needs_confirm(call):
                self._pending = {"messages": messages, "call": call}
                return self._confirm_question(call)
            result = self.tools.execute(call)
            print(f"  · {call['name']}({_short_args(call)}) -> {result}",
                  flush=True)
            messages.append(
                {"role": "user", "content": f"Observation: {result}\n{_CONTINUE}"})
        return ("I ran out of steps before finishing that — "
                "can you narrow the task or break it up?")

    def _resume_after_confirm(self, pending: dict) -> str:
        call, messages = pending["call"], pending["messages"]
        result = self.tools.execute(call)
        print(f"  · {call['name']}({_short_args(call)}) -> {result}", flush=True)
        messages.append(
            {"role": "user", "content": f"Observation: {result}\n{_CONTINUE}"})
        return self._loop(messages)

    # ---- entry point -----------------------------------------------------
    def handle(self, user_text: str, context: str = "") -> str:
        # A risky action is waiting on the user's yes/no.
        if self._pending is not None:
            pending, self._pending = self._pending, None
            if _is_affirmative(user_text):
                return self._resume_after_confirm(pending)
            if _is_negative(user_text):
                return "Okay, I won't do that."
            # Neither yes nor no: drop the pending action and treat this as a
            # fresh request (falls through to a new task below).

        messages = [{"role": "system", "content": self._system_prompt()}]
        # Recent conversation (no system msg) for continuity / "it"/"that" refs.
        messages += [m for m in self.llm.history[1:]]
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": user_text})
        return self._loop(messages)
