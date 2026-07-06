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

import datetime
import re
from typing import Optional

_CONTINUE = ("If the task is now complete, reply to the user in one short "
             "plain-text sentence (no JSON). Otherwise call the next tool.")

# --- Freshness router -------------------------------------------------------
# Some questions have answers that change over time (who holds an office, who
# won, current prices). A small local model answers these confidently from its
# STALE training weights and never chooses to search — so we don't leave it to
# the model. When a question matches, we force a web_search up front and hand
# the result to the model as authoritative context (see Orchestrator._forced_
# search). Kept deliberately tight so ordinary questions aren't slowed down.
_OFFICE_RE = re.compile(
    r"\b(president|vice[-\s]?president|prime[-\s]?minister|pm|premier|"
    r"chancellor|king|queen|monarch|emperor|empress|pope|ceo|"
    r"chair(?:man|woman|person)?|governor|mayor|senator|"
    r"chief\s+minister|chief\s+justice|secretary\s+of\s+state|"
    r"head\s+of\s+(?:state|government)|leader)\b", re.IGNORECASE)
_FRESH_RE = re.compile(
    r"\b(current(?:ly)?|latest|most\s+recent|right\s+now|as\s+of\s+(?:now|"
    r"today)|nowadays|these\s+days|today'?s|this\s+(?:year|month|week))\b",
    re.IGNORECASE)
_WHO_IS_RE = re.compile(r"\bwho(?:\s*'?s|\s+is|\s+are|\s+was|\s+were)\b",
                        re.IGNORECASE)
_WHO_WON_RE = re.compile(r"\bwho\s+(?:won|is\s+winning|leads|is\s+leading)\b",
                         re.IGNORECASE)
# Any question stem — a "current/latest ..." phrasing paired with one of these
# is time-sensitive regardless of whether it's a "who" question.
_ASK_RE = re.compile(r"\b(who|what|what'?s|which|where|when|whom|"
                     r"how\s+(?:much|many))\b", re.IGNORECASE)
# Relative dates ("yesterday's match", "last night") mark a query as being about
# recent events — time-sensitive even with no question word. Paired with a
# results/scores noun (below) they force a live search the model wouldn't run.
_RELDATE_RE = re.compile(
    r"\b(yesterday|last\s+night|day\s+before\s+yesterday|today'?s?|tonight|"
    r"this\s+(?:morning|afternoon|evening))\b", re.IGNORECASE)
_RESULT_RE = re.compile(
    r"\b(scores?|scored|results?|final|won|win(?:ner|ning)?|beat|standings?|"
    r"fixtures?|matches?|match|games?|race|tournament|championship)\b",
    re.IGNORECASE)

# A run_command that deletes files still asks first (there's no dedicated
# delete-file tool, so deletions happen through the shell). Matches a delete
# verb in command position: rm, rmdir, rd, del, erase, unlink, shred, Remove-Item.
_DELETE_CMD_RE = re.compile(
    r"(?i)(?:^|[\s;&|(])(rm|rmdir|rd|del|erase|unlink|shred|remove-item)\b")

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


# Phrases that mark a tool result as a failure, so the loop can stop retrying
# and report honestly instead of letting the model claim a false success.
_ERROR_MARKERS = (
    "didn't work", "did not work", "couldn't", "could not", "failed",
    "isn't available", "is not available", "no such", "invalid", "error:",
    "unmatched", "not set up", "unavailable", "timed out",
)


def _is_error_result(result: str) -> bool:
    low = (result or "").lower()
    return any(m in low for m in _ERROR_MARKERS)


# A plain-text answer claiming an action was performed — used to catch the model
# hallucinating "I opened/turned/set ..." without ever calling a tool.
_ACTION_CLAIM = re.compile(
    r"\b(opened|launched|closed|quit|turned (?:up|down|on|off)|set (?:the )?"
    r"(?:volume|brightness)|muted|unmuted|increased|decreased|adjusted|lowered|"
    r"raised|took (?:a )?screenshot|captured|typed|pressed|clicked|scrolled|"
    r"moved the (?:mouse|cursor)|played|paused|skipped|locked|brightened|"
    r"dimmed|maximi[sz]ed|minimi[sz]ed|done\b|"
    # State-of-the-world claims the model narrates instead of acting, e.g.
    # "Facebook is now open in your browser" — no past-tense verb, so caught
    # here. Gated by "now" or a browser/tab context so factual answers
    # ("the store is open until 9pm") don't trip the guard.
    r"(?:is|are|it'?s|they'?re) (?:now )?(?:open|running|launched|playing)|"
    r"now (?:open|running|launched|playing)|"
    r"(?:open|opened|running|launched|playing) in (?:your|the) (?:browser|tab))",
    re.IGNORECASE)


def _claims_action(text: str) -> bool:
    return bool(_ACTION_CLAIM.search(text or ""))


# A plain-text answer refusing/excusing instead of calling a tool — e.g. "there
# was no valid tool", "I can't do that", "it doesn't require a tool".
_NO_ACTION = re.compile(
    r"(no (valid )?tool|tool to call|without (a |any )?tool|does(n't| not) "
    r"(require|need) a tool|did(n'?t| not) (perform|do|take)|no action|i can('?t|"
    r"not)|i'?m (unable|not able)|unable to|i don'?t have (a |the |any )?(tool|"
    r"ability|way)|couldn'?t find (a |the )?tool)", re.IGNORECASE)


def _refuses_action(text: str) -> bool:
    return bool(_NO_ACTION.search(text or ""))


# A plain-text answer asserting a world-STATE ("Notepad isn't running", "already
# closed", "no such window") instead of calling the tool to actually check/act.
# Only used when NO tool ran (tool_calls == 0), so a genuine tool result that
# says "not running" is never second-guessed.
_FALSE_STATE = re.compile(
    r"((is|are|it'?s|that'?s)?\s*(n'?t| not)\s+(currently\s+)?(running|open|"
    r"active|launched)|already (closed|been closed|quit|shut down|not running)|"
    r"does(n'?t| not) (appear|seem) to be (running|open)|"
    r"no (such )?(app|application|window|program|process)|"
    r"nothing to (close|open|quit))", re.IGNORECASE)


def _false_state(text: str) -> bool:
    return bool(_FALSE_STATE.search(text or ""))


class Orchestrator:
    def __init__(self, llm, tools, cfg):
        self.llm = llm
        self.tools = tools
        self.cfg = cfg
        # When a risky action awaits the user's spoken confirmation:
        # {"messages": [...react history...], "call": {...tool call...}}.
        self._pending: Optional[dict] = None
        # Name of a risky tool already approved for the current task, so a
        # corrected retry isn't asked to confirm again.
        self._confirmed: Optional[str] = None

    # ---- prompt assembly -------------------------------------------------
    def _system_prompt(self) -> str:
        from llm import _date_note  # lazy: llm imports agents

        sys = self.cfg.react_system_prompt
        if self.tools is not None:
            sys += "\n\n" + self.tools.prompt_block(instructions=False)
            sys += self.tools.folders_note()
        sys += _date_note()
        if getattr(self.llm.cfg, "disable_thinking", False):
            sys += " /no_think"
        return sys

    # ---- freshness router -----------------------------------------------
    @staticmethod
    def _needs_fresh_data(text: str) -> bool:
        """True if `text` asks about something that changes over time and so must
        be answered from a live web_search, not the model's stale weights."""
        t = text or ""
        if _WHO_WON_RE.search(t):
            return True
        if _OFFICE_RE.search(t) and (_WHO_IS_RE.search(t) or _FRESH_RE.search(t)):
            return True
        if _FRESH_RE.search(t) and _ASK_RE.search(t):
            return True
        # "score of yesterday's match", "who won last night" — recent-event
        # queries that often carry no question word for _ASK_RE to catch.
        if _RELDATE_RE.search(t) and (_ASK_RE.search(t) or _RESULT_RE.search(t)):
            return True
        return False

    def _forced_search(self, user_text: str) -> str:
        """For a time-sensitive question, run web_search now and return an
        authoritative context note the model must answer from. '' if it doesn't
        apply, web_search isn't available, or the search failed."""
        if self.tools is None or "web_search" not in getattr(
                self.tools, "_registry", {}):
            return ""
        if not self._needs_fresh_data(user_text):
            return ""
        query = f"{user_text.strip()} {datetime.datetime.now().year}"
        try:
            result = self.tools.execute(
                {"name": "web_search", "arguments": {"query": query}})
        except Exception:
            return ""
        print(f"  · web_search(forced) -> {result}", flush=True)
        if not result or result.startswith("Web search is unavailable"):
            return ""
        return (
            "Authoritative, up-to-date web_search result for the user's "
            "question. Answer ONLY from this — it overrides anything you "
            "remember or think you know, which is likely stale. If it shows the "
            "question's premise is wrong (e.g. an office that does not exist), "
            f"say so.\n{result}")

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
        if name == "run_command":
            # Only deletions need a yes/no; other commands run straight away.
            return bool(_DELETE_CMD_RE.search(str(args.get("command", ""))))
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
            cmd = str(args.get("command", ""))
            verb = "delete files with" if _DELETE_CMD_RE.search(cmd) else "run"
            return f"{verb} the command: {cmd}"
        if name == "debug_python":
            return f"run {args.get('path', 'the script')} to debug it"
        if name == "write_file":
            return f"overwrite the existing file {args.get('path', '')}"
        if name == "reset_all":
            return ("factory-reset everything — erase all memory, conversation "
                    "history, the cache, and your enrolled voice, face, and "
                    "password")
        if name == "create_tool":
            return (f"write a new tool called '{args.get('name', '')}' into my "
                    "own code")
        if name == "remove_tool":
            return f"delete the custom tool '{args.get('name', '')}'"
        return f"run {name}"

    def _confirm_question(self, call: dict) -> str:
        return f"I'm about to {self._describe(call)}. Should I go ahead?"

    # ---- the ReAct loop --------------------------------------------------
    def _run_call(self, call: dict, messages: list[dict]) -> str:
        """Execute a parsed tool call, log it, and append the observation."""
        result = self.tools.execute(call)
        print(f"  · {call['name']}({_short_args(call)}) -> {result}", flush=True)
        messages.append(
            {"role": "user", "content": f"Observation: {result}\n{_CONTINUE}"})
        return result

    def _loop(self, messages: list[dict]) -> str:
        # Track per-task: a tool already confirmed (so a retry isn't re-asked),
        # the last tool that failed (so we stop instead of looping/lying), how
        # many tools actually ran, and whether we've already nudged once against
        # a hallucinated action claim.
        last_fail: Optional[str] = None
        last_error = ""          # error text of the most recent tool, else ""
        tool_calls = 0
        nudged = False
        for _ in range(self.cfg.max_iterations):
            text = self.llm.raw_complete(messages)
            call = self.tools.parse(text) if self.tools is not None else None
            if call is None:
                answer = text.strip() or "Done."
                # If the LAST tool actually failed but the model is now claiming
                # success, override with the truth (don't let it lie).
                if last_error and not _is_error_result(answer):
                    return f"That didn't work — {last_error}"
                # Anti-hallucination: if it claims it did (or refuses/excuses an
                # action) without ever calling a tool, force one corrective retry.
                if (tool_calls == 0 and not nudged
                        and (_claims_action(answer) or _refuses_action(answer)
                             or _false_state(answer))):
                    nudged = True
                    messages.append({"role": "assistant", "content": text})
                    messages.append({"role": "user", "content": (
                        "You did NOT call any tool, so nothing happened. You DO "
                        "have tools — re-check the tool list. If the request maps "
                        "to a tool (e.g. closing an app -> close_app), output ONLY "
                        "that tool's JSON now. You cannot perform system actions "
                        "without a tool. If truly no tool fits, answer the user "
                        "directly — do not claim you did something you didn't.")})
                    continue
                return answer
            messages.append({"role": "assistant", "content": text})
            # Confirm risky actions — but don't re-ask for a tool already
            # approved for this task (e.g. a corrected retry of create_tool).
            if self._needs_confirm(call) and call["name"] != self._confirmed:
                self._pending = {"messages": messages, "call": call}
                return self._confirm_question(call)
            result = self._run_call(call, messages)
            tool_calls += 1
            if _is_error_result(result):
                last_error = result
                # Don't let it retry the same failing tool forever (and then
                # claim success): after a second failure, report it plainly.
                if last_fail == call["name"]:
                    return f"That didn't work — {result}"
                last_fail = call["name"]
            else:
                last_fail = None
                last_error = ""
        return ("I ran out of steps before finishing that — "
                "can you narrow the task or break it up?")

    def _resume_after_confirm(self, pending: dict) -> str:
        call, messages = pending["call"], pending["messages"]
        self._confirmed = call["name"]   # approved for the rest of this task
        result = self._run_call(call, messages)
        if _is_error_result(result):
            # The very action the user just approved failed — say so, don't lie.
            return f"That didn't work — {result}"
        return self._loop(messages)

    # ---- entry point -----------------------------------------------------
    def handle(self, user_text: str, context: str = "") -> str:
        # A risky action is waiting on the user's yes/no.
        if self._pending is not None:
            pending, self._pending = self._pending, None
            if _is_affirmative(user_text):
                return self._resume_after_confirm(pending)
            if _is_negative(user_text):
                self._confirmed = None
                return "Okay, I won't do that."
            # Neither yes nor no: drop the pending action and treat this as a
            # fresh request (falls through to a new task below).

        self._confirmed = None   # fresh request: re-enable confirmation

        # Freshness router: for time-sensitive questions (current officeholders,
        # who won, latest/current things) fetch live data up front rather than
        # trusting the model's stale memory, and inject it as authoritative
        # context. This guarantees fresh facts even if the model wouldn't have
        # chosen to search on its own.
        forced = self._forced_search(user_text)
        if forced:
            context = f"{context}\n\n{forced}" if context else forced

        messages = [{"role": "system", "content": self._system_prompt()}]
        # Recent conversation (no system msg) for continuity / "it"/"that" refs.
        messages += [m for m in self.llm.history[1:]]
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": user_text})
        return self._loop(messages)
