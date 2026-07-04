"""Stage 5: the LLM brain via llama-cpp-python (in-process, no server).

Loads a GGUF model in-process (no Ollama, no daemon). The model stays resident
in VRAM for the assistant's whole lifetime. Keeps a short rolling conversation
history so replies have context across turns, streams tokens so downstream TTS
can start speaking the first sentence early, and supports tool calling.

Tool calling uses the JSON protocol in tools.py: each turn we peek at the start
of the stream; if it's a tool-call JSON we resolve the tool(s) silently and then
stream the final spoken answer; otherwise we stream the answer directly.

Run this file directly for a standalone Step 4/6 test (typed prompts):
    python llm.py
"""

import datetime
import os
import re
import time
from typing import Iterator

from llama_cpp import Llama

# Opt-in per-stage timing (ATLAS_TIMING=1). Off by default; prints where each
# turn's seconds go so latency can be diagnosed without a profiler.
_TIMING = bool(os.environ.get("ATLAS_TIMING"))


def _lap(label: str, t0: float) -> float:
    if _TIMING:
        print(f"  [t] {label}: {time.perf_counter() - t0:.2f}s", flush=True)
    return time.perf_counter()

from config import LLMConfig

# Qwen3 emits chain-of-thought inside <think>...</think>. We suppress it with the
# "/no_think" soft switch (see __init__) and strip any residual block here, so a
# spoken reply never includes the model thinking out loud.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_TAG = re.compile(r"</?think>")


def strip_think(text: str) -> str:
    """Remove Qwen3 thinking output.

    Handles the well-formed case (<think>...</think>) and the malformed one we
    actually see with /no_think, where the model opens <think>, leaves it empty,
    and never closes it before the real content/tool-call. We drop complete
    blocks first, then any stray/unclosed think tags.
    """
    text = _THINK_BLOCK.sub("", text)
    text = _THINK_TAG.sub("", text)
    return text.strip()


def _date_note() -> str:
    """A system-prompt fragment anchoring the model in the current date."""
    today = datetime.datetime.now().strftime("%A, %B %d, %Y")
    return (
        f"\n\nToday's date is {today}. Treat your own knowledge as possibly "
        "out of date: for anything about current events, recent winners, "
        "prices, or 'latest' things, use web_search and include the current "
        "year in the query."
    )


class LLM:
    def __init__(self, cfg: LLMConfig, tools=None, memory=None, store=None,
                 docs=None, agents_cfg=None):
        self.cfg = cfg
        self.tools = tools if cfg.enable_tools else None
        self.memory = memory  # optional semantic memory (see memory.py)
        self.store = store    # optional durable state (see state.py)
        self.docs = docs      # optional RAG document store (see rag.py)
        self._turn_memory = ""  # transient recalled context for the current turn
        self._user_identity = None  # (name, role) of the current user; see main.py
        self.llm = Llama(
            model_path=cfg.model_path,
            n_gpu_layers=cfg.n_gpu_layers,
            n_ctx=cfg.n_ctx,
            verbose=False,
        )
        system = cfg.system_prompt + _date_note()
        if self.tools is not None:
            system += "\n\n" + self.tools.prompt_block()
        if cfg.disable_thinking:
            system += " /no_think"
        self.history = [{"role": "system", "content": system}]

        # Reload the last few turns from durable state so the conversation has
        # continuity across restarts.
        if self.store is not None and self.store.enabled:
            for role, content in self.store.recent_messages(self.store.cfg.load_recent):
                if role in ("user", "assistant"):
                    self.history.append({"role": role, "content": content})

        # Single model, multiple agent roles: an orchestrator drives a
        # planner -> worker -> critic pipeline using this same model (see
        # agents.py). Imported lazily to avoid a circular import.
        self.orchestrator = None
        if agents_cfg is not None and agents_cfg.enable_agents:
            from agents import Orchestrator
            self.orchestrator = Orchestrator(self, self.tools, agents_cfg)

    # ---- low-level completion -------------------------------------------
    def _complete(self) -> str:
        """Generate a full reply for the current history (think stripped).

        We collect the whole reply before deciding rather than streaming token
        by token: Qwen3's /no_think output is too inconsistent (often a
        malformed, unclosed <think>) to reliably classify mid-stream. Replies
        are short and generation is fast, and TTS-level streaming downstream
        keeps the response feeling instant regardless.
        """
        # Inject recalled memories as a transient system note (not persisted in
        # rolling history) so it informs this turn's tool rounds and answer.
        messages = self.history
        if self._turn_memory:
            messages = (
                [self.history[0], {"role": "system", "content": self._turn_memory}]
                + self.history[1:]
            )
        resp = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            stream=False,
        )
        return strip_think(resp["choices"][0]["message"].get("content") or "")

    # ---- primitives used by the agent orchestrator (agents.py) ----------
    def raw_complete(self, messages: list) -> str:
        """One non-streaming completion over an explicit message list (think
        stripped). Used for the planner/critic and as the worker's engine."""
        resp = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            stream=False,
        )
        return strip_think(resp["choices"][0]["message"].get("content") or "")

    def set_user_identity(self, name, role) -> None:
        """Tell the model who it's talking to (from the startup identity gate /
        registry) so it can answer 'who am I' about the USER, not itself."""
        self._user_identity = (name, role) if name else None

    def build_turn_context(self, user_text: str) -> str:
        """Shared per-turn context: who the user is + recalled memories +
        relevant document passages. Empty string if nothing relevant."""
        parts: list[str] = []
        if self._user_identity and self._user_identity[0]:
            name, role = self._user_identity
            ident = f"The person you are talking to is {name}"
            if role:
                ident += f" (authority level: {role})"
            ident += ('. If they ask "who am I", their name, or their '
                      "authority/role, they mean THEMSELVES — answer with this, "
                      "not a description of yourself.")
            parts.append(ident)
        if self.memory is not None:
            # Explicit notes the user asked Atlas to remember — always applied
            # (not similarity-gated), so standing preferences/corrections stick.
            notes = self.memory.notes()
            if notes:
                parts.append(
                    "Standing instructions and facts the user told you to "
                    "remember (always honor these):\n"
                    + "\n".join(f"- {n}" for n in notes)
                )
            recalled = self.memory.recall(user_text)
            if recalled:
                parts.append(
                    "Things you remember about the user and past conversations:\n"
                    + "\n".join(f"- {m}" for m in recalled)
                )
        if self.docs is not None:
            passages = self.docs.search(user_text)
            if passages:
                parts.append(
                    "Possibly-relevant excerpts from the user's own documents "
                    "(they may or may not relate to the question — use them only "
                    "if they actually answer it, and cite the source file if you "
                    "do). If they do NOT contain the answer, do NOT reply that the "
                    "information isn't in the text; instead use web_search (for "
                    "prices, products, current facts) or answer from your own "
                    "knowledge:\n"
                    + "\n".join(f"[{src}] {txt}" for txt, src in passages)
                )
        return "\n\n".join(parts)

    def tool_loop(self, system_prompt: str, task: str, context: str = "",
                  allowed: "set | None" = None) -> str:
        """Run one role's turn: role prompt + scoped tools + recent history +
        context, looping tool calls (restricted to `allowed`). Returns the
        role's answer text. Does not mutate self.history."""
        system = system_prompt
        if self.tools is not None:
            system += "\n\n" + self.tools.prompt_block(allowed)
        system += _date_note()
        if self.cfg.disable_thinking:
            system += " /no_think"

        messages = [{"role": "system", "content": system}]
        messages += self.history[1:]  # recent turns for continuity (skip system)
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": task})

        text = ""
        used_tool = False
        last_result = ""
        for _ in range(self.cfg.max_tool_rounds + 1):
            text = self.raw_complete(messages)
            call = self.tools.parse(text) if self.tools is not None else None
            if call is not None and allowed is not None and call["name"] not in allowed:
                call = None  # tool not permitted for this role
            if call is None:
                return text
            if used_tool:
                # One tool per turn: a tool already ran (which may have a side
                # effect). Don't run another — speak the first result.
                return last_result
            used_tool = True
            messages.append({"role": "assistant", "content": text})
            last_result = self.tools.execute(call)
            messages.append({
                "role": "user",
                "content": f"Tool result for {call['name']}: {last_result}\n"
                "Reply with one short spoken sentence. Do not call another tool.",
            })
        return last_result or text

    # ---- public API ------------------------------------------------------
    def stream_reply(self, user_text: str) -> Iterator[str]:
        """Resolve any tool calls, then yield the final spoken answer.

        Tool-call rounds are silent. The yielded text is the final answer (think
        stripped). It's yielded in one piece; downstream sentence-splitting +
        play_stream handle the streaming playback.
        """
        # Agent: iterative ReAct task loop (act -> observe -> repeat) on this
        # same model; pauses for spoken confirmation before risky actions.
        if self.orchestrator is not None:
            _t = time.perf_counter()
            context = self.build_turn_context(user_text)
            _t = _lap("build_context (memory recall + RAG)", _t)
            answer = self.orchestrator.handle(user_text, context)
            _lap("agent.handle (ReAct LLM loop + tools)", _t)
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": answer})
            self._trim_history()
            if answer:
                yield answer
            if self.memory is not None:
                self.memory.remember(f"User said: {user_text}\nAtlas replied: {answer}")
            if self.store is not None:
                self.store.add_message("user", user_text)
                self.store.add_message("assistant", answer)
            return

        self.history.append({"role": "user", "content": user_text})

        # Build this turn's transient context (best-effort): recalled memories +
        # relevant passages retrieved from the user's documents (RAG). Both are
        # gated by similarity, so unrelated turns inject nothing.
        self._turn_memory = ""
        context_parts: list[str] = []
        if self.memory is not None:
            recalled = self.memory.recall(user_text)
            if recalled:
                context_parts.append(
                    "Things you remember about the user and past conversations:\n"
                    + "\n".join(f"- {m}" for m in recalled)
                )
        if self.docs is not None:
            passages = self.docs.search(user_text)
            if passages:
                context_parts.append(
                    "Possibly-relevant excerpts from the user's own documents "
                    "(they may or may not relate to the question — use them only "
                    "if they actually answer it, and cite the source file if you "
                    "do). If they do NOT contain the answer, do NOT reply that the "
                    "information isn't in the text; instead use web_search (for "
                    "prices, products, current facts) or answer from your own "
                    "knowledge:\n"
                    + "\n".join(f"[{source}] {text}" for text, source in passages)
                )
        self._turn_memory = "\n\n".join(context_parts)

        for _ in range(self.cfg.max_tool_rounds + 1):
            text = self._complete()
            call = self.tools.parse(text) if self.tools is not None else None

            if call is None:
                # Final spoken answer.
                self.history.append({"role": "assistant", "content": text})
                self._trim_history()
                self._turn_memory = ""
                if text:
                    yield text
                if self.memory is not None:
                    self.memory.remember(
                        f"User said: {user_text}\nAtlas replied: {text}"
                    )
                if self.store is not None:
                    self.store.add_message("user", user_text)
                    self.store.add_message("assistant", text)
                return

            # Tool call: record it, run it, feed the result back, re-answer.
            self.history.append({"role": "assistant", "content": text})
            result = self.tools.execute(call)  # type: ignore[union-attr]
            self.history.append(
                {
                    "role": "user",
                    "content": f"Tool result for {call['name']}: {result}\n"
                    "Relay this to the user in one direct spoken sentence. No "
                    "filler, no follow-up questions.",
                }
            )

        # Exhausted tool rounds without a plain answer — fall back gracefully.
        self._turn_memory = ""
        yield "Sorry, I couldn't complete that."

    def generate(self, user_text: str) -> str:
        """Stream the reply to stdout and return the final text (for testing)."""
        parts: list[str] = []
        for delta in self.stream_reply(user_text):
            parts.append(delta)
            print(delta, end="", flush=True)
        print()
        return "".join(parts).strip()

    def _trim_history(self) -> None:
        """Keep the system prompt plus the last keep_turns user/assistant pairs."""
        keep = self.cfg.keep_turns
        if len(self.history) > keep * 2 + 1:
            self.history = [self.history[0]] + self.history[-keep * 2:]

    def reset_session(self) -> None:
        """Forget the live in-RAM conversation (used by reset_all, so a reset
        also wipes what was said this session, not just the persistent stores)."""
        self.history = [self.history[0]]   # keep only the system prompt
        self._turn_memory = ""
        if self.orchestrator is not None:
            self.orchestrator._pending = None
            self.orchestrator._confirmed = None


if __name__ == "__main__":
    import os
    from tools import Tools

    cfg = LLMConfig()
    if not os.path.exists(cfg.model_path):
        raise SystemExit(
            f"GGUF not found at {cfg.model_path}. Download the Qwen3-8B-Instruct "
            "Q4_K_M GGUF and place it there (see README)."
        )

    print(f"Loading {cfg.model_path} (n_gpu_layers={cfg.n_gpu_layers})...")
    brain = LLM(cfg, tools=Tools())
    print("Ready. Type a message (blank line or Ctrl+C to quit).\n")
    try:
        while True:
            user = input("you: ").strip()
            if not user:
                break
            print("atlas: ", end="")
            brain.generate(user)
            print()
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
