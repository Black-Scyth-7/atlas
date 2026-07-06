"""Tiny decoupled event bus so a UI (see gui.py) can reflect the assistant's
live state without the core loop knowing anything about the UI.

The main loop / audio / playback just call the emitters below (set_state,
user_said, atlas_delta, ...). A UI registers ONE subscriber via subscribe();
each event is delivered to it as (kind, data-dict). With NO subscribers every
emit is a cheap no-op, so the terminal app (`python main.py`) is unaffected.

Threading: emits happen on whatever thread the loop runs on (a worker thread
under the GUI). Subscribers must be thread-safe — the Qt bridge in gui.py simply
re-emits a Qt signal, which Qt queues onto the GUI thread for us.
"""

from __future__ import annotations

from typing import Callable

# Valid assistant states (also the orb's color keys in GuiConfig.state_colors).
STATES = ("idle", "listening", "thinking", "speaking")

_subs: list[Callable[[str, dict], None]] = []


def subscribe(callback: Callable[[str, dict], None]) -> None:
    """Register callback(kind: str, data: dict). Ignores duplicates."""
    if callback not in _subs:
        _subs.append(callback)


def unsubscribe(callback: Callable[[str, dict], None]) -> None:
    if callback in _subs:
        _subs.remove(callback)


def _emit(kind: str, **data) -> None:
    # Iterate a copy so a subscriber can (un)subscribe during dispatch, and never
    # let a broken subscriber crash the assistant.
    for cb in list(_subs):
        try:
            cb(kind, data)
        except Exception:
            pass


# ---- convenience emitters used by the loop ------------------------------
def set_state(state: str) -> None:
    """idle | listening | thinking | speaking."""
    _emit("state", state=state)


def user_said(text: str, lang: str = "") -> None:
    _emit("user", text=text, lang=lang)


def atlas_delta(text: str) -> None:
    """A streamed fragment of Atlas's spoken reply (appended to the live line)."""
    _emit("delta", text=text)


def atlas_done() -> None:
    """Atlas finished the current reply (close off the transcript line)."""
    _emit("done")


def audio_level(rms: float) -> None:
    """Live mic/speech loudness in [0, ~1], drives the orb's pulse."""
    _emit("level", rms=float(rms))


def status(**kv) -> None:
    """One-off readouts for the status bar: user, authority, model, wake_word, mic."""
    _emit("status", **kv)


def text_mode(on: bool) -> None:
    """Entered/left F1 text mode — the GUI shows/hides its type-in box."""
    _emit("text_mode", on=bool(on))


def ready() -> None:
    """Atlas has finished loading models + identity and is online and ready.
    A UI may reveal its window now (it stays hidden until this fires)."""
    _emit("ready")


def tool_activity(name: str, args: str = "", status: str = "run") -> None:
    """A tool call is happening. `status` is one of:
        "run"  — the tool has just started executing,
        "ok"   — it finished successfully,
        "fail" — it raised,
        "deny" — it was blocked (e.g. an admin-only tool for a non-admin).
    `args` is a short human-readable argument preview. Lets a UI show which
    tool Atlas is reaching for right now. No-op with no UI."""
    _emit("tool", name=str(name), args=str(args), status=str(status))


def llm_activity(trace: list) -> None:
    """Real per-token generation trace for the neural-activity panel.

    `trace` is a list (in generation order) of dicts:
        {"t": token, "p": confidence 0..1, "e": entropy 0..1,
         "k": [[candidate_token, prob], ...]}
    All values are derived from the model's own logits (softmax over the
    next-token distribution) for the reply just generated — not decoration.
    A UI can play the trace back to visualise how sure/uncertain the model was
    token-by-token and which alternatives it weighed. No-op with no UI."""
    _emit("llm", trace=trace)


# ---- reverse channel: UI -> assistant (typed commands) ------------------
_input_handler: Callable[[str], None] | None = None


def set_input_handler(callback: Callable[[str], None] | None) -> None:
    """The loop registers a handler that injects typed text as a command."""
    global _input_handler
    _input_handler = callback


def submit_text(text: str) -> None:
    """Called by the GUI's text box to send a typed command to Atlas.
    No-op if the loop hasn't registered a handler (e.g. terminal mode)."""
    text = (text or "").strip()
    if text and _input_handler is not None:
        try:
            _input_handler(text)
        except Exception:
            pass
