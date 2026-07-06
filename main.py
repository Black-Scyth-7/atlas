"""Main loop: wake word -> record -> verify speaker -> transcribe -> LLM -> speak.

Flow per turn:
  1. Wait for the wake word.
  2. Record the command until you stop talking (VAD).
  3. Verify it's the enrolled owner (SpeechBrain ECAPA-TDNN). If not, ignore.
  4. Transcribe to text (faster-whisper).
  5. Generate a reply (llama-cpp-python), resolving tool calls, streaming tokens.
  6. Speak the reply (Piper), streaming so the first sentence plays while the
     rest is still being generated and synthesized. Saying the wake word while
     Atlas is speaking interrupts it (barge-in) and starts a new command.

Run enroll.py once first to create the voiceprint (if the speaker gate is on).
"""

import os
import queue
import re
import sys
import threading
import time
import warnings

import numpy as np

# Opt-in per-stage latency timing (set ATLAS_TIMING=1). Prints how long each
# per-turn stage takes so a slow response can be traced to its stage.
_TIMING = bool(os.environ.get("ATLAS_TIMING"))


def _lap(label: str, t0: float) -> float:
    if _TIMING:
        print(f"  [t] {label}: {time.perf_counter() - t0:.2f}s", flush=True)
    return time.perf_counter()

# webrtcvad imports the deprecated `pkg_resources` at import time; silence that
# specific (harmless) warning so startup output stays clean.
warnings.filterwarnings("ignore", message=r".*pkg_resources is deprecated.*")
import webrtcvad

# Print Unicode (e.g. Hindi/Devanagari transcripts and replies) safely on
# consoles that default to a non-UTF-8 codepage (Windows cp1252) rather than
# crashing with UnicodeEncodeError. (reconfigure exists on TextIOWrapper at
# runtime; fetched via getattr since the static TextIO type doesn't declare it.)
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        try:
            _reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from config import (
    Config, LLMConfig, TTSConfig, PlaybackConfig, ToolsConfig, MemoryConfig,
    StateConfig, CacheConfig, RAGConfig, AgentsConfig, VisionConfig, FaceConfig,
    AuthConfig, DspConfig, CodeAgentConfig, VaultConfig,
)
import audio_input
import ui_events as ux
from speaker_id import SpeakerVerifier
from stt import Transcriber
from llm import LLM, strip_think
from tts import TTS
from tools import Tools
from memory import Memory
from state import Store
from cache import Cache
from rag import DocStore
from vision import Vision
from face_id import FaceRecognizer
from code_agent import CodeAgent
from meeting import MeetingRecorder
from audio_dsp import NoiseSuppressor, EchoCanceller, ReferenceBuffer, MicStream
import playback

# A sentence is "complete" once we see end punctuation followed by whitespace;
# everything after stays buffered until the next boundary or end of stream.
_SENTENCE_BOUNDARY = re.compile(r"^(.*?[.!?।॥]+)(\s+)(.*)$", re.DOTALL)

# Explicit "shut Atlas down" intent, detected straight from the transcript so a
# critical action doesn't depend on the LLM choosing to call the shutdown tool.
# Every branch requires a self-reference (yourself / Atlas / the assistant / the
# program) adjacent to the shutdown verb, so unrelated phrases like "turn off the
# lights" or "shut the door" never match.
# [\s,]+ separators tolerate a comma ("Atlas, shut down") from the transcript.
_SELF_SHUTDOWN_RE = re.compile(
    r"shut\s*(down|off)[\s,]+(your\s?self|atlas|the\s+(assistant|program|app))"  # shut down yourself
    r"|shut[\s,]+(your\s?self|atlas)\s+(down|off)"                             # shut yourself down
    r"|turn[\s,]+(your\s?self|atlas)\s+off"                                    # turn yourself off
    r"|turn\s+off[\s,]+(your\s?self|atlas)"                                    # turn off atlas
    r"|power\s+(down|off)[\s,]+(your\s?self|atlas)"                            # power off atlas
    r"|(your\s?self|atlas)[\s,]+(shut\s*(down|off)|power\s+(down|off))"        # atlas, shut down
    r"|(exit|quit|close)[\s,]+(your\s?self|atlas|the\s+(program|app|assistant))",  # exit atlas
    re.IGNORECASE)


def _is_self_shutdown(text: str) -> bool:
    """True if the transcript is an explicit request to shut Atlas itself down."""
    return bool(text and _SELF_SHUTDOWN_RE.search(text))


def _flush_sentences(buffer: str) -> tuple[list[str], str]:
    """Split off any complete sentences, returning (sentences, remainder)."""
    sentences = []
    while True:
        match = _SENTENCE_BOUNDARY.match(buffer)
        if not match:
            break
        sentences.append(match.group(1).strip())
        buffer = match.group(3)
    return sentences, buffer


def _bargein_watcher(stream, oww, cfg: Config,
                     stop_event: threading.Event, interrupted: threading.Event,
                     aec=None, refbuf=None) -> None:
    """While Atlas speaks, watch the mic and signal an interrupt (barge-in).

    Runs only during playback, when the main loop is NOT reading the mic, so it
    has exclusive access to the shared input stream. When an echo canceller +
    reference buffer are supplied, Atlas's own voice (the speaker output) is
    cancelled from the mic first, so it works on open speakers without
    false-triggering on itself; otherwise it assumes headphones.

    Two modes (cfg.bargein_mode):
      "speech"   - fire on near-end speech ENERGY (RMS after echo cancellation).
                   Cheap per frame, so it keeps up with real time and interrupts
                   WHILE Atlas talks. The wake model is NOT run.
      "wakeword" - require the spoken wake word via the model. Accurate, but its
                   per-chunk MFCC+ONNX can't keep pace during playback on a busy
                   CPU, so the interrupt tends to land AFTER the reply ends.
    """
    oww.reset()
    if aec is not None:
        aec.reset()
    if refbuf is not None:
        refbuf.clear()
    # Barge-in uses its OWN, more sensitive settings (see Config): fewer
    # consecutive hits and a shorter warm-up than the strict idle wake, because
    # your voice has to cut through Atlas's own (partially cancelled) speech.
    warmup = max(0, getattr(cfg, "bargein_warmup_chunks",
                            getattr(cfg, "wake_warmup_chunks", 0)))
    need = max(1, getattr(cfg, "bargein_consecutive",
                          getattr(cfg, "wake_consecutive", 1)))
    mode = getattr(cfg, "bargein_mode", "speech")
    threshold = (getattr(cfg, "bargein_speech_rms", 0.02) if mode == "speech"
                 else getattr(cfg, "bargein_threshold",
                              getattr(cfg, "wake_threshold", 0.5)))
    debug = bool(os.environ.get("ATLAS_BARGEIN_DEBUG"))
    peak = 0.0
    seen = 0
    run = 0
    buf = np.empty(0, dtype=np.int16)
    while not stop_event.is_set():
        try:
            frame, _ = stream.read_raw(cfg.frame_samples)
        except Exception:
            break
        pcm = frame[:, 0]
        if aec is not None and refbuf is not None:
            pcm = aec.cancel(pcm, refbuf.take(cfg.frame_samples))

        if mode == "speech":
            # Near-end energy: cheap enough to stay real-time, so the interrupt
            # fires the instant you start talking (no wake word needed).
            seen += 1
            score = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2))) / 32768.0
            peak = max(peak, score)
            if debug and (score > threshold * 0.5 or seen % 16 == 0):
                warm = " (warmup)" if seen <= warmup else ""
                print(f"\n[bargein] rms={score:.4f} run={run} peak={peak:.4f}"
                      f" thr={threshold:.4f}{warm}", file=sys.stderr, flush=True)
            if seen <= warmup:
                run = 0
                continue
            if score >= threshold:
                run += 1
                if run >= need:
                    if debug:
                        print(f"\n[bargein] FIRED (speech) rms={score:.4f}",
                              file=sys.stderr, flush=True)
                    interrupted.set()
                    stop_event.set()
                    return
            else:
                run = 0
            continue

        # wakeword mode: score the "Atlas" model over the rolling window.
        buf = np.concatenate([buf, pcm])
        while len(buf) >= cfg.wake_chunk:
            chunk, buf = buf[: cfg.wake_chunk], buf[cfg.wake_chunk:]
            score = oww.predict(np.ascontiguousarray(chunk))
            seen += 1
            if score > peak:
                peak = score
            if debug and (score > 0.1 or seen % 12 == 0):
                warm = " (warmup)" if seen <= warmup else ""
                print(f"\n[bargein] score={score:.3f} run={run} peak={peak:.3f}{warm}",
                      file=sys.stderr, flush=True)
            if seen <= warmup:
                run = 0
                continue
            if score >= threshold:
                run += 1
                if run >= need:
                    if debug:
                        print(f"\n[bargein] FIRED at score={score:.3f}",
                              file=sys.stderr, flush=True)
                    interrupted.set()
                    stop_event.set()
                    return
            else:
                run = 0
    if debug:
        print(f"\n[bargein] ended without firing; peak={peak:.4f} "
              f"(threshold={threshold})", file=sys.stderr, flush=True)
    oww.reset()


def _respond(brain: LLM, tts: TTS, user_text: str, stream, oww, cfg: Config,
             pb_cfg: PlaybackConfig, lang: str = "", aec=None, refbuf=None,
             bargein: bool = True) -> bool:
    """Generate, speak (streaming), and allow barge-in. Returns True if interrupted.

    `lang` is the detected input language; the reply is synthesized with the
    matching TTS voice (the LLM is already told to answer in that language).
    `aec`/`refbuf`, if given, cancel Atlas's own voice from the mic during
    barge-in (see audio_dsp.py). `bargein=False` disables the wake-word watcher
    entirely — used in text mode, where Atlas shouldn't listen at all.
    """
    stop_event = threading.Event()
    interrupted = threading.Event()

    watcher = None
    if pb_cfg.allow_barge_in and bargein:
        watcher = threading.Thread(
            target=_bargein_watcher,
            args=(stream, oww, cfg, stop_event, interrupted, aec, refbuf),
            daemon=True,
        )
        watcher.start()

    def audio_chunks():
        print("  atlas: ", end="", flush=True)
        ux.set_state("speaking")
        buffer = ""
        for delta in brain.stream_reply(user_text):
            if stop_event.is_set():
                break
            print(delta, end="", flush=True)
            ux.atlas_delta(delta)
            buffer += delta
            sentences, buffer = _flush_sentences(buffer)
            for sentence in sentences:
                spoken = strip_think(sentence)
                if spoken:
                    yield tts._synth(spoken, lang)
        print()
        if not stop_event.is_set():
            tail = strip_think(buffer)
            if tail:
                yield tts._synth(tail, lang)
        ux.atlas_done()

    playback.play_stream(audio_chunks(), tts.sample_rate, stop_event,
                         reference_sink=refbuf,
                         tail_padding_ms=pb_cfg.tail_padding_ms)

    stop_event.set()  # tell the watcher to stop if playback ended on its own
    if watcher is not None:
        watcher.join(timeout=1.0)
    oww.reset()  # clear wake-word state before the main loop uses it again
    return interrupted.is_set()


def _text_key_watcher(cfg: Config, toggle_event: threading.Event,
                      stop_event: threading.Event) -> None:
    """Set `toggle_event` on each press of the text-mode key (default F1).

    Polls the global key state (Windows GetAsyncKeyState), so it works whether or
    not the terminal is focused, without an extra dependency. Edge-triggered on
    key-down. No-op on non-Windows. F1 toggles between voice mode and text mode.
    """
    import ctypes
    import time

    try:
        get_state = ctypes.windll.user32.GetAsyncKeyState
    except Exception:
        return  # not Windows / no user32 — text mode unavailable
    vk = cfg.text_input_vk
    prev_down = False
    while not stop_event.is_set():
        down = bool(get_state(vk) & 0x8000)
        if down and not prev_down:
            toggle_event.set()
        prev_down = down
        time.sleep(0.04)


def _text_mode_read(toggle_event: threading.Event, gui_q=None):
    """Read one typed line in text mode (Windows msvcrt, char by char) OR from the
    GUI text box (gui_q).

    Returns the line on Enter, or None if F1 was pressed (leave text mode). While
    here the mic is NOT read — text mode means voice listening is off until you
    press F1 again. Echoes terminal input and supports backspace.
    """
    import msvcrt
    import time

    sys.stdout.write("  text> ")
    sys.stdout.flush()
    buf: list[str] = []
    while True:
        if toggle_event.is_set():           # F1 pressed -> exit text mode
            toggle_event.clear()
            sys.stdout.write("\n")
            sys.stdout.flush()
            return None
        if gui_q is not None:               # a line typed in the GUI box
            try:
                line = gui_q.get_nowait()
                sys.stdout.write("\n")
                sys.stdout.flush()
                return line.strip()
            except queue.Empty:
                pass
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ("\r", "\n"):
                sys.stdout.write("\n")
                sys.stdout.flush()
                return "".join(buf).strip()
            if ch == "\x03":                # Ctrl+C
                raise KeyboardInterrupt
            if ch == "\x08":                # backspace
                if buf:
                    buf.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue
            if ch in ("\x00", "\xe0"):      # function/arrow key: drop 2nd code
                if msvcrt.kbhit():
                    msvcrt.getwch()
                continue
            buf.append(ch)
            sys.stdout.write(ch)
            sys.stdout.flush()
        else:
            time.sleep(0.015)


def main() -> None:
    cfg = Config()
    llm_cfg = LLMConfig()
    tts_cfg = TTSConfig()
    tools_cfg = ToolsConfig()
    pb_cfg = PlaybackConfig()
    auth_cfg = AuthConfig()

    print("Loading models...")
    oww = audio_input.load_wake_model(cfg)
    vad = webrtcvad.Vad(cfg.vad_aggressiveness)
    transcriber = Transcriber(cfg)
    tts = TTS(tts_cfg)

    # Timers announce themselves out loud via Piper.
    def announce(msg: str) -> None:
        try:
            playback.play(tts.synthesize(msg), tts.sample_rate)
        except Exception:
            print(f"\n{msg}")

    cache = Cache(CacheConfig())
    if cache.enabled:
        print("Cache: enabled (Redis).")
    else:
        print(f"Cache: DISABLED — {cache.disabled_reason}.")

    rag_cfg = RAGConfig()
    docs = DocStore(rag_cfg, cache=cache)
    if docs.enabled and rag_cfg.auto_ingest and os.path.isdir(rag_cfg.docs_dir):
        # Index new/changed files in docs/ now (main holds the lock, so this is
        # the conflict-free place to do it). Unchanged files are skipped.
        added_files, _ = docs.ingest_dir(rag_cfg.docs_dir, force=False)
        if added_files:
            print(f"Documents: indexed {added_files} new/changed file(s).")
    if docs.enabled:
        print(f"Documents: {docs.count()} chunks indexed"
              + (" (RAG active)." if docs.count() else f" — add files to {rag_cfg.docs_dir}/."))
    else:
        print(f"Documents: DISABLED — {docs.disabled_reason}.")

    vision = Vision(VisionConfig())
    print("Vision: " + ("enabled (lazy-loaded on first screen question)."
                        if vision.available else f"DISABLED — {vision.reason}."))

    faces = FaceRecognizer(FaceConfig())
    if faces.available:
        enrolled = faces.names()
        print(f"Faces: enabled ({len(enrolled)} enrolled"
              + (f": {', '.join(enrolled)})." if enrolled
                 else " — run `python enroll_face.py <name>`)."))
    else:
        print(f"Faces: DISABLED — {faces.reason}.")

    code_agent = CodeAgent(CodeAgentConfig())
    print("Coding agent: " + ("CrewAI (delegated, isolated venv)."
          if code_agent.available else f"DISABLED — {code_agent.reason}."))

    memory = Memory(MemoryConfig(), cache=cache)
    if memory.enabled:
        print(f"Memory: enabled ({memory.count()} stored).")
    else:
        print(f"Memory: DISABLED — {memory.disabled_reason}.")

    store = Store(StateConfig())
    if store.enabled:
        print(f"State: enabled ({store.message_count()} messages logged).")
    else:
        print(f"State: DISABLED — {store.disabled_reason}.")

    agents_cfg = AgentsConfig()

    # Encrypted credential vault for website logins (optional; see vault.py).
    vault = None
    vault_cfg = VaultConfig()
    if vault_cfg.enabled:
        try:
            from vault import Vault

            vault = Vault(vault_cfg.vault_path, use_dpapi=vault_cfg.use_dpapi,
                          master_password_required=vault_cfg.master_password)
            n = len(vault.list_sites())
            print(f"Vault: enabled ({n} saved login{'s' * (n != 1)}, "
                  f"{'DPAPI + master password' if vault_cfg.master_password else 'DPAPI'}).")
        except Exception as e:
            print(f"Vault: DISABLED — {e}")
            vault = None

    tools = Tools(
        on_timer_fire=announce,
        web_search_backend=tools_cfg.web_search_backend,
        tavily_api_key=tools_cfg.tavily_api_key,
        tavily_mcp_url=tools_cfg.tavily_mcp_url,
        cache=cache,
        enable_system_control=tools_cfg.enable_system_control,
        allow_power_off=tools_cfg.allow_power_off,
        vision=vision,
        faces=faces,
        code_agent=code_agent,
        transcriber=transcriber,
        meeting_recorder=MeetingRecorder(cfg),
        memory=memory,
        store=store,
        voiceprint_path=cfg.voiceprint_path,
        password_path=auth_cfg.password_path,
        enable_coding=tools_cfg.enable_coding,
        command_timeout=tools_cfg.command_timeout,
        enable_self_extend=tools_cfg.enable_self_extend,
        plugins_dir=tools_cfg.plugins_dir,
        auto_install_deps=tools_cfg.auto_install_tool_deps,
        test_mode=auth_cfg.test_mode,
        admin_only_tools=agents_cfg.admin_only_tools,
        vault=vault,
        login_autofill_delay=vault_cfg.autofill_delay,
        use_cdp_login=vault_cfg.use_cdp_login,
    )

    brain = LLM(llm_cfg, tools=tools, memory=memory, store=store, docs=docs,
                agents_cfg=agents_cfg)
    # Local fallback for writing tool code when CrewAI isn't set up (create_tool
    # prefers the CrewAI code_agent). Wired now that the LLM exists.
    tools._code_gen = brain.raw_complete
    # Let reset_all also wipe the live in-session conversation history.
    tools._clear_session = brain.reset_session # type: ignore
    if brain.orchestrator is not None:
        confirm = "confirm risky actions" if agents_cfg.confirm_risky \
            else "fully autonomous"
        print(f"Agents: ReAct task loop (max {agents_cfg.max_iterations} "
              f"steps, {confirm}).")

    # Test mode (ATLAS_TEST_MODE in .env) bypasses ALL auth — the startup
    # identity gate (face/voice/password) and the per-turn speaker gate — so you
    # can run Atlas without enrolling anything. Default off.
    if auth_cfg.test_mode:
        print("** TEST MODE: identity gate and speaker check DISABLED "
              "(ATLAS_TEST_MODE). Not secure — for development only. **")
    require_identity = auth_cfg.require_identity and not auth_cfg.test_mode
    require_speaker = cfg.require_speaker_match and not auth_cfg.test_mode

    # A speaker verifier is needed for the per-turn gate AND for the startup
    # identity gate; create it if either is on. Voiceprint is loaded after any
    # first-run onboarding (below).
    need_verifier = require_speaker or require_identity
    verifier = SpeakerVerifier(cfg) if need_verifier else None
    voiceprint = None
    if (require_speaker and not require_identity
            and not os.path.exists(cfg.voiceprint_path)):
        raise SystemExit("No voiceprint found. Run `python enroll.py` first.")

    try:
        stream = audio_input.open_stream(cfg)
    except Exception as e:
        raise SystemExit(f"Could not open the microphone: {e}")

    # --- Audio DSP: noise suppression on capture + echo cancellation on barge-in ---
    dsp_cfg = DspConfig()
    suppressor = NoiseSuppressor() if dsp_cfg.enable_noise_suppression else None
    if dsp_cfg.enable_noise_suppression:
        ok = suppressor is not None and suppressor.available
        print("Noise suppression: " + ("enabled (RNNoise)."
              if ok else f"DISABLED — {suppressor.reason if suppressor else 'off'}."))
    aec = refbuf = None
    if dsp_cfg.enable_echo_cancellation:
        aec, refbuf = EchoCanceller(dsp_cfg), ReferenceBuffer(dsp_cfg)
        print("Echo cancellation: enabled (cancels Atlas's own voice on barge-in).")
    stream = MicStream(stream, suppressor)

    # --- Startup identity gate: register on first run, then require face+voice ---
    session_role = "admin"   # who's driving this session; gates admin-only tools
    session = None
    if require_identity:
        import auth
        if auth.needs_onboarding(cfg, auth_cfg, faces):
            print("First-time setup: registering your face and voice...")
            auth.onboard(cfg, auth_cfg, verifier, faces, stream, announce)
        if verifier is not None and os.path.exists(cfg.voiceprint_path):
            voiceprint = verifier.load_voiceprint()
        print("Identity check — verify your face and voice to start Atlas.")
        session = auth.authenticate(cfg, auth_cfg, verifier, voiceprint, faces,
                                    stream, vad, announce)
        if not session:
            for c in (stream.stop, stream.close):
                try:
                    c()
                except Exception:
                    pass
            raise SystemExit("Identity not verified — Atlas will not start. "
                             "(Set AuthConfig.require_identity=False if locked out.)")
        session_role = session.get("role") or "admin"
        if session.get("voiceprint") is not None:
            voiceprint = session["voiceprint"]   # per-turn gate uses their print
        print(f"Signed in as {session.get('name')} (authority: {session_role}).")
    elif (require_speaker and verifier is not None
            and os.path.exists(cfg.voiceprint_path)):
        voiceprint = verifier.load_voiceprint()

    # Apply the session's authority to the tools (test mode leaves it fully
    # unrestricted; a disabled gate / no identity means the owner = admin).
    tools.set_authority("admin" if auth_cfg.test_mode else session_role)

    # Tell the model who the user is so it can answer "who am I". Prefer the
    # authenticated session; otherwise fall back to the registry (admin/first
    # user), or a single enrolled face — works even in test mode.
    import auth as _auth
    owner_name, owner_role = None, session_role
    if session:
        owner_name = session.get("name")
        owner_role = session.get("role") or session_role
    else:
        owner_name = _auth.admin_name(auth_cfg) or _auth._first_user(auth_cfg)
        if owner_name:
            owner_role = _auth._load_users(auth_cfg).get(
                owner_name, {}).get("role", "admin")
        elif faces is not None and getattr(faces, "available", False):
            enrolled = [n for n in faces.names() if n and n.lower() != "owner"]
            if len(enrolled) == 1:
                owner_name, owner_role = enrolled[0], "admin"
    if owner_name and owner_name.lower() != "owner":
        brain.set_user_identity(owner_name, owner_role)
        print(f"User identity: {owner_name} (authority: {owner_role}).")

    # "Register a new user" at runtime (admin-only). After registering, if we
    # still don't know who the user is, adopt the new admin/first user so
    # "who am I" works without a restart.
    def _register_and_adopt(name, role):
        msg = _auth.register_new_user(cfg, auth_cfg, verifier, faces, stream,
                                      announce, name, role)
        if brain._user_identity is None:
            who = _auth.admin_name(auth_cfg) or name
            r = _auth._load_users(auth_cfg).get(who, {}).get("role", role)
            if who and who.lower() != "owner":
                brain.set_user_identity(who, r)
        return msg
    tools._register_user_fn = _register_and_adopt

    wake_name = os.path.splitext(os.path.basename(cfg.wake_model))[0] \
        if cfg.wake_model.endswith((".onnx", ".tflite")) else cfg.wake_model

    # Secondary input: F1 TOGGLES between voice mode and text mode. A background
    # watcher sets toggle_event on each F1 press; the loop switches modes. In
    # text mode the mic isn't read at all (voice listening is off until F1).
    toggle_event = threading.Event()
    text_stop = threading.Event()
    text_input_on = cfg.enable_text_input and os.name == "nt"
    if text_input_on:
        threading.Thread(target=_text_key_watcher,
                         args=(cfg, toggle_event, text_stop), daemon=True).start()

    # GUI text box (gui.py): typed commands land in this queue and are read in
    # text mode, exactly like terminal typing. The box only appears after F1.
    gui_text_q: "queue.Queue[str]" = queue.Queue()
    ux.set_input_handler(gui_text_q.put)

    hint = " or press F1 to type" if text_input_on else ""
    print(f"Ready. Say the wake word ('{wake_name}'){hint}. Ctrl+C to quit.\n")
    ux.status(user=owner_name or "—", authority=owner_role,
              model=os.path.splitext(os.path.basename(llm_cfg.model_path))[0],
              wake_word=wake_name, mic=True)
    ux.set_state("idle")
    ux.ready()      # online + ready: the GUI reveals its (pre-populated) window now
    if cfg.startup_phrase.strip():
        announce(cfg.startup_phrase)   # speak the ready greeting aloud
    skip_wake = False   # barge-in: record immediately, no wake word
    follow_up = False   # hands-free follow-up: listen briefly without the wake word
    text_mode = False   # F1 toggles this: in text mode, voice listening is off
    restart = False     # reset_all sets this to relaunch the process
    try:
        while True:
            typed_text = None
            audio = np.empty(0, dtype=np.float32)  # set by the voice branches below

            if text_mode:
                # --- Text mode: type commands (terminal or GUI box); voice OFF. ---
                line = _text_mode_read(toggle_event, gui_text_q)
                if line is None:                   # F1 pressed -> back to voice
                    text_mode = False
                    ux.text_mode(False)            # GUI: hide the text box
                    print("  (voice mode — say the wake word)\n")
                    continue
                if not line:
                    continue                       # empty line -> stay in text mode
                typed_text = line
            elif skip_wake:
                skip_wake = False
                ux.set_state("listening")
                audio = audio_input.record_until_silence(
                    stream, vad, cfg,
                    start_timeout_ms=cfg.command_start_timeout_ms)
            elif follow_up:
                follow_up = False
                print("(listening for follow-up...)")
                ux.set_state("listening")
                audio = audio_input.record_until_silence(
                    stream, vad, cfg, start_timeout_ms=cfg.followup_window_ms)
                if audio.size == 0:
                    print("  (no follow-up — say the wake word)\n")
                    continue  # window expired; wait for the wake word next
            else:
                # Drop any audio buffered while Atlas was speaking (incl. its
                # own greeting/replies) so it can't wake itself. Then wait for
                # the wake word, or F1 to switch into text mode.
                if hasattr(stream, "drain"):
                    stream.drain()
                oww.reset()
                ux.set_state("idle")
                woke = audio_input.wait_for_wake_word(
                    stream, oww, cfg, interrupt=toggle_event if text_input_on else None)
                if not woke:                       # F1 pressed -> enter text mode
                    toggle_event.clear()
                    text_mode = True
                    while not gui_text_q.empty():  # drop any stale queued input
                        try:
                            gui_text_q.get_nowait()
                        except queue.Empty:
                            break
                    ux.text_mode(True)             # GUI: show + focus the text box
                    print("  (text mode — type commands; press F1 for voice)")
                    continue
                print("(listening...)")
                ux.set_state("listening")
                audio = audio_input.record_until_silence(
                    stream, vad, cfg,
                    start_timeout_ms=cfg.command_start_timeout_ms)

            if typed_text is not None:
                # --- Typed command: skip recording, speaker gate, and STT ---
                text, lang = typed_text, (cfg.stt_language or "en")
                print(f"  you [text]: {text}")
                ux.set_state("thinking")
                ux.user_said(text, lang)
            else:
                if audio.size == 0:
                    print("  (no speech detected)\n")
                    if cfg.not_understood_phrase.strip():
                        announce(cfg.not_understood_phrase)
                    continue

                # --- Speaker gate (personalization, NOT security; spoofable) ---
                # Only per-turn; a verifier may also exist solely for the startup
                # identity gate, which must not silently filter every command here.
                _t = time.perf_counter()
                if require_speaker and verifier is not None \
                        and voiceprint is not None:
                    is_match, score = verifier.verify(audio, voiceprint)
                    if not is_match:
                        print(f"  speaker not recognized (score {score:.2f}) — ignoring.\n")
                        continue
                    print(f"  speaker verified (score {score:.2f}).")
                    _t = _lap("speaker verify", _t)

                # --- Transcribe (auto-detects language) ---
                ux.set_state("thinking")
                text, lang = transcriber.transcribe(audio)
                _lap("stt (transcribe)", _t)
                if not text:
                    print("  (no speech recognized)\n")
                    if cfg.not_understood_phrase.strip():
                        announce(cfg.not_understood_phrase)
                    continue
                print(f"  you [{lang}]: {text}")
                ux.user_said(text, lang)

            # --- Deterministic self-shutdown (don't rely on the LLM tool call) ---
            # "shut down yourself" etc. is a critical, hard-to-recover action, so
            # handle it here from the transcript directly. The LLM was observed to
            # sometimes just SAY "Shutting down." without calling shutdown_self,
            # leaving Atlas running. This guarantees it exits. (Requires an explicit
            # self-reference so "turn off the lights" won't match.)
            if _is_self_shutdown(text):
                print("\nShutting down (requested).")
                tools.shutdown_requested = True
                if cfg.shutdown_phrase.strip():    # guaranteed goodbye (not the LLM)
                    announce(cfg.shutdown_phrase)
                break

            # --- Think + speak. In text mode, no barge-in (voice stays off). ---
            _t = time.perf_counter()
            interrupted = _respond(brain, tts, text, stream, oww, cfg, pb_cfg,
                                   lang, aec=aec, refbuf=refbuf,
                                   bargein=(typed_text is None))
            _lap("respond (LLM + TTS + playback)", _t)
            if tools.restart_requested:            # reset_all asked to restart
                restart = True
                break
            if tools.shutdown_requested:           # user asked Atlas to shut itself down
                print("\nShutting down (requested).")
                if cfg.shutdown_phrase.strip():    # guaranteed goodbye (not the LLM)
                    announce(cfg.shutdown_phrase)
                break
            if typed_text is not None:
                print()
                continue                           # stay in text mode
            if interrupted:
                print("  (interrupted — listening)\n")
                skip_wake = True
            elif cfg.enable_followup:
                follow_up = True  # open a hands-free window after speaking
            print()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        # Release resources, each guarded so one failure can't block the rest.
        # Order matters: flush/close the memory store so its lock is released.
        for cleanup in (
            text_stop.set,
            tools.cancel_all,
            memory.close,
            store.close,
            docs.close,
            cache.close,
            stream.stop,
            stream.close,
        ):
            try:
                cleanup()
            except Exception:
                pass
        # Relaunch (after a reset) now that locks/mic are released, so onboarding
        # runs fresh; otherwise force-exit. We've already closed everything that
        # needs flushing above. (llama-cpp / PortAudio / onnxruntime can leave
        # native threads alive that would otherwise hang interpreter shutdown and
        # keep the embedded Qdrant lock held.)
        if restart:
            try:
                print("\nRestarting Atlas...\n", flush=True)
                os.execv(sys.executable, [sys.executable, *sys.argv])
            except Exception:
                pass
        os._exit(0)


def _want_gui() -> bool:
    """Launch the holographic GUI instead of the terminal app?

    True if `--gui` is passed on the command line OR ATLAS_GUI is set truthy
    (in the environment or .env). `--no-gui` forces the terminal app even if the
    env var is on. The GUI runs this same loop unchanged on a worker thread.
    """
    argv = sys.argv[1:]
    if "--no-gui" in argv:
        return False
    if "--gui" in argv:
        return True
    return os.environ.get("ATLAS_GUI", "").strip().lower() in ("1", "true", "yes", "on")


if __name__ == "__main__":
    if _want_gui():
        import gui
        gui.main()          # opens the window; runs main() on a worker thread
    else:
        main()
