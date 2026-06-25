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
import re
import sys
import threading
import warnings

import numpy as np

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
    AuthConfig,
)
import audio_input
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
import playback

# A sentence is "complete" once we see end punctuation followed by whitespace;
# everything after stays buffered until the next boundary or end of stream.
_SENTENCE_BOUNDARY = re.compile(r"^(.*?[.!?।॥]+)(\s+)(.*)$", re.DOTALL)


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
                     stop_event: threading.Event, interrupted: threading.Event) -> None:
    """While Atlas speaks, listen for the wake word and signal an interrupt.

    Runs only during playback, when the main loop is NOT reading the mic, so it
    has exclusive access to the shared input stream. Assumes output goes to
    headphones; on open speakers Atlas's own voice could false-trigger this.
    """
    oww.reset()
    buf = np.empty(0, dtype=np.int16)
    while not stop_event.is_set():
        try:
            frame, _ = stream.read(cfg.frame_samples)
        except Exception:
            break
        buf = np.concatenate([buf, frame[:, 0]])
        while len(buf) >= cfg.wake_chunk:
            chunk, buf = buf[: cfg.wake_chunk], buf[cfg.wake_chunk:]
            scores = oww.predict(np.ascontiguousarray(chunk))
            if max(scores.values()) >= cfg.wake_threshold:
                interrupted.set()
                stop_event.set()
                return
    oww.reset()


def _respond(brain: LLM, tts: TTS, user_text: str, stream, oww, cfg: Config,
             pb_cfg: PlaybackConfig, lang: str = "") -> bool:
    """Generate, speak (streaming), and allow barge-in. Returns True if interrupted.

    `lang` is the detected input language; the reply is synthesized with the
    matching TTS voice (the LLM is already told to answer in that language).
    """
    stop_event = threading.Event()
    interrupted = threading.Event()

    watcher = None
    if pb_cfg.allow_barge_in:
        watcher = threading.Thread(
            target=_bargein_watcher,
            args=(stream, oww, cfg, stop_event, interrupted),
            daemon=True,
        )
        watcher.start()

    def audio_chunks():
        print("  atlas: ", end="", flush=True)
        buffer = ""
        for delta in brain.stream_reply(user_text):
            if stop_event.is_set():
                break
            print(delta, end="", flush=True)
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

    playback.play_stream(audio_chunks(), tts.sample_rate, stop_event)

    stop_event.set()  # tell the watcher to stop if playback ended on its own
    if watcher is not None:
        watcher.join(timeout=1.0)
    oww.reset()  # clear wake-word state before the main loop uses it again
    return interrupted.is_set()


def main() -> None:
    cfg = Config()
    llm_cfg = LLMConfig()
    tts_cfg = TTSConfig()
    tools_cfg = ToolsConfig()
    pb_cfg = PlaybackConfig()

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
        memory=memory,
        store=store,
        voiceprint_path=cfg.voiceprint_path,
        enable_coding=tools_cfg.enable_coding,
        command_timeout=tools_cfg.command_timeout,
    )

    agents_cfg = AgentsConfig()
    brain = LLM(llm_cfg, tools=tools, memory=memory, store=store, docs=docs,
                agents_cfg=agents_cfg)
    if brain.orchestrator is not None:
        confirm = "confirm risky actions" if agents_cfg.confirm_risky \
            else "fully autonomous"
        print(f"Agents: ReAct task loop (max {agents_cfg.max_iterations} "
              f"steps, {confirm}).")

    # A speaker verifier is needed for the per-turn gate AND for the startup
    # identity gate; create it if either is on. Voiceprint is loaded after any
    # first-run onboarding (below).
    auth_cfg = AuthConfig()
    need_verifier = cfg.require_speaker_match or auth_cfg.require_identity
    verifier = SpeakerVerifier(cfg) if need_verifier else None
    voiceprint = None
    if (cfg.require_speaker_match and not auth_cfg.require_identity
            and not os.path.exists(cfg.voiceprint_path)):
        raise SystemExit("No voiceprint found. Run `python enroll.py` first.")

    try:
        stream = audio_input.open_stream(cfg)
    except Exception as e:
        raise SystemExit(f"Could not open the microphone: {e}")

    # --- Startup identity gate: register on first run, then require face+voice ---
    if auth_cfg.require_identity:
        import auth
        if auth.needs_onboarding(cfg, auth_cfg, faces):
            print("First-time setup: registering your face and voice...")
            auth.onboard(cfg, auth_cfg, verifier, faces, stream, announce)
        if verifier is not None and os.path.exists(cfg.voiceprint_path):
            voiceprint = verifier.load_voiceprint()
        print("Identity check — verify your face and voice to start Atlas.")
        if not auth.authenticate(cfg, auth_cfg, verifier, voiceprint, faces,
                                 stream, vad, announce):
            for c in (stream.stop, stream.close):
                try:
                    c()
                except Exception:
                    pass
            raise SystemExit("Identity not verified — Atlas will not start. "
                             "(Set AuthConfig.require_identity=False if locked out.)")
    elif (cfg.require_speaker_match and verifier is not None
            and os.path.exists(cfg.voiceprint_path)):
        voiceprint = verifier.load_voiceprint()

    wake_name = os.path.splitext(os.path.basename(cfg.wake_model))[0] \
        if cfg.wake_model.endswith((".onnx", ".tflite")) else cfg.wake_model
    print(f"Ready. Say the wake word ('{wake_name}'). Ctrl+C to quit.\n")
    skip_wake = False   # barge-in: record immediately, no wake word
    follow_up = False   # hands-free follow-up: listen briefly without the wake word
    try:
        while True:
            if skip_wake:
                skip_wake = False
                audio = audio_input.record_until_silence(stream, vad, cfg)
            elif follow_up:
                follow_up = False
                print("(listening for follow-up...)")
                audio = audio_input.record_until_silence(
                    stream, vad, cfg, start_timeout_ms=cfg.followup_window_ms)
                if audio.size == 0:
                    print("  (no follow-up — say the wake word)\n")
                    continue  # window expired; wait for the wake word next
            else:
                audio_input.wait_for_wake_word(stream, oww, cfg)
                print("(listening...)")
                audio = audio_input.record_until_silence(stream, vad, cfg)

            if audio.size == 0:
                print("  (no speech detected)\n")
                continue

            # --- Speaker gate (personalization, NOT security; spoofable) ---
            # Only per-turn; a verifier may also exist solely for the startup
            # identity gate, which must not silently filter every command here.
            if cfg.require_speaker_match and verifier is not None \
                    and voiceprint is not None:
                is_match, score = verifier.verify(audio, voiceprint)
                if not is_match:
                    print(f"  speaker not recognized (score {score:.2f}) — ignoring.\n")
                    continue
                print(f"  speaker verified (score {score:.2f}).")

            # --- Transcribe (auto-detects language) ---
            text, lang = transcriber.transcribe(audio)
            if not text:
                print("  (no speech recognized)\n")
                continue
            print(f"  you [{lang}]: {text}")

            # --- Think + speak (with barge-in), replying in the same language ---
            if _respond(brain, tts, text, stream, oww, cfg, pb_cfg, lang):
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
        # Force the process to exit. llama-cpp / PortAudio / onnxruntime can
        # leave non-daemon threads or native resources alive that otherwise hang
        # interpreter shutdown — which would keep the embedded Qdrant lock held
        # and break memory on the next run. We've already closed everything that
        # needs flushing above, so a hard exit here is safe.
        os._exit(0)


if __name__ == "__main__":
    main()
