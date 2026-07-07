"""Central configuration for Atlas, the local voice assistant.

Every tunable lives here in dataclasses. Nothing about sample rates, model
paths, thresholds, or prompts should be hardcoded anywhere else.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

# Load a local .env (next to this file) so credentials like the Tavily key work
# in any shell without setx / restarting your terminal. Values in .env win over
# stale environment variables. Keep .env out of version control.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).with_name(".env"), override=True)
except ImportError:
    pass


@dataclass
class Config:
    # --- Audio I/O (one shared 16 kHz mono InputStream for the whole app) ---
    sample_rate: int = 16000        # shared by wake word, VAD, STT, and speaker ID
    frame_ms: int = 30              # webrtcvad frame size (must be 10, 20, or 30)
    wake_chunk: int = 1280          # fed to the detector in 80 ms chunks (1280 samples)

    # --- Wake word (custom PyTorch MatchboxNet; see wake_pytorch/) ---
    # "Atlas" detector trained locally with wake_pytorch/train.py and exported to
    # ONNX. The runtime WakeDetector (wake_pytorch/detector.py) runs it via
    # onnxruntime over a rolling 1.5 s mic window. Single-word wake words are the
    # hardest case, so tune wake_threshold against your mic/environment (see
    # wake_pytorch/eval.py for a recall-vs-false-wakes/hour sweep).
    wake_model: str = "models/atlas.onnx"
    # 0.75: your enrolled voice scores >=0.998 on "Atlas" (huge margin) while the
    # hardest sound-alike in your own speech peaks ~0.63, so this cleanly rejects
    # your chatter with zero recall cost. Lower toward 0.5 if a quiet/far "Atlas"
    # ever gets missed on a live mic; raise to reduce false triggers further.
    wake_threshold: float = 0.75
    # Speech-energy gate (frame RMS, 0..1): the detector ignores frames quieter
    # than this so a wake can only fire on an audible burst. The MatchboxNet
    # scores the QUIET DECAY TAIL of any preceding sound very high (a word ends,
    # its energy leaves the rolling window, and a near-silent frame reads as
    # "Atlas") — that is the "wakes on any word" false trigger. Real "Atlas"
    # fires at rms >0.05; live false wakes fire at rms ~0.009, so 0.02 rejects
    # them with a wide margin (observed live false wakes peaked at rms 0.013, so
    # 0.015 blocks every one while real spoken "Atlas" fires at rms 0.05-0.36).
    # Set 0.0 to disable (raw model). Raise if faint room noise still triggers;
    # lower toward 0.01 if a soft/distant "Atlas" is ever missed.
    # NOTE: this gate must sit ABOVE your room's background-noise RMS. 0.015 was
    # tuned for a quiet room; in a noisier room the decay-tail after ANY word
    # clears the gate and false-wakes ("wakes on every word from idle" — verified:
    # noise at rms 0.03 fired 6/60 clips at gate 0.015, but 0 at gate 0.05).
    # 0.025 gives some noise margin while keeping recall; RAISE toward 0.04-0.05
    # if it still false-wakes from idle, LOWER toward 0.015 if a normal "Atlas"
    # gets missed. Measure your room's floor with wake_pytorch/live_mic_test.py.
    wake_min_rms: float = 0.025
    # The detector's rolling window fills over the first ~1.5 s after a reset, so
    # early scores see mostly zero-padded audio; ignore detections during this
    # warm-up of 80 ms chunks so Atlas doesn't wake itself at startup.
    wake_warmup_chunks: int = 12
    # Require this many CONSECUTIVE over-threshold chunks to fire. A real "atlas"
    # spans several chunks; a lone spike from silence/transient noise doesn't.
    wake_consecutive: int = 3
    # --- Barge-in wake detection (interrupt mid-reply) ---
    # ON/OFF switch for barge-in is PlaybackConfig.allow_barge_in (set False to
    # disable entirely). The settings below only TUNE it when it's on.
    # Detecting "atlas" WHILE Atlas is speaking is harder than from idle: your
    # voice competes with Atlas's own (partially echo-cancelled) voice in the mic.
    # So barge-in gets its OWN, more sensitive settings, decoupled from the strict
    # idle thresholds above (a missed barge-in is worse than a rare early stop).
    # A shorter warm-up also shrinks the ~1 s dead-zone at the start of each reply.
    bargein_threshold: float = 0.5      # lower than wake_threshold (more sensitive)
    bargein_consecutive: int = 2        # fewer consecutive chunks than wake
    bargein_warmup_chunks: int = 4      # shorter dead-zone at reply start
    # HOW barge-in detects you:
    #   "speech"   - fire when YOU start talking (near-end RMS after echo
    #                cancellation). Cheap, so it stays real-time and interrupts
    #                WHILE Atlas speaks. You don't need to say the wake word.
    #   "wakeword" - run the "Atlas" model during playback. Accurate, but its
    #                per-chunk MFCC+ONNX can't keep up on a busy CPU, so the
    #                interrupt lands AFTER the reply finishes (observed lag).
    # Speech mode relies on decent echo cancellation (run calibrate_aec.py) or it
    # self-triggers on Atlas's own voice.
    bargein_mode: str = "speech"
    # Near-end RMS (0..1, post-AEC) that counts as you speaking, in speech mode.
    # Real speech is ~0.05-0.2; RAISE if Atlas's echo self-interrupts it, LOWER if
    # your interrupt is missed. Watch live values with ATLAS_BARGEIN_DEBUG=1.
    bargein_speech_rms: float = 0.02
    # Set env ATLAS_BARGEIN_DEBUG=1 to print live barge-in scores while Atlas
    # speaks, so you can tune the threshold to your mic/echo.

    # --- Recording / voice activity detection (webrtcvad) ---
    silence_tail_ms: int = 800      # stop after this much trailing silence
    # Minimum time to keep recording AFTER speech starts, even if you pause — so a
    # short gap mid-command doesn't cut you off early. The silence-tail stop only
    # takes effect once at least this much has been recorded. 0 disables.
    min_command_ms: int = 3000
    max_command_ms: int = 12000     # hard cap on a single command
    vad_aggressiveness: int = 2     # 0 (lenient) .. 3 (aggressive)
    preroll_frames: int = 5         # ~150 ms of lead-in kept before speech starts
    # After the wake word fires, how long to wait for you to START speaking. If
    # you begin within this window, Atlas records until you go silent (see
    # silence_tail_ms); if the window passes in silence, it stops and returns to
    # waiting for the wake word instead of hanging for the full max_command_ms.
    command_start_timeout_ms: int = 3000

    # --- Hands-free follow-up ---
    # After Atlas replies, keep listening this long for a follow-up command
    # without needing the wake word again. If you stay silent, it returns to
    # waiting for the wake word.
    enable_followup: bool = True
    followup_window_ms: int = 6000  # how long to wait for a follow-up to begin

    # --- Text input (secondary input) ---
    # Press this key (default F1) at any time to type a command instead of
    # speaking it; the typed text runs through the same LLM/response pipeline.
    # Windows-only (uses GetAsyncKeyState). Set enable_text_input False to skip.
    enable_text_input: bool = True
    text_input_vk: int = 0x70       # virtual-key code (0x70 = F1)

    # --- Startup phrase ---
    # Spoken aloud once Atlas finishes loading and is ready. "" = silent start.
    startup_phrase: str = "Atlas is online and ready."

    # --- Shutdown phrase ---
    # Spoken aloud (guaranteed, by main.py — not the LLM) when Atlas is asked to
    # shut itself down via the shutdown_self tool, so you always hear it before
    # the program exits. "" = exit silently.
    shutdown_phrase: str = "Goodbye."

    # --- "Didn't get it" phrase ---
    # Spoken when Atlas wakes but hears nothing / can't transcribe the command,
    # so you get feedback instead of silence. "" = stay silent.
    not_understood_phrase: str = "Sorry, I didn't catch that."

    # --- Speaker verification (SpeechBrain ECAPA-TDNN, CPU) ---
    # NOTE: this gate is PERSONALIZATION, not security. It only checks whether a
    # clip sounds like the enrolled owner; it is trivially spoofable by a
    # recording of your voice. Do not rely on it to keep anyone out.
    speaker_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    speaker_device: str = "cpu"     # spec: run speaker ID on CPU
    voiceprint_path: str = "voiceprint.npy"   # written by enroll.py
    speaker_threshold: float = 0.30  # cosine similarity cutoff; tune per-mic
    require_speaker_match: bool = True  # set False to disable the gate entirely

    # --- Speech-to-text (faster-whisper, GPU) ---
    stt_model: str = "medium"        # tiny / base / small / medium / large-v3
    stt_device: str = "cuda"        # "cuda" (GPU, fp16) or "cpu" (int8)
    stt_compute_type: str = "float16"  # "float16" on GPU, "int8" on CPU
    stt_language: str = "en"        # English only ("" would auto-detect any language)
    # GPU 'small' fp16 is fast (~0.7 s) and leaves ~1.4 GB VRAM free alongside the
    # resident 8B LLM (measured). beam_size>1 weighs alternatives (better proper
    # nouns); the initial_prompt biases decoding toward these brand/app spellings
    # so e.g. "GitHub" isn't heard as "gate hub". stt.py falls back to CPU int8 if
    # the GPU load fails. Needs the nvidia-*-cu12 CUDA libs (see requirements.txt).
    stt_beam_size: int = 5
    stt_initial_prompt: str = (
        "GitHub, YouTube, Spotify, Atlas, VS Code, Chrome, Edge, Notepad, "
        "Gmail, WhatsApp, Reddit, LinkedIn, Discord, Google, ChatGPT."
    )
    # --- Anti-hallucination guards (Whisper invents text on silence/noise) ---
    # vad_filter runs faster-whisper's built-in Silero VAD to strip non-speech
    # before decoding — the single most effective guard. The thresholds make
    # Whisper mark low-confidence / silent regions as no-speech so they're
    # dropped instead of turned into phantom words (e.g. an echo of the brand
    # names in stt_initial_prompt, or repeated "thank you"). Loosen only if real
    # speech is being cut: lower no_speech_threshold, or set stt_vad_filter False.
    stt_vad_filter: bool = True
    stt_no_speech_threshold: float = 0.6      # >this no_speech_prob => treat as silence
    stt_log_prob_threshold: float = -1.0      # avg_logprob below => low-confidence
    stt_compression_ratio_threshold: float = 2.4  # >this => repetitive gibberish
    # Reject near-silent clips (a breath, a click, room noise that slips past the
    # webrtcvad recorder) before decoding — those are what Whisper turns into
    # confident caption phrases. Real speech is rms ~0.05-0.2; measured a
    # hallucinating clip at 0.017. 0.02 is a safe floor; lower it (e.g. 0.01) if a
    # genuinely soft command is ever rejected, or 0.0 to disable this check.
    stt_min_rms: float = 0.02
    # Whisper was trained on YouTube captions, so on short/low-content audio it
    # emits caption-style phrases ("thanks for watching", "bye", "that's all
    # there is to it"). A transcript whose EVERY sentence matches one of these
    # (case/punctuation/apostrophe-insensitive) is dropped as a hallucination.
    # Whole-phrase match only: "thank you atlas" or "bye atlas" still get through
    # since they carry extra words. Add any new phantom phrases you observe.
    stt_hallucination_phrases: tuple = (
        "you", "thank you", "thanks", "thank you very much", "thank you so much",
        "thanks for watching", "thank you for watching", "thanks for listening",
        "please subscribe", "like and subscribe", "dont forget to subscribe",
        "subscribe to my channel", "see you next time", "see you in the next video",
        "ill see you next time", "see you", "bye", "bye bye", "goodbye",
        "okay", "ok", "yeah", "yep", "so", "um", "uh", "hmm", "mm hmm",
        "thats it", "and thats it", "thats all", "thats all there is to it",
        "yep thats all there is to it", "have a great day", "the end", "alright",
    )

    @property
    def frame_samples(self) -> int:
        """Samples per VAD frame at the configured rate (e.g. 480 @ 16 kHz/30 ms)."""
        return self.sample_rate * self.frame_ms // 1000


@dataclass
class LLMConfig:
    # --- LLM brain (llama-cpp-python, in-process, no server) ---
    # Qwen3 8B Instruct at Q4_K_M. Download the GGUF (e.g. from a bartowski repo
    # on Hugging Face) and place it at this path. Needs ~6 GB VRAM, stays
    # resident the whole session.
    model_path: str = "models/Qwen3-8B-Q4_K_M.gguf"
    n_gpu_layers: int = -1          # -1 = offload all layers to GPU
    n_ctx: int = 8192
    max_tokens: int = 512
    temperature: float = 0.7
    keep_turns: int = 8             # rolling history: system prompt + last N turns
    disable_thinking: bool = True   # Qwen3: append "/no_think" to skip <think> output
    enable_tools: bool = True       # timers, time/date, web search (see tools.py)
    max_tool_rounds: int = 3        # cap tool calls per turn to avoid loops
    system_prompt: str = (
        "You are Atlas, a concise, helpful local voice assistant. "
        "Always respond in English. "
        "Reply in one or two short, natural sentences — they will be spoken aloud. "
        "Be direct: state the result and stop. Do not add filler, do not ask the "
        "user to 'let you know', and never claim a status you cannot actually "
        "observe (you can't see whether a timer is still running). "
        "When the user tells you a personal fact or preference (their name, "
        "pets, likes), briefly acknowledge it and remember it — it is saved "
        "automatically; never say you can't access something they just told you. "
        "When the context includes remembered facts or excerpts from the user's "
        "documents, answer from them and never claim you lack access to "
        "information that appears there. "
        "When a tool would help (timers, the current time/date, web search), use "
        "it rather than guessing."
    )


@dataclass
class TTSConfig:
    # --- Text-to-speech (Piper) ---
    # English only. (The voices map still supports per-language voices if you
    # ever re-enable multilingual STT — add a row + download the Piper voice;
    # all voices must share `sample_rate`, 22.05 kHz for "medium" voices.)
    default_lang: str = "en"
    sample_rate: int = 22050
    voices: dict = field(default_factory=lambda: {
        "en": ("models/piper/en_US-amy-medium.onnx",
               "models/piper/en_US-amy-medium.onnx.json"),
    })


@dataclass
class ToolsConfig:
    # web_search backend:
    #   "auto"        - Tavily REST if a key is available, else DuckDuckGo.
    #   "tavily_rest" - Tavily REST API (one round-trip, synthesized answer).
    #   "tavily_mcp"  - Tavily via its MCP server (uses the `mcp` SDK).
    #   "duckduckgo"  - free, no key.
    web_search_backend: str = "auto"

    # System control (volume, media, apps, websites, power). allow_power_off
    # gates shutdown/restart specifically — off by default so a misheard command
    # can't power down the machine (lock/sleep stay available).
    enable_system_control: bool = True
    allow_power_off: bool = True

    # Coding tools: read/write/edit files and run commands (write_file, read_file,
    # edit_file, list_dir, run_command). run_command and overwriting an existing
    # file are gated by spoken confirmation in the agent. command_timeout caps how
    # long a run_command may take (seconds).
    enable_coding: bool = True
    command_timeout: int = 60

    # Self-extension: let Atlas write new tools/functions of its own and load
    # them permanently. Each new tool is a file in plugins_dir, auto-loaded at
    # startup and registered live when created. create_tool/remove_tool are
    # gated by spoken confirmation (they add/remove code that runs in-process).
    enable_self_extend: bool = True
    plugins_dir: str = "plugins"
    # Auto-install third-party packages a created tool declares (`# pip: ...`)
    # into Atlas's venv so the new tool works immediately.
    auto_install_tool_deps: bool = True

    # Tavily credentials, read from the environment (the MCP URL embeds the key,
    # so it doubles as a key source for the REST backend):
    #   setx TAVILY_API_KEY "tvly-..."
    #   setx ATLAS_TAVILY_MCP_URL "https://mcp.tavily.com/mcp/?tavilyApiKey=tvly-..."
    tavily_api_key: str = field(
        default_factory=lambda: os.environ.get("TAVILY_API_KEY", "")
    )
    tavily_mcp_url: str = field(
        default_factory=lambda: os.environ.get("ATLAS_TAVILY_MCP_URL", "")
    )


@dataclass
class MemoryConfig:
    # Semantic long-term memory. Qdrant runs embedded (in-process, on-disk) — no
    # server daemon — and fastembed produces the vectors. Best-effort: if it
    # can't start, Atlas runs without memory.
    enable_memory: bool = True
    qdrant_path: str = "qdrant_data"          # embedded store directory
    collection: str = "atlas_memory"
    embed_model: str = "BAAI/bge-small-en-v1.5"  # 384-d, CPU-fast
    recall_k: int = 4                          # memories injected per turn
    score_threshold: float = 0.45             # ignore weak matches (0..1 cosine)


@dataclass
class RAGConfig:
    # Retrieval-augmented generation over your own documents (txt/md/pdf).
    # Stored in a SEPARATE embedded Qdrant collection from conversation memory.
    # Exposed to the LLM as the `search_documents` tool. Index with ingest.py.
    enable_rag: bool = True
    auto_ingest: bool = True             # index new/changed docs_dir files at startup
    qdrant_path: str = "qdrant_docs"     # separate store from memory's qdrant_data
    collection: str = "atlas_docs"
    embed_model: str = "BAAI/bge-small-en-v1.5"  # same model as memory
    docs_dir: str = "docs"               # default folder ingest.py reads
    chunk_chars: int = 800               # target characters per chunk
    chunk_overlap: int = 150             # overlap between consecutive chunks
    top_k: int = 5                       # chunks returned per search
    # bge-small cosine scores cluster in a narrow band (~0.45-0.55) for these
    # docs regardless of query, so a high threshold misses short personal queries
    # ("my name", "my projects"). We keep the cutoff low and let the LLM ignore
    # irrelevant injected context — verified that time/math queries still work
    # even when a chunk is injected. (Smaller chunk_chars improves separation if
    # you have a larger corpus.)
    score_threshold: float = 0.25        # ignore only the weakest matches (0..1)


@dataclass
class StateConfig:
    # Durable agent state in PostgreSQL: the ordered conversation log + a small
    # user-profile store, so Atlas keeps continuity across restarts. Distinct
    # from memory.py (semantic recall). Best-effort: runs without it if down.
    # DSN from .env, e.g. postgresql://postgres:atlas@localhost:5432/atlas
    enable_state: bool = True
    dsn: str = field(default_factory=lambda: os.environ.get("ATLAS_PG_DSN", ""))
    load_recent: int = 8   # messages reloaded into context on startup (continuity)


@dataclass
class CacheConfig:
    # Redis (Memurai on Windows) cache for repeated/expensive results:
    # web-search responses (short TTL) and text embeddings (long-lived).
    # Best-effort: runs uncached if Redis is down. URL from .env.
    enable_cache: bool = True
    url: str = field(
        default_factory=lambda: os.environ.get(
            "ATLAS_REDIS_URL", "redis://localhost:6379/0"
        )
    )
    web_ttl: int = 3600   # seconds to cache a web_search result
    embed_ttl: int = 0    # 0 = embeddings never expire (deterministic)


@dataclass
class VisionConfig:
    """Local screen vision via a small VLM (Qwen2.5-VL-3B) in llama.cpp.

    Lazy-loaded (only when you ask about the screen) and on CPU by default so it
    coexists with the GPU-resident Qwen3 on an 8 GB card.
    """
    enable_vision: bool = True
    model_path: str = "models/vision/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
    mmproj_path: str = "models/vision/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf"
    n_ctx: int = 4096
    n_gpu_layers: int = 0     # CPU (set -1 for GPU if you have VRAM headroom)
    max_image_px: int = 1024  # downscale the screenshot's longest side (CPU speed)
    # Grounding (locate an element to click) needs more detail than describing,
    # so the screen is downscaled less for locate() — higher = more accurate on
    # small UI targets, but slower. Set equal to max_image_px to disable.
    # NOTE: the CLIP image encoder runs on CUDA even when n_gpu_layers=0, and its
    # per-image buffers scale ~quadratically with this. On an 8 GB card already
    # holding the 8B LLM + STT, 1568 px overflowed VRAM and aborted the process
    # (CUDA error at encode). 1024 cuts encode memory to ~40%; drop to 768 if it
    # still OOMs, or set n_gpu_layers=-1 here AND free LLM VRAM to run it on GPU.
    ground_image_px: int = 1024
    max_tokens: int = 256


@dataclass
class FaceConfig:
    """Local face recognition via InsightFace (ArcFace embeddings, ONNX, CPU).

    Parallels the voice speaker gate: enroll known people by name, then identify
    who is in front of the webcam by cosine-matching 512-d embeddings. Like the
    voice gate, this is personalization, NOT security — a photo can fool it.
    """
    enable_faces: bool = True
    model_pack: str = "buffalo_l"   # InsightFace pack (auto-downloads on 1st use)
    db_path: str = "faces.npz"      # enrolled name -> embeddings store
    det_size: int = 640             # detector input size (px)
    # Cosine similarity (ArcFace normed embeddings) needed to call it a match.
    # Same person ~0.4-0.8, different <0.3; raise to be stricter.
    match_threshold: float = 0.40
    min_det_score: float = 0.40     # detector confidence floor (lower = lenient)


@dataclass
class AuthConfig:
    """Startup identity gate: register the owner's face + voice on first run,
    then require BOTH to match on every later startup before Atlas runs.

    NOTE: like the underlying voice/face checks, this is PERSONALIZATION, not
    security — a photo or a recording can pass it. If you ever get locked out
    (bad lighting, broken camera, lost voiceprint), set require_identity = False
    here to disable the gate.
    """
    require_identity: bool = True
    # Test mode: skip ALL startup auth (no face, voice, or password) and the
    # per-turn speaker gate, for frictionless development. Toggled in .env via
    # ATLAS_TEST_MODE; default disabled (the gate is on).
    test_mode: bool = field(default_factory=lambda: os.environ.get(
        "ATLAS_TEST_MODE", "").strip().lower() in ("1", "true", "yes", "on"))
    owner_name: str = "owner"        # name the owner's face is enrolled under
    face_shots: int = 5              # face samples captured during onboarding
    voice_samples: int = 5           # voice phrases recorded during onboarding
    voice_seconds: float = 4.0       # seconds recorded per onboarding phrase
    auth_attempts: int = 3           # verification tries before Atlas refuses to start
    # Typed password — a real "something you know" factor (not spoofable like
    # face/voice). Set on first run, then required (with face + voice) every
    # startup. Stored only as a salted PBKDF2 hash in password_path.
    require_password: bool = True
    password_path: str = "auth_secret.dat"
    # Registry of registered people and their authority (role). At registration
    # Atlas asks for a NAME and an AUTHORITY; the single 'admin' is unique.
    # owner_name above is only the fallback used before anyone is registered —
    # the real name/role come from this file (see auth.onboard / auth.register).
    users_path: str = "users.json"
    authorities: tuple = ("admin", "user", "guest")


@dataclass
class VaultConfig:
    """Encrypted store for website logins so Atlas can sign the user in on
    request (see vault.py). Passwords are NEVER stored in plaintext.

    Two composable at-rest protections:
      * Windows DPAPI (default): sealed to your Windows account, no prompt —
        Atlas logs in hands-free. Useless if the file is copied elsewhere.
      * Master password (optional): adds an AES layer under a key derived from a
        password you type once per session; keeps the vault encrypted even
        against other programs running as you. Required on non-Windows (no DPAPI).
    """

    enabled: bool = True
    vault_path: str = "vault.dat"
    use_dpapi: bool = True
    # Set True to also require a master password (prompted once per session).
    master_password: bool = False
    # Seconds to wait for a login page to load before autofilling by keyboard.
    # Raise it on a slow connection; the username field must be focused by then.
    autofill_delay: float = 4.0
    # Preferred login method: drive the browser DOM over the DevTools Protocol
    # (fast, reliable, no VRAM). Needs `pip install playwright` (lib only — it
    # attaches to your real Brave/Chrome). If playwright isn't installed, login
    # falls back to vision-guided control, then blind keyboard autofill.
    use_cdp_login: bool = True


@dataclass
class DspConfig:
    """Mic-side audio DSP: RNNoise noise suppression + acoustic echo
    cancellation (see audio_dsp.py). Both are best-effort — if a backend can't
    load, that stage passes audio through unchanged."""

    # Noise suppression (RNNoise) on the captured mic audio — helps the wake
    # word, VAD, and STT work in a noisy room.
    enable_noise_suppression: bool = True

    # Acoustic echo cancellation during playback, so Atlas's own voice coming
    # out of the speakers doesn't false-trigger the wake word (barge-in without
    # headphones). Frequency-domain (partitioned) NLMS adaptive filter.
    enable_echo_cancellation: bool = True
    aec_block: int = 160          # samples @16k per AEC block (10 ms)
    aec_filter_blocks: int = 16   # filter length = block*this (~160 ms of echo)
    aec_mu: float = 0.5           # NLMS adaptation step (0..1)
    # Extra bulk delay (ms) to align the playback reference with the mic; tune
    # up if echo isn't being removed (system output+input latency varies).
    aec_ref_delay_ms:  int = 208


@dataclass
class PlaybackConfig:
    # --- Playback / barge-in ---
    # MASTER ON/OFF for barge-in (interrupting Atlas mid-reply). Set False to
    # turn the whole feature off — Atlas will finish every reply and never listen
    # to the mic while speaking. When True, HOW it detects you is tuned by the
    # Config.bargein_* settings (bargein_mode, bargein_speech_rms, ...).
    allow_barge_in: bool = False
    # Trailing silence (ms) written after the last sentence, plus a matching
    # wait, so the end of a reply isn't clipped when the output stream stops.
    # Raise this if you still hear the last word cut off.
    tail_padding_ms: int = 450


@dataclass
class GuiConfig:
    """JARVIS-style holographic dashboard (see gui.py, launched via `python gui.py`).

    Only used by the optional PySide6 GUI; the terminal app ignores this. A
    central atom (glowing nucleus + electron orbits) sits inside concentric HUD
    tick-rings, flanked by chamfered data panels over a dark, bokeh-lit field.
    The orb's colour follows the assistant's state; its pulse follows the live
    mic/speech level streamed over ui_events.
    """
    fps: int = 60                     # animation refresh rate
    window_w: int = 1180              # matches the reference HUD's wide aspect
    window_h: int = 760
    orb_max_frac: float = 0.30        # atom radius as a fraction of the orb widget
    # Fixed orb colour (r, g, b) — the atom stays this colour in every state; the
    # state still drives motion (faster spin while thinking) and the text labels.
    # Set to None to instead colour the atom per state via state_colors.
    orb_color: tuple | None = None
    level_smoothing: float = 0.35     # 0..1 EMA weight for new audio levels
    level_decay: float = 0.90         # idle decay of the pulse toward 0 per frame
    transcript_max_lines: int = 200   # trim the transcript beyond this
    # Atom colour per state, as (r, g, b) — all within the reference's blue/cyan
    # family so the HUD reads as one holographic system.
    state_colors: dict = field(default_factory=lambda: {
        "idle": (44, 110, 165),        # deep calm blue
        "listening": (40, 150, 120),   # muted teal (hearing you)
        "thinking": (70, 120, 185),    # deep blue (working)
        "speaking": (52, 140, 190),    # dim cyan (replying)
    })


@dataclass
class AgentsConfig:
    """Iterative ReAct task agent on the one resident Qwen3: decide -> call a
    tool -> observe -> repeat until done. Pauses for spoken confirmation before
    risky/irreversible actions. See agents.py."""

    enable_agents: bool = True
    max_iterations: int = 8       # max tool steps before giving up on a task

    # Confirm before risky/irreversible actions (spoken "yes" required).
    confirm_risky: bool = True
    # Tools that always need confirmation. system_power and write_file are
    # handled specially (only shutdown/restart, and only overwrites, need it).
    # Spoken confirmation is reserved for the few genuinely destructive actions:
    # powering off the PC, overwriting a file, and deleting files. Those are
    # handled explicitly in Orchestrator._needs_confirm; this list is the
    # catch-all for any OTHER tool that should also ask first (empty by default
    # so normal commands run without a yes/no).
    risky_tools: list = field(default_factory=list)
    # Tools only the ADMIN may run (login authority). Non-admins (user/guest)
    # get a refusal; ignored entirely in test mode (everything unrestricted).
    # Names that don't exist are harmless — they simply never match.
    admin_only_tools: list = field(
        default_factory=lambda: [
            "run_command", "debug_python", "reset_all", "create_tool",
            "remove_tool", "create_github_repo", "close_app", "open_app",
            "write_file", "edit_file", "register_user", "shutdown",
            "restart_system", "control_app", "open_website", "close_website"])

    react_system_prompt: str = (
        "You are Atlas, a capable voice assistant that completes tasks by using "
        "tools, one step at a time. To use a tool, reply with ONLY a JSON object "
        'and nothing else: {"tool": "<name>", "arguments": {...}}. You then get an '
        "'Observation:' with the result, after which you call the next tool or "
        "finish. When the task is complete, reply in plain text (no JSON) with one "
        "short sentence for the user (it is spoken aloud). Use the fewest steps "
        "needed. ALWAYS perform actions by calling the matching tool — never claim "
        "you did something without calling its tool. You CANNOT change the system "
        "(volume, brightness, apps, mouse, keyboard, media, screenshots, power) by "
        "saying you did — you MUST emit the tool JSON, and may only confirm AFTER "
        "the Observation, reporting only what it says. Examples: 'turn up the "
        'volume\' -> {"tool":"set_volume","arguments":{"percent":70}}; \'open '
        'notepad\' -> {"tool":"open_app","arguments":{"name":"notepad"}}; \'take a '
        'screenshot\' -> {"tool":"take_screenshot","arguments":{}}. '
        "If a tool's Observation shows "
        "it failed or returned an error, tell the user plainly that it failed and "
        "why — NEVER claim it worked, was created, fixed, or is ready; do not retry "
        "the same failing tool more than once. If no tool is needed, just answer "
        "directly. ROUTING: building or creating ANY software — a website, web "
        "app, API, script, program, project, or code file — is a CODING task: "
        "use the code_agent tool (or write_file if code_agent is unavailable). "
        "The create_tool tool is ONLY for adding a new voice command/ability to "
        "yourself (e.g. 'add a tool that tells jokes'); NEVER use create_tool to "
        "build a website, app, script, program, or file."
    )


@dataclass
class CodeAgentConfig:
    """Delegate coding tasks to a CrewAI agent running in an isolated venv
    (.venv-crew) via a subprocess bridge (see code_agent.py / crew_runner.py).

    CrewAI is kept out of Atlas's own environment because its dependency tree
    would downgrade protobuf/pydantic and risk breaking onnxruntime (wake word,
    vision, faces) and qdrant (memory). It runs on a cloud LLM, so this is the
    one feature that leaves the machine. Best-effort: if the venv or an API key
    isn't set up, the code_agent tool simply isn't offered and Atlas falls back
    to its local coding tools.
    """
    enable_code_agent: bool = True
    crew_venv: str = ".venv-crew"          # isolated env holding crewai
    runner: str = "crew_runner.py"         # standalone script run inside it
    timeout: int = 180                     # seconds (cloud + crew orchestration)
    output_dir: str = "crew_output"        # plain (non-project) answers saved here
    # When a request is to build a project/app/website, the agent emits its files
    # and Atlas writes them to disk. Default base is the user's Desktop (or a
    # folder named in the request: desktop/documents/downloads).
    build_projects: bool = True

    # A fresh CrewAI agent defined here (role/goal/backstory) — no CrewAI login
    # or published repository needed; it runs on Gemini with just GEMINI_API_KEY.
    # Override any of these in .env.
    agent_role: str = field(default_factory=lambda: os.environ.get(
        "ATLAS_CREW_AGENT_ROLE",
        "Senior AI Software Engineer and Systems Architect"))
    agent_goal: str = field(default_factory=lambda: os.environ.get(
        "ATLAS_CREW_AGENT_GOAL",
        "Turn the user's request into secure, scalable, maintainable, "
        "well-tested code, following modern engineering best practices."))
    agent_backstory: str = field(default_factory=lambda: os.environ.get(
        "ATLAS_CREW_AGENT_BACKSTORY",
        "A seasoned engineer specializing in software architecture, backend and "
        "frontend development, APIs, databases, AI/LLM integrations, automation, "
        "DevOps, and cloud-native applications. You write production-grade "
        "solutions and explain key decisions concisely."))

    # Gemini model via litellm (needs GEMINI_API_KEY). Override with ATLAS_CREW_MODEL.
    model: str = field(default_factory=lambda: os.environ.get(
        "ATLAS_CREW_MODEL", "gemini/gemini-2.5-flash"))
