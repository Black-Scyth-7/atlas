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
    wake_chunk: int = 1280          # openWakeWord expects 80 ms chunks (1280 samples)

    # --- Wake word (openWakeWord) ---
    # Custom "Atlas" model trained locally (see wake_training/). To revert to a
    # bundled phrase, set this to "hey_jarvis". Single-word wake words are the
    # hardest case, so tune wake_threshold against your mic/environment.
    wake_model: str = "models/atlas.onnx"
    wake_framework: str = "onnx"    # "onnx" (reliable on Windows) or "tflite"
    wake_threshold: float = 0.5     # raise to reduce false triggers

    # --- Recording / voice activity detection (webrtcvad) ---
    silence_tail_ms: int = 800      # stop after this much trailing silence
    max_command_ms: int = 12000     # hard cap on a single command
    vad_aggressiveness: int = 2     # 0 (lenient) .. 3 (aggressive)
    preroll_frames: int = 5         # ~150 ms of lead-in kept before speech starts

    # --- Hands-free follow-up ---
    # After Atlas replies, keep listening this long for a follow-up command
    # without needing the wake word again. If you stay silent, it returns to
    # waiting for the wake word.
    enable_followup: bool = True
    followup_window_ms: int = 6000  # how long to wait for a follow-up to begin

    # --- Speaker verification (SpeechBrain ECAPA-TDNN, CPU) ---
    # NOTE: this gate is PERSONALIZATION, not security. It only checks whether a
    # clip sounds like the enrolled owner; it is trivially spoofable by a
    # recording of your voice. Do not rely on it to keep anyone out.
    speaker_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    speaker_device: str = "cpu"     # spec: run speaker ID on CPU
    voiceprint_path: str = "voiceprint.npy"   # written by enroll.py
    speaker_threshold: float = 0.30  # cosine similarity cutoff; tune per-mic
    require_speaker_match: bool = True  # set False to disable the gate entirely

    # --- Speech-to-text (faster-whisper, CPU) ---
    stt_model: str = "small"        # tiny / base / small / medium / large-v3
    stt_device: str = "cpu"         # spec: run STT on CPU
    stt_compute_type: str = "int8"  # "int8" on CPU, "float16" on GPU
    stt_language: str = "en"        # English only ("" would auto-detect any language)

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
    n_ctx: int = 4096
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
    allow_power_off: bool = False

    # Coding tools: read/write/edit files and run commands (write_file, read_file,
    # edit_file, list_dir, run_command). run_command and overwriting an existing
    # file are gated by spoken confirmation in the agent. command_timeout caps how
    # long a run_command may take (seconds).
    enable_coding: bool = True
    command_timeout: int = 60

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
    ground_image_px: int = 1568
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
    owner_name: str = "owner"        # name the owner's face is enrolled under
    face_shots: int = 5              # face samples captured during onboarding
    voice_samples: int = 5           # voice phrases recorded during onboarding
    voice_seconds: float = 4.0       # seconds recorded per onboarding phrase
    auth_attempts: int = 3           # verification tries before Atlas refuses to start


@dataclass
class PlaybackConfig:
    # --- Playback / barge-in ---
    # When True, Atlas listens for the wake word while speaking and stops
    # talking if it hears you (barge-in). See playback.py / main.py for the hook.
    allow_barge_in: bool = True


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
    risky_tools: list = field(
        default_factory=lambda: ["close_app", "create_github_repo",
                                 "run_command", "debug_python", "reset_all"])

    react_system_prompt: str = (
        "You are Atlas, a capable voice assistant that completes tasks by using "
        "tools, one step at a time. To use a tool, reply with ONLY a JSON object "
        'and nothing else: {"tool": "<name>", "arguments": {...}}. You then get an '
        "'Observation:' with the result, after which you call the next tool or "
        "finish. When the task is complete, reply in plain text (no JSON) with one "
        "short sentence for the user (it is spoken aloud). Use the fewest steps "
        "needed. ALWAYS perform actions by calling the matching tool — never claim "
        "you did something without calling its tool. If a tool fails, adapt or "
        "briefly say why. If no tool is needed, just answer directly."
    )
