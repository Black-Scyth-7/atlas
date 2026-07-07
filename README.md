# Atlas — local, privacy-first voice assistant

Atlas listens for a wake word, verifies it's you speaking, transcribes your
command, reasons about it with a **local** LLM, optionally acts on your computer,
and speaks the reply back. The entire AI pipeline — wake word, speaker check,
speech-to-text, the language model, and text-to-speech — runs **offline on your
machine**. No cloud APIs, no server daemons, nothing leaves the device.

The only parts that ever touch the network are opt-in and obvious: the
`web_search` tool, `play_music` (YouTube), and `create_github_repo` (the `gh`
CLI). Everything else is local.

---

## Table of contents

- [What Atlas can do](#what-atlas-can-do)
- [How it works](#how-it-works)
- [Technology stack](#technology-stack)
- [Requirements](#requirements)
- [Setup](#setup)
- [GPU build (optional)](#gpu-build-optional-for-nvidia-gpus)
- [Optional services (memory, state, cache)](#optional-services)
- [Features in depth](#features-in-depth)
- [Configuration reference](#configuration-reference)
- [Privacy](#privacy)
- [Troubleshooting](#troubleshooting)
- [Project layout](#project-layout)

---

## What Atlas can do

| Area | Capabilities |
| --- | --- |
| **Conversation** | Natural spoken Q&A with short rolling memory, grounded in the current date. Answers in English. |
| **Multi-step tasks** | Acts as an iterative **ReAct agent** — chains several tool calls, observing each result, to complete a single spoken request (e.g. *"open Notepad and write a haiku"*). |
| **Web search** | Live web answers via Tavily (REST or MCP) or DuckDuckGo, with automatic fallback and caching. |
| **Web & info** | Read/summarize a URL, weather, news, stock & crypto prices, translate, safe calculator, currency & unit conversion (free, no API keys). |
| **System control** | Set system/per-app volume, screen brightness, take screenshots, lock/sleep/shutdown. |
| **Media** | Play a song from YouTube; play/pause/next/previous of whatever's already playing; say what's now playing (Windows SMTC). |
| **Apps & windows** | Open and close applications and websites — optionally in a chosen browser (*"open GitHub in Brave"*). |
| **Passwords & login** | Save website logins in an **encrypted vault** and sign you in on request (*"log into Gmail"*) — filling your real browser. Passwords are encrypted at rest (Windows DPAPI + optional master password), **never** stored in plaintext. |
| **Keyboard & mouse** | Type text, press hotkeys, click/move/scroll the mouse — full input control. |
| **Screen vision** | Look at the screen and answer questions about it (*"read this"*, *"what does this error say?"*) with a local vision model. |
| **Vision-guided control** | Find a UI element by description and click/type into it (*"click the Save button"*) — operating apps it can see, not just ones with named tools. Best on large, labeled targets. |
| **Camera** | Look through the webcam (*"look at me"*, *"what am I holding?"*) and take photos, interpreted by the same local vision model. |
| **Facial recognition** | Enroll people by name and identify who's at the camera (*"who is this?"*, *"remember my face as Alex"*) via local ArcFace embeddings. |
| **Identity gate** | First run registers a password + your face + voice; every later startup requires all three before Atlas will start. |
| **Coding** | Write, read, and edit source files, and run/compile/test code by voice — the agent can iterate on errors. |
| **Cloud coding agent** | Optionally delegate coding tasks to a locally-defined **CrewAI** agent running on **Gemini**, in an isolated env. |
| **Developer** | Create a GitHub repository by voice via the `gh` CLI. |
| **Self-extension** | Write *new* tools/functions into its own code on request (*"add a tool that…"*) — saved as plugins and available immediately + after restart. |
| **Memory / learning** | Remembers across sessions (semantic recall), and *learns* your facts/preferences/corrections on request (*"remember that…"*, *"forget that…"*) — always applied thereafter. |
| **Your documents** | Answers grounded in your own files (RAG over `docs/`). |
| **Documents & OCR** | Read/answer/summarize a PDF or text file on demand, and OCR (read all text from) the screen or an image. |
| **Meeting assistant** | Record a meeting, transcribe it, and write up a summary + action items. |
| **Hands-free** | Wake-word activation, barge-in (interrupt mid-reply), and a follow-up window so you needn't repeat the wake word. Or press **F1** to type a command instead. |
| **Clean audio** | RNNoise noise suppression so it hears you in a noisy room, and echo cancellation so it doesn't trigger on its own voice from the speakers. |
| **Safety** | Pauses and asks for a spoken *"yes"* before irreversible actions (close app, create repo, shutdown, factory reset). |
| **Factory reset** | *"Reset yourself"* wipes everything — all memory, conversation history, cache, and your enrolled voice + face (after confirmation). |

Everything is **best-effort and degrades gracefully**: if an optional service
(memory, state, cache, vision) can't start, Atlas prints why on startup and keeps
running without it.

---

## How it works

### Per-turn pipeline

1. **Wake word** — an always-on mic listens for *"Atlas"* (a custom
   locally-trained MatchboxNet model: `models/atlas.onnx`, run via onnxruntime).
2. **Record** — captures audio until you stop talking (webrtcvad voice-activity
   detection).
3. **Speaker gate** — verifies the clip sounds like the enrolled owner
   (SpeechBrain ECAPA-TDNN). Unrecognized speakers are ignored.
4. **Speech-to-text** — transcribes with faster-whisper (small model, CPU,
   English).
5. **LLM brain** — Qwen3-8B via llama-cpp-python, in-process and resident in
   VRAM. Runs as a **ReAct task agent**: it decides whether to answer directly or
   call a tool, observes the result, and loops until the task is done.
6. **Text-to-speech** — Piper synthesizes the reply, streamed sentence-by-sentence
   so the first sentence plays while the rest is still being generated.

After replying, Atlas opens a short **follow-up window** so you can continue
without saying the wake word again. Saying *"Atlas"* while it's talking
**interrupts** it (barge-in) and starts a new command.

```text
"Atlas"  →  record  →  is it you?  →  transcribe  →  ReAct agent  →  speak
 wake        VAD        ECAPA          Whisper      Qwen3 + tools     Piper
                                                        │
                                          decide → call a tool → observe
                                          → decide → … → final answer
```

### Design principles

- **Local-first.** Every model runs on-device; the network is touched only by
  explicitly network-bound tools.
- **One resident model, many jobs.** A single Qwen3-8B is the conversationalist,
  the planner, and the tool-using agent — no extra models loaded for routing.
- **Modular stages.** Each stage is an independent module you can run on its own:
  `python audio_input.py`, `speaker_id.py`, `stt.py`, `llm.py`, `tts.py`,
  `playback.py`, `vision.py`, `memory.py`, `state.py`, `cache.py`.
- **Best-effort everything.** Optional subsystems print `enabled` / `DISABLED —
  <reason>` at startup and never block the core loop.

> **The speaker gate is personalization, not security.** It only checks whether a
> clip *sounds like* you and is trivially spoofable by a recording. Don't rely on
> it for access control. Toggle it with `require_speaker_match` in `config.py`.

---

## Technology stack

Everything runs in a single Python 3.12 process. Models are local; no servers or
daemons are required for the core pipeline.

### Core voice pipeline

| Stage | Technology | Notes |
| --- | --- | --- |
| Wake word | **MatchboxNet** (PyTorch → ONNX runtime) | Custom locally-trained "Atlas" model; see `wake_pytorch/`. |
| Recording | **webrtcvad** + **sounddevice** (PortAudio) | Voice-activity detection trims silence. |
| Audio cleanup | **RNNoise** (noise suppression) + numpy frequency-domain **AEC** | Denoise the mic; cancel the speaker echo. |
| Speaker gate | **SpeechBrain** ECAPA-TDNN (PyTorch, CPU) | Cosine-similarity voiceprint match. |
| Speech-to-text | **faster-whisper** (CTranslate2) | `medium` model on GPU (CUDA, fp16); auto-falls back to CPU int8. Parked off-GPU while the vision model runs so both fit in VRAM. |
| LLM brain | **Qwen3-8B** (Q4_K_M GGUF) via **llama-cpp-python** | In-process, GPU or CPU, streaming. |
| Text-to-speech | **Piper** (ONNX) | Streamed sentence-by-sentence. |

### Intelligence & tools

| Capability | Technology |
| --- | --- |
| Task agent | Custom **ReAct loop** on the one resident Qwen3 (`agents.py`). |
| Tool calling | JSON protocol parsed in `tools.py` (reliable for local GGUFs). |
| Web search | **Tavily** (REST API + **MCP** SDK) and **DuckDuckGo**. |
| Screen vision | **Qwen2.5-VL-3B** (GGUF + mmproj) via llama.cpp, CPU. |
| Camera | **OpenCV** (`opencv-python`) webcam capture, fed to the same VLM. |
| Facial recognition | **InsightFace** (ArcFace/SCRFD ONNX, CPU) + cosine match. |
| Coding | File I/O + `subprocess` command execution (cross-platform). |

### System integration (Windows)

| Capability | Technology |
| --- | --- |
| Volume (system + per-app) | **pycaw** (Core Audio APIs). |
| Brightness | **screen-brightness-control**. |
| Screenshots / image handling | **Pillow**. |
| Media transport + "now playing" | **winsdk** (Windows SMTC), media-key fallback. |
| Keyboard & mouse | **ctypes** `SendInput` / `mouse_event` (Win32). |
| App launch/close, power | `subprocess`, `taskkill`, `shutdown` (Win32). |
| GitHub repos | **GitHub CLI** (`gh`). |

### Storage & memory

| Layer | Technology |
| --- | --- |
| Semantic memory | **Qdrant** (embedded) + **fastembed** (BGE-small-en-v1.5, 384-d). |
| Durable state | **PostgreSQL 17** via **psycopg**. |
| Cache | **Redis** / **Memurai** via **redis-py**. |
| Document RAG | **Qdrant** + **fastembed** + **pypdf**, mtime-manifest incremental ingest. |

### Platform & tooling

- **Python 3.12**, **PyTorch** (CPU for SpeechBrain), **NumPy**.
- **CUDA 12.8+/13.x** + **MSVC** + **Ninja** for the optional from-source GPU build
  of llama-cpp-python (tested on a Blackwell RTX 5050, `sm_120`).
- **python-dotenv** for `.env` secret loading.

---

## Requirements

- **Python 3.12** (3.13/3.14 lack wheels for torch / llama-cpp / onnxruntime).
- A microphone; **headphones recommended** (barge-in assumes no speaker echo).
- ~7 GB disk for models (plus ~3 GB if you add the vision model). GPU optional —
  CPU works out of the box.
- **Windows** for system control / media / vision (those use Win32, pycaw, SMTC).
  The core voice pipeline is cross-platform.
- **Linux only:** install PortAudio first — `sudo apt install portaudio19-dev`
  (needed by `sounddevice`). Windows/macOS need nothing extra.

---

## Setup

### 1. Virtual environment + dependencies

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs the **CPU** build of `llama-cpp-python` (which
ignores `n_gpu_layers`). For GPU offload, see [GPU build](#gpu-build-optional-for-nvidia-gpus).

### 2. Download the LLM (Qwen3-8B, Q4_K_M GGUF)

Get the GGUF from a bartowski repo on Hugging Face, e.g.
[`bartowski/Qwen_Qwen3-8B-GGUF`](https://huggingface.co/bartowski/Qwen_Qwen3-8B-GGUF),
and place the `Q4_K_M` file at **`models/Qwen3-8B-Q4_K_M.gguf`**:

```powershell
pip install huggingface_hub
huggingface-cli download bartowski/Qwen_Qwen3-8B-GGUF Qwen_Qwen3-8B-Q4_K_M.gguf --local-dir models
# then rename the downloaded file to models\Qwen3-8B-Q4_K_M.gguf (match config.py)
```

Needs ~6 GB VRAM (GPU) and stays resident for the session. Path set by
`LLMConfig.model_path` in `config.py`.

### 3. Download a Piper voice

```powershell
python -m piper.download_voices en_US-amy-medium --download-dir models/piper
```

This writes `en_US-amy-medium.onnx` + `.onnx.json` to `models/piper/` (paths set
by `TTSConfig` in `config.py`).

### 4. Run — first launch registers you

```powershell
python main.py
```

With the [identity gate](#startup-identity-gate-password--face--voice-login) on by default,
the **first launch walks you through registering your face, voice, and a typed
password** (look at the camera, repeat a few phrases, then set a password). Every
later launch then asks you to verify all three before Atlas starts. Say
**"Atlas"**, speak a command, and it replies aloud. Ctrl+C to quit.

*Optional holographic GUI* — a JARVIS-style window (central atom + live data
panels) that runs the exact same assistant on a worker thread:

```powershell
python main.py --gui     # same app, in the window (or: python gui.py)
python main.py --no-gui  # force the terminal app even if ATLAS_GUI is set
```

Or set `ATLAS_GUI=1` in your `.env` to make the window the default. Needs
`PySide6` (in `requirements.txt`). The window mirrors state (idle / listening /
thinking / speaking), pulses with the mic, and shows the live transcript; **F1**
still toggles the type-in box.

*Optional manual enrollment / tuning* (also usable any time to add samples):

```powershell
python enroll.py            # voice only -> voiceprint.npy
python enroll_face.py Alex  # face only  -> faces.npz
python speaker_id.py        # test voice accept/reject scoring
```

Tune `speaker_threshold` (voice) and `FaceConfig.match_threshold` (face) in
`config.py`, or set `AuthConfig.require_identity = False` to skip the gate.

> First runs download the ECAPA-TDNN and Whisper models (internet needed once);
> the wake-word model is a local file (`models/atlas.onnx`) and needs no download.
> Afterwards the core AI pipeline is fully offline.

The startup banner reports the status of every subsystem, e.g.:

```text
Cache: enabled (Redis).
Documents: 42 chunks indexed (RAG active).
Vision: enabled (lazy-loaded on first screen question).
Memory: enabled (128 stored).
State: enabled (310 messages logged).
Faces: enabled (1 enrolled: owner).
Agents: ReAct task loop (max 8 steps, confirm risky actions).
Noise suppression: enabled (RNNoise).
Echo cancellation: enabled (cancels Atlas's own voice on barge-in).
Identity check — verify your password, face, and voice to start Atlas.
Ready. Say the wake word ('atlas') or press F1 to type. Ctrl+C to quit.
```

---

## GPU build (optional, for NVIDIA GPUs)

The prebuilt `llama-cpp-python` wheels don't support recent GPUs (e.g. Blackwell
`sm_120`), so GPU offload needs a from-source build:

1. **CUDA Toolkit 12.8+** (13.x recommended): `winget install Nvidia.CUDA`.
2. **MSVC C++ toolchain** (Windows): a working VS Build Tools with the
   "Desktop development with C++" / VCTools workload —
   `winget install Microsoft.VisualStudio.2022.BuildTools --override "--passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"`.
3. **Build** (from a shell where `vcvarsall.bat x64` has been run):

   ```bat
   set CMAKE_GENERATOR=Ninja
   set CMAKE_ARGS=-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler
   pip install llama-cpp-python --no-binary llama-cpp-python --force-reinstall --no-cache-dir
   ```

   Set `CMAKE_CUDA_ARCHITECTURES` to your GPU (120 = Blackwell, 89 = Ada, 86 = Ampere).
4. **Make the CUDA runtime DLLs findable:**

   ```powershell
   python setup_gpu.py
   ```

   CUDA 13 ships its runtime DLLs in `<CUDA>\bin\x64`, and llama-cpp's loader
   won't pick them up via `PATH`; `setup_gpu.py` copies the needed ones next to
   `ggml-cuda.dll`. **Rerun it after any llama-cpp reinstall.**

Verify: `python -c "from llama_cpp import llama_cpp as C; print(C.llama_supports_gpu_offload())"`
should print `True`. On an RTX 5050 this takes Qwen3-8B Q4_K_M from ~5 to ~47 tok/s.

---

## Optional services

These three storage layers are all optional and best-effort — Atlas runs without
any of them, and the startup line shows each as `enabled` or `DISABLED — <reason>`.

### Durable agent state (PostgreSQL)

Keeps the ordered conversation log, sessions, and a user-profile table; reloads
the last few messages on startup for continuity across restarts.

```powershell
winget install --id PostgreSQL.PostgreSQL.17 -e --override "--mode unattended --unattendedmodeui none --superpassword atlas --serverport 5432 --disable-components stackbuilder"
& "C:\Program Files\PostgreSQL\17\bin\createdb.exe" -U postgres -h localhost atlas   # PGPASSWORD=atlas
```

Then put the DSN in `.env`:

```ini
ATLAS_PG_DSN=postgresql://postgres:atlas@localhost:5432/atlas
```

Tune via `StateConfig` (`enable_state`, `load_recent`). Test with `python state.py`.

### Cache (Redis / Memurai)

Caches web-search responses (short TTL) and text embeddings (long-lived) to cut
latency and external calls. On Windows use **Memurai**, which speaks the Redis
protocol:

```powershell
# winget's MSI run can fail with 1603 (SFXCA temp error); install the MSI directly instead:
$msi = "$env:TEMP\Memurai-Developer.msi"
Invoke-WebRequest https://dist.memurai.com/releases/Memurai-Developer/4.1.2/Memurai-Developer-v4.1.2.msi -OutFile $msi
Start-Process msiexec.exe -ArgumentList "/i `"$msi`" /qn /norestart" -Verb RunAs -Wait
```

It runs a service on `localhost:6379` (auto-start). Override with `ATLAS_REDIS_URL`
in `.env`; tune TTLs via `CacheConfig`. Test with `python cache.py`.

### Semantic memory (Qdrant, embedded)

No install needed — it runs *embedded* (in-process, `qdrant_data/` on disk).
See [Persistent memory](#persistent-memory) below.

---

## Features in depth

### Task agent (iterative ReAct loop)

For real multi-step tasks the resident Qwen3 runs as a **ReAct agent**
(`agents.py`, config in `AgentsConfig`): each turn it either calls one tool or
gives a final answer, and after every tool it sees an `Observation:` with the
result, so it can chain and adapt across steps:

```text
decide → call a tool → observe result → decide → … → speak the final answer
```

This lets one spoken request span several actions — e.g. *"open Notepad and write
hello world"* (open_app → type_text) or *"create a repo called X, then open it on
GitHub"*. It has the full tool set available at every step, so a task can cross
domains (system + web + developer), and loops up to `max_iterations` (default 8).

**Confirmation before risky actions.** When the agent wants to do something
irreversible — close an app, create a GitHub repo, shut down/restart — it pauses
and asks out loud (*"I'm about to … Should I go ahead?"*). Say **"yes"** to carry
on (the task resumes exactly where it paused) or **"no"** to cancel. Read-only and
benign actions run without asking. Configure via `AgentsConfig.confirm_risky` (set
`False` for fully autonomous) and `risky_tools` (which tools require a yes). Set
`enable_agents=False` to restore plain single-prompt behavior.

Each executed step prints `· tool(args) -> result` to the console so you can watch
a task unfold.

### Tools

`tools.py` exposes capabilities to the model via a JSON protocol it's prompted to
follow (reliable for local GGUFs). Base tools: `get_time`, `get_date`,
`set_timer`, and `web_search`. The system-control, media, input, vision, and
GitHub tools below are added when `enable_system_control` is on.

### Web search

`web_search` has interchangeable backends, chosen by `ToolsConfig.web_search_backend`
(your query goes to the internet either way):

- **`tavily_rest`** (recommended) — Tavily REST API. One round-trip, returns a
  synthesized answer ready to speak.
- **`tavily_mcp`** — Tavily via its MCP server (uses the `mcp` SDK). Returns raw
  snippets the LLM then summarizes; more protocol overhead.
- **`duckduckgo`** — free, no key. Best for entity lookups, sparse for
  question-form queries.
- **`auto`** (default) — Tavily REST if a key is available, else DuckDuckGo.

Set credentials in a local **`.env`** in the project root (get a key at
<https://tavily.com>). Atlas loads it at startup, so it works in any shell with no
`setx` and no terminal restart. The MCP URL embeds the key, so it also feeds the
REST backend — you only need one of these lines:

```ini
ATLAS_TAVILY_MCP_URL=https://mcp.tavily.com/mcp/?tavilyApiKey=tvly-...
# or:
TAVILY_API_KEY=tvly-...
```

`.env` is gitignored — keep your key out of version control. (Real environment
variables still work; `.env` values take precedence over stale ones.) When a
backend returns raw snippets, the LLM condenses them into one spoken sentence;
if a backend fails, Atlas automatically falls back to the next available one.

### Website login & credential vault

Atlas can store your website logins and sign you in on request (*"log into
Gmail"*, *"log into Facebook on Brave"*). Passwords are **never** kept in
plaintext — the vault (`vault.py`) encrypts each secret at rest with **Windows
DPAPI** (tied to your account, no prompt), plus an **optional master password**
(AES/Fernet, prompted once per session) for a second layer.

**Save a login** in a terminal (secrets never go through voice/STT):

```bash
python vault.py --set --site gmail --username you@gmail.com   # prompts for the password (hidden)
python vault.py --list          # show saved sites
python vault.py --delete gmail  # remove one
```

**Then, by voice:** *"log into gmail"*. Atlas opens the site's login page and
fills your saved credentials, preferring — in order:

1. **CDP / Playwright** — drives the real browser's DOM by selector (fast,
   reliable). Enable with `pip install playwright` (library only; it attaches to
   your existing Brave/Chrome — no browser download).
2. **Vision-guided** — locates the fields on screen with the vision model and
   clicks/types (handles Google's account chooser + two-step flow).
3. **Keyboard autofill** — types into the focused login form.

Add *"in &lt;browser&gt;"* (Chrome, Brave, Firefox, Edge) to open/login in a
specific browser. Config lives in `VaultConfig`; the vault file (`vault.dat`) is
gitignored and wiped by a factory reset. 2FA / CAPTCHA still need you to finish
them. Set `enabled = False` to disable the feature.

### System control

Atlas can act on the machine via voice (`system_control.py`, exposed as tools).
**Windows-only.** Turn the whole feature off with `enable_system_control = False`
in `ToolsConfig`.

- **`set_volume`** — set system volume to a percentage, or mute/unmute (pycaw).
- **`set_app_volume`** — set volume/mute for a specific running app (e.g. Spotify).
- **`set_brightness`** — set display brightness (screen-brightness-control).
- **`take_screenshot`** — capture the screen to `screenshots/` (Pillow).
- **`system_power`** — `lock` / `sleep` / `cancel`, plus `shutdown` / `restart`.

> **Power guardrails.** `shutdown`/`restart` are disabled by default
> (`allow_power_off` in `ToolsConfig`) so a misheard command can't power down the
> machine. When enabled, they run with a 15 s delay you can abort with "cancel
> shutdown" — and the ReAct agent additionally asks for confirmation first.

### Media

- **`play_music`** — start a specific song/video: searches YouTube, opens the top
  result's watch page in the browser, and nudges it to play (SMTC).
- **`media_control`** — play / pause / next / previous / stop of *already-playing*
  media (the real session via Windows SMTC, falling back to media keys).
- **`now_playing`** — says the current track ("X by Y") via SMTC.

### Apps, windows, and the web

- **`open_app`** — launch an app (notepad, calculator, chrome, …; names with shell
  metacharacters are refused).
- **`close_app`** — close a running app by name (taskkill by image; refuses
  critical processes like `explorer.exe`).
- **`open_website`** — open a URL in the browser.

### Keyboard & mouse

- **`type_text`** — type text into the focused window (Win32 SendInput, full
  Unicode).
- **`press_keys`** — press a key or hotkey (`enter`, `ctrl+s`, `alt+tab`, `win+d`).
- **`mouse_click` / `move_mouse` / `mouse_scroll`** — click left/right/middle
  (optional x,y and double-click), move the cursor, and scroll.

### Screen vision (local VLM)

Atlas can look at your screen and answer questions about it (`vision.py`,
`see_screen` tool). It uses **Qwen2.5-VL-3B** (GGUF + mmproj) via llama.cpp,
**lazy-loaded on the first screen question** and run on **CPU** so it coexists
with the GPU-resident Qwen3 on an 8 GB card. The screenshot is downscaled before
inference for speed; expect a few seconds per look on CPU.

Setup — download the model into `models/vision/`:

```powershell
$b="https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main"
iwr "$b/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf" -OutFile models/vision/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf
iwr "$b/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf" -OutFile models/vision/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf
```

Startup prints `Vision: enabled …` (or `DISABLED — vision model not downloaded`).
Tune via `VisionConfig` (`n_gpu_layers` — set `-1` for GPU if you have VRAM
headroom; `max_image_px`). Ask: *"what's on my screen?"*, *"read this"*, *"what
does this error say?"*. Test standalone with `python vision.py`.

#### Vision-guided control

Qwen2.5-VL is a *grounding* model — it can return the bounding box of an element
you describe — so Atlas can act on things it sees, even apps without a dedicated
tool. `vision.locate()` asks for the box, converts its center to a fraction of
the screen, and the tools map that to a real pixel and click/type there:

- **`find_on_screen`** — locate an element and report where it is, **without
  clicking** (safe; good for *"where is the X?"* and for checking before acting).
- **`click_element`** — find an element by description and click it
  (*"click the Save button"*, *"click the search box"*; left/right, double).
- **`type_into`** — click a field by description, then type into it (optionally
  pressing Enter): *"type 'hello' into the search box"*.

Grounding runs at a higher resolution than describing for accuracy
(`VisionConfig.ground_image_px`, default 1568).

> **Accuracy caveat.** This is a small (3B) local model. Coordinate mapping is
> exact, but the model's *localization* is reliable mainly for **large, clearly
> visible, labeled** elements (buttons with text, fields, links). It struggles
> with tiny/dense targets like small taskbar icons, and can mis-read small text.
> For tricky cases, ask Atlas to `find_on_screen` first to confirm the spot, or
> raise `ground_image_px`. For much stronger grounding, swap in a larger VLM
> (e.g. Qwen2.5-VL-7B) if you have the VRAM/CPU. You can also add `click_element`
> / `type_into` to `AgentsConfig.risky_tools` to require a spoken "yes" before
> each click.

#### Camera

Atlas can also look through the **webcam** — same local VLM, just a camera frame
instead of a screenshot (`vision.capture_camera()` via OpenCV):

- **`see_camera`** — grab a frame and answer a question about it: *"look at me,
  how do I look?"*, *"what am I holding up?"*, *"is anyone behind me?"*. Optional
  `camera` index for a second/external camera.
- **`take_photo`** — capture a still and save it to `photos/`.

The camera is opened only for the moment a frame is grabbed (it warms up for
~0.6 s first so auto-exposure settles — otherwise the frame is dark) and then
released — Atlas does not hold the camera open. Needs `opencv-python` (in
`requirements.txt`); without it these tools return a clear "not installed"
message and the rest of Atlas is unaffected.

#### Facial recognition

Atlas can recognize people at the webcam — the visual parallel to the voice
speaker gate (`face_id.py`). It uses **InsightFace** (ArcFace, ONNX, CPU):
detect a face, turn it into a 512-d embedding, and cosine-match it against
enrolled people (exactly like `speaker_id.py` does for voiceprints). Models
(~300 MB) download automatically on first use.

Enroll people (one time, parallel to `enroll.py`):

```powershell
python enroll_face.py Alex          # capture a few shots from the webcam
python enroll_face.py Alex 6        # add more samples (vary your angle)
python face_id.py                   # identify whoever is at the camera now
```

Then, by voice:

- **`recognize_face`** — *"who is this?"*, *"do you recognize me?"* → answers with
  the enrolled name, *"someone I don't recognize"*, or *"I don't see anyone"*.
  Handles multiple people in frame.
- **`enroll_face`** — *"remember my face as Alex"*, *"this is Sam"* → saves the
  current face under that name for next time.
- **`open_face_window`** — *"open the face recognition window"* → opens a live
  webcam window (`face_window.py`) with boxes + names drawn on faces (green +
  name/score for enrolled people, red "unknown" otherwise); recognition runs on
  a background thread so the video stays smooth. Press **q** to close. Also
  runnable standalone: `python face_window.py`.

Startup prints `Faces: enabled (N enrolled: …)` or
`Faces: DISABLED — <reason>`. Tune via `FaceConfig`: `match_threshold` (cosine
similarity to count as a match — raise to be stricter), `min_det_score`,
`db_path`. Enrolled embeddings live in `faces.npz` (gitignored).

> **Personalization, not security.** Like the voice gate, face recognition can be
> fooled by a photo of someone. Don't use it to protect anything sensitive.

### Startup identity gate (password + face + voice login)

Atlas can require **you** — a typed password **and** your face **and** your voice
— before it will start (`auth.py`, config in `AuthConfig`, on by default):

- **First ever run** → onboarding. Atlas speaks you through registering your
  **face** (a few webcam shots → `faces.npz`), your **voice** (repeat a few short
  phrases → `voiceprint.npy`), and a **password** (typed twice → stored only as a
  salted PBKDF2-SHA256 hash in `auth_secret.dat`, never in plaintext).
- **Every later run** → login. Atlas asks you to **type your password** (hidden
  input), then look at the camera and say something, and starts **only if all
  three match**. The password is checked first (cheapest, and a true secret); you
  get `auth_attempts` tries (default 3) per factor, otherwise it refuses to start.

```text
First-time setup: registering your face, voice, and password...
Identity check — verify your password, face, and voice to start Atlas.
  Password: ********
  [auth] voice score 0.71 (ok).
Ready. Say the wake word ('atlas').
```

The **password** is the one factor that's genuinely a secret — unlike face/voice,
it can't be spoofed by a photo or a recording. Tune via `AuthConfig`:
`require_identity` (master switch), `require_password`, `owner_name`, `face_shots`,
`voice_samples`, `voice_seconds`, `auth_attempts`. The face match uses
`FaceConfig.match_threshold`; the voice match uses `Config.speaker_threshold`. If
face recognition or the camera is unavailable, the gate **degrades** to the
remaining factors rather than locking you out. After a
[factory reset](#factory-reset) (which clears all three), the next startup
automatically re-runs onboarding.

> **Mixed strength — read this.** The **password** is real authentication (a
> salted hash of a secret you know). The **face and voice** checks are
> personalization and are spoofable (a photo / a recording can pass them), so
> they add convenience, not strong security. **If you get locked out** (bad
> lighting, broken camera, lost voiceprint, forgotten password), set
> `AuthConfig.require_identity = False` in `config.py`, or delete
> `auth_secret.dat` to reset just the password (you'll set a new one on the next
> run).

**Test mode.** For frictionless development you can skip the whole gate (and the
per-turn speaker check) without enrolling anything. It's toggled in `.env` and is
**disabled by default**:

```ini
ATLAS_TEST_MODE=true     # skip all auth — NOT secure, dev only
```

When on, startup prints a `** TEST MODE … **` warning and goes straight to the
wake word. Set it back to `false` (or remove the line) to restore the gate.

### Coding agent (write, edit, and run code)

> These local coding tools are the **fallback**: if the [CrewAI cloud coding
> agent](#cloud-coding-agent-crewai-optional) is set up, it takes over all coding
> and these aren't registered. They're used only when CrewAI is absent.

Atlas can actually build code on disk, not just dictate it. Combined with the
ReAct loop, it writes a file, runs it, reads the error, fixes it, and runs
again — a full edit/run/debug cycle by voice. Tools (in `tools.py`, gated by
`ToolsConfig.enable_coding`):

- **`write_file`** — create or overwrite a file with given content (parent
  folders are created).
- **`read_file`** — read a file's contents back into context.
- **`edit_file`** — exact find→replace inside an existing file (optionally all
  occurrences).
- **`list_dir`** — list a directory.
- **`run_command`** — run a shell command (run/compile/test), returning its
  output and exit status. `command_timeout` (default 60 s) caps how long it runs.
- **`check_syntax`** — *(debugging)* compile-check a Python file **without running
  it** (safe) and report the first syntax error's line, or that it's clean.
- **`debug_python`** — *(debugging)* run a Python script and, on a crash, return a
  **parsed traceback**: the error type/message, the exact `file:line`, the
  offending source line, and any stdout printed before the crash — so Atlas can
  pinpoint the bug instead of guessing from raw stderr.

With these, the agent does a real debug loop: write → `check_syntax` →
`debug_python` → read the parsed error → `edit_file` the fix → run again, until
it's clean. For example, given a script that divides by zero, `debug_python`
returns:

```text
[crashed] ZeroDivisionError: division by zero
  at C:\tmp\crash.py:2 in divide
    2 | return a / b
```

File paths are unrestricted — Atlas reads and writes wherever you tell it
(`~`, relative paths, and **Desktop / Documents / Downloads** references are
resolved to your real folders, including OneDrive-redirected ones, so *"on my
desktop"* lands in the right place instead of a guessed username path). Because that's powerful, the actions
that execute code or destroy data are gated by a spoken **"yes"**:
**`run_command`** and **`debug_python`** (both run code on your machine) and
**`write_file` when it would overwrite an existing file** (creating new files,
and the read-only `check_syntax` / `read_file` / `list_dir`, run freely). Turn
the whole feature off with `enable_coding = False`.

Example — *"write a Python script in C:\\tmp\\fib.py that prints the first ten
Fibonacci numbers, then run it"*: the agent calls `write_file`, then asks to
confirm `run_command`, and on "yes" runs it and reads you the output.

### Self-extension (Atlas writes its own tools)

Atlas can permanently add **new tools/functions to itself** when you ask
(*"add a tool that tells me a random joke"*, *"create a function that converts
Celsius to Fahrenheit"*). Rather than editing the core `tools.py` (risky), each
new capability is saved as a **plugin file** in `plugins/`:

- **`create_tool`** — the agent writes a `name`, `description`, an `arguments`
  schema, and the Python `code` (`def run(args): … return "<spoken result>"`).
  Atlas validates it (it must compile and define `run`), saves
  `plugins/<name>.py`, and **registers it live** — so it's usable in the same
  conversation *and* auto-loaded on every future startup.
- **`remove_tool`** — deletes a custom tool you previously added (built-ins are
  protected and can't be removed or overwritten).

**Dependencies install automatically.** The tool code is written by the CrewAI
agent (with the local model as fallback), and may use third-party packages — it
declares them on a `# pip: <packages>` first line. `create_tool` **pip-installs
those into Atlas's venv** before validating, so the new tool works immediately
(e.g. *"Added a new tool 'fetch_weather' (installed: requests)."*). If a plugin's
dependency is ever missing at startup, Atlas reinstalls it from that `# pip:`
line. Turn this off with `ToolsConfig.auto_install_tool_deps = False`.

On startup Atlas prints `Custom tools: loaded N from plugins/.` if any exist.
Configure via `ToolsConfig` (`enable_self_extend`, `plugins_dir`,
`auto_install_tool_deps`).

> **This is self-modifying code that runs in-process.** A new tool's code
> executes with Atlas's full permissions, so `create_tool` and `remove_tool` are
> gated by the agent's spoken confirmation (*"I'm about to write a new tool
> called '…' into my own code — should I go ahead?"*). Built-in tools can't be
> overwritten, names are sanitised, and code that doesn't compile or define
> `run` is rejected without being saved — but a tool you ask it to write can
> still do whatever its code says. Review `plugins/` if in doubt; delete a file
> to remove a tool. Turn the feature off with `enable_self_extend = False`.

### Cloud coding agent (CrewAI, optional)

For heavier coding, Atlas delegates the task to a **CrewAI agent** running on
**Gemini** (`code_agent.py` + `crew_runner.py`, config in `CodeAgentConfig`). The
agent is **defined locally** — a role/goal/backstory senior-engineer (no CrewAI
login or published repository needed). When you give a coding command, the
`code_agent` tool runs it. For a **project/app/website** it builds the actual
files on disk (in a named subfolder of your **Desktop** by default, or
Documents/Downloads if you say so) and reports where; for a one-off snippet it
saves the answer to `crew_output/`. Either way it speaks a short confirmation.

**Why a separate environment.** CrewAI's dependencies would downgrade
`protobuf` and `pydantic` in Atlas's venv, which can break onnxruntime (wake
word, vision, faces) and qdrant (memory). So CrewAI is installed in an **isolated
`.venv-crew`** and Atlas talks to it over a **subprocess bridge** (JSON via a
pipe) — your core stack is never touched.

Setup (one time):

```powershell
python setup_crew.py                 # creates .venv-crew, installs crewai[google-genai]
```

Then in `.env` set your Gemini key (from Google AI Studio):

```ini
GEMINI_API_KEY=...
# ATLAS_CREW_MODEL=gemini/gemini-2.5-flash      # optional model override
# ATLAS_CREW_AGENT_ROLE / _GOAL / _BACKSTORY    # optional persona overrides
```

> **Model note:** the default is `gemini/gemini-2.5-flash`. Some keys have no
> free-tier quota for `gemini-2.0-flash` (a `429 … limit: 0`); switch models via
> `ATLAS_CREW_MODEL` if you hit a quota/availability error.

Startup prints `Coding agent: CrewAI (…)` or `DISABLED — <reason>`. It's
**best-effort and off until set up**. **When the CrewAI agent is active it owns
all coding** — the local coding tools (`write_file`, `run_command`,
`debug_python`, …) are **not** registered, so every coding request goes to
CrewAI. They come back automatically as the **fallback** only when CrewAI is
absent (no `.venv-crew`/key), so Atlas can still code locally.

> **Privacy + cost.** This is the one feature that leaves the machine — coding
> prompts and code go to the cloud LLM, which needs an API key/billing, and each
> task takes several seconds. The slug must exist in your CrewAI org/repo.

### Developer: create a GitHub repository

- **`create_github_repo`** — create a GitHub repo via the `gh` CLI (needs
  `gh auth login` once; supports public/private and a description). The agent
  confirms before creating, since it publishes to your account. It creates an
  empty remote repo (it does not push local code).

### Persistent memory

Atlas remembers across sessions via **semantic memory** (`memory.py`):

- **Qdrant**, run *embedded* (`qdrant_data/` on disk, no server daemon), stores
  each exchange as a vector.
- **fastembed** (BAAI/bge-small-en-v1.5, 384-d, CPU) produces the vectors.
- Each turn, the query is embedded, the most relevant past memories are injected
  into the LLM context, and the new exchange is stored afterward.

**This is how Atlas "learns."** Rather than retraining the model (it runs as a
quantized GGUF, which is inference-only), Atlas adapts to you through memory — no
GPU cost, no risk of breaking the model, effective immediately:

- **`remember`** — when you say *"remember that…"*, *"from now on…"*, *"call me…"*,
  or correct it (*"no, it's actually…"*), Atlas saves that as an explicit **note**.
- Notes are **always injected** into every turn's context as *standing
  instructions to honor* (not just recalled when a topic happens to match), so
  preferences and corrections **reliably stick**.
- **`forget`** — *"forget that I like X"* removes the closest matching note (with
  a high similarity threshold so it won't delete the wrong one).

Toggle/tune via `MemoryConfig` (`enable_memory`, `recall_k`, `score_threshold`).
Test standalone with `python memory.py`. Delete `qdrant_data/` to wipe memories.

> Note: actual weight **fine-tuning** (LoRA on the full-precision model, then
> re-quantizing to GGUF) is a separate offline, GPU-heavy batch job — not live
> self-training. Memory-based learning above is the safe, instant path and is
> what's built in.

> **Run one instance at a time.** The embedded store is single-process; a second
> `main.py` (or a leftover one still running) can't open it and will start with
> memory disabled. The startup line will say so.

### RAG over your documents

Atlas can answer questions grounded in **your own files** (`rag.py`). Drop `.txt`
/ `.md` / `.pdf` files in **`docs/`** and run `main.py` — it **auto-indexes
new/changed files at startup** (`RAGConfig.auto_ingest`), skipping unchanged ones
via an mtime manifest. On every turn the most relevant chunks (above a similarity
threshold) are auto-injected into the LLM's context, so answers cite your notes
without you invoking anything.

```powershell
python ingest.py ~/Documents     # manual/bulk indexing of a folder elsewhere
```

Run `ingest.py` only while `main.py` is **not** up (the store is single-process).
Tune via `RAGConfig` (`chunk_chars`, `top_k`, `score_threshold`, `docs_dir`).

> Design note: RAG is **auto-retrieval + injection** (like memory recall), not a
> tool the model must choose to call — a small model can't reliably tell which
> questions concern your private docs, so always surfacing relevant passages is
> far more reliable.

### The three storage layers

- **`memory.py` (Qdrant)** — *semantic recall*: find relevant past exchanges by meaning.
- **`state.py` (Postgres)** — *durable transcript + profile*, and verbatim
  recent-turn continuity on restart.
- **`cache.py` (Redis/Memurai)** — *transient cache* of web-search results and embeddings.

### Factory reset

Saying *"reset yourself"* / *"reset everything"* / *"wipe everything"* triggers
the **`reset_all`** tool, which erases all of Atlas's personal state in one go:

- **semantic memory** (Qdrant collection),
- **conversation history + profile** (Postgres),
- **cache** (all `atlas:*` keys in Redis),
- your **voice enrollment** (`voiceprint.npy`),
- your **face enrollment** (`faces.npz`), and
- your **startup password** (`auth_secret.dat`).

It's irreversible, so it's gated by the agent's spoken confirmation (*"I'm about
to factory-reset everything … Should I go ahead?"* → say **"yes"**). Afterwards
the next startup re-runs onboarding (face, voice, and password). Each store is
cleared independently, so a reset still succeeds if, say, Postgres is down (it
just skips that one).

### Web & info

Focused tools that return clean spoken answers, using **free endpoints (no API
keys)** or the local model (`tools.py`):

- **`read_url`** — fetch a web page and summarize it (or answer a question about
  it). *"Read me this link", "summarize that article."*
- **`get_weather`** — current conditions for a city (or your location) via
  `wttr.in`.
- **`get_news`** — top headlines (or about a topic) from Google News RSS.
- **`get_stock`** — latest price + daily change for a ticker (Yahoo Finance).
- **`get_crypto`** — price + 24h change in USD (CoinGecko); accepts names or
  symbols (`btc`, `eth`, …).
- **`translate`** — translate text into any language (local model, offline).
- **`calculate`** — safe math (arithmetic + functions like `sqrt`, `sin`, and
  `pi`/`e`) evaluated with an AST walker — **no `eval`/code execution**.
- **`convert_currency`** — latest-rate currency conversion (Frankfurter/ECB).
- **`convert_units`** — length, mass, volume, and temperature conversions.

### Productivity: documents, OCR & meetings

Local productivity tools (`tools.py`, with the meeting recorder in `meeting.py`):

- **`read_pdf`** — read a PDF / txt / md file (via `pypdf`) and return its text,
  or answer a question about it. Paths resolve `~`/Desktop/Documents like the
  coding tools. *"Read me that PDF on my desktop."*
- **`summarize_document`** — read a pdf/txt/md and produce a short summary + key
  points (map-reduce over long files, on the local model). *"Summarize report.pdf."*
- **`read_text` (OCR)** — read **all** the text in the **screen** or an **image
  file**, verbatim, using the local vision model (no Tesseract install needed).
  *"Read the text on screen", "extract the text from this image."*
- **`start_meeting` / `stop_meeting`** — `start_meeting` records the meeting from
  the microphone in the background (you can keep talking to Atlas); `stop_meeting`
  transcribes it with Whisper, writes a **summary + key points + action items**,
  saves the notes and full transcript to `meetings/`, and reads you a one-line
  recap.

> **Meeting audio caveat.** It records from the **microphone**, so it captures
> your voice, anyone on open speakers, and in-person rooms — but **not** system
> audio when you're on **headphones** (remote calls). That needs WASAPI loopback,
> which this `sounddevice` build doesn't expose; a `soundcard`-based loopback
> capture can be added if you need it. Recording is held in memory, so keep
> sessions reasonable (tens of minutes).

### Hands-free follow-up

After Atlas replies it keeps listening for **`followup_window_ms`** (default 6 s)
so you can ask a follow-up **without repeating the wake word**. Stay silent and it
returns to wake-word mode. Toggle with `enable_followup`. (Leading silence is
trimmed from each clip so the speaker gate isn't thrown off by the pause before
you speak.)

### Noise suppression & echo cancellation

Two mic-side DSP stages clean the audio before the wake word / VAD / STT see it
(`audio_dsp.py`, config in `DspConfig`). Both are best-effort — if a backend
can't load, that stage passes audio through and Atlas keeps working.

- **Noise suppression (RNNoise).** The recorded **command** audio is denoised by
  **RNNoise** (a small recurrent denoiser, run at its native 48 kHz and
  resampled around our 16 kHz pipeline), so voice-activity detection and
  transcription hold up in a **noisy room**. The **wake word runs on raw audio**
  — the MatchboxNet model is trained on raw mic input (with MUSAN noise
  augmentation), and denoising shifts the signal enough to hurt detection, so it's
  applied downstream only. RNNoise ships as a tiny bundled DLL inside `pyrnnoise`; Atlas
  loads that low-level binding directly (sidestepping the package's heavy PyAV
  dependency).
- **Acoustic echo cancellation (AEC).** So Atlas doesn't **trigger on its own
  voice** coming out of the speakers, a frequency-domain (partitioned) NLMS
  adaptive filter subtracts the speaker output from the mic during playback. The
  TTS being played is captured as a reference (`ReferenceBuffer`) and the echo
  canceller removes it from the mic before the barge-in wake-word check — so
  barge-in works on **open speakers**, not just headphones.

Startup prints `Noise suppression: enabled (RNNoise).` and `Echo cancellation:
enabled …`. Tune via `DspConfig`: `enable_noise_suppression`,
`enable_echo_cancellation`, `aec_filter_blocks` (echo tail length), `aec_mu`
(adaptation speed), and **`aec_ref_delay_ms`** — raise this if echo isn't being
removed, since the right value depends on your machine's audio output+input
latency. AEC is a from-scratch numpy filter (no Windows AEC library builds), so
treat it as effective-but-tunable; it reaches ~12–16 dB echo reduction on
synthetic tests. Run **`python calibrate_aec.py`** to measure your machine's
round-trip latency and pick the best `aec_ref_delay_ms` automatically.

**Barge-in modes** (`Config.bargein_mode`): the default **`speech`** interrupts
the moment you start talking (near-end energy after AEC — cheap, so it reacts in
real time, no wake word needed); **`wakeword`** requires you to say "Atlas" but
its per-chunk model can lag on a busy CPU. On/off is `PlaybackConfig.allow_barge_in`.

### Text input (press F1)

Don't want to talk? **Press F1 to toggle into text mode.** In text mode Atlas
**stops listening** (the mic isn't read at all) and you type commands at the
`text>` prompt — each runs through the exact same pipeline (LLM, tools, memory)
as a spoken one, and the reply is still spoken and printed. You stay in text mode
for as many commands as you like; **press F1 again to return to voice**. A
watcher polls the key globally (Windows `GetAsyncKeyState`, no extra dependency),
so the toggle works whether or not the terminal is focused. Typed turns skip the
speaker gate (you're already at the keyboard). Toggle the whole feature with
`enable_text_input` / change the key with `text_input_vk` in `config.py`.

### Barge-in

While Atlas speaks, saying the wake word interrupts it and starts a new command.
With echo cancellation on (above) this works on open speakers; with it off it
assumes headphones, since Atlas's own voice can otherwise false-trigger it.
Disable barge-in entirely with `allow_barge_in = False`.

### Wake word

`Config.wake_model` points at **`models/atlas.onnx`** — a custom single-word
"Atlas" detector trained **from scratch in PyTorch** with the
[**MatchboxNet**](https://arxiv.org/abs/2004.08531) architecture (1D
time-channel separable conv ResNet, ~89k params). The full training pipeline
lives in **`wake_pytorch/`** and has no openWakeWord dependency:

- **Features** — 64 MFCC @ 16 kHz (25 ms / 10 ms), defined once in
  `wake_pytorch/features.py` and imported by *both* training and the runtime
  detector so the front-end can never drift.
- **Positives** — synthesized with **[Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M)**
  across ~28 English voices × prosody variants × speeds (`gen_positives.py`).
- **Negatives + augmentation** — the **[MUSAN](https://www.openslr.org/17/)**
  corpus (music / speech / noise): speech gives hard babble negatives, music and
  noise are mixed into positives at 0–20 dB SNR, plus synthetic quiet/ambient
  clips (`prepare_negatives.py`). SpecAugment + time-shift applied at train time.
- **Runtime** — `wake_pytorch/detector.py` (`WakeDetector`) runs the exported
  ONNX via onnxruntime over a rolling 1.5 s mic window, returning `P("Atlas")`.

Retrain end-to-end (in the isolated GPU venv `wake_pytorch/.venv`):

```powershell
wake_pytorch\.venv\Scripts\python wake_pytorch\gen_positives.py       # Kokoro positives
wake_pytorch\.venv\Scripts\python wake_pytorch\prepare_negatives.py   # MUSAN + quiet negs
wake_pytorch\.venv\Scripts\python wake_pytorch\train.py               # -> checkpoints/
wake_pytorch\.venv\Scripts\python wake_pytorch\eval.py                # threshold sweep
wake_pytorch\.venv\Scripts\python wake_pytorch\export_onnx.py         # -> models/atlas.onnx
```

Single short words false-trigger more easily, so tune `wake_threshold` to your
mic — `eval.py` sweeps it and reports recall vs false-wakes/hour.

### Language (English-only)

Atlas runs in English: STT is pinned to English (`stt_language = "en"`), the LLM
is told to always respond in English, and TTS uses the English Piper voice
(`en_US-amy-medium`). To re-enable multilingual support: set `stt_language = ""`
(Whisper auto-detects per utterance), add per-language Piper voices to
`TTSConfig.voices` (a `lang → (model, config)` map sharing one `sample_rate`),
and tell the LLM to reply in the user's language. The TTS code already selects a
voice by language and handles non-Latin sentence terminators.

---

## Configuration reference

Everything tunable lives in `config.py` (dataclasses):

| Dataclass | What it controls |
| --- | --- |
| `Config` | Sample rate, VAD/silence timing, wake model + `wake_threshold` + `wake_min_rms` (room-noise gate) + `wake_consecutive`, barge-in (`bargein_mode` speech/wakeword, `bargein_speech_rms`), `min_command_ms` (minimum listen after speech starts), speaker threshold and `require_speaker_match`, STT model/device/language, follow-up window, `enable_text_input` / `text_input_vk` (F1 typed input), `startup_phrase` (spoken greeting). |
| `LLMConfig` | Model path, `n_gpu_layers`, context size, system prompt, `disable_thinking` (Qwen3 `/no_think`), `enable_tools`, `max_tool_rounds`. |
| `TTSConfig` | Piper voice paths, output sample rate, `default_lang`, `voices` map. |
| `PlaybackConfig` | `allow_barge_in`. |
| `DspConfig` | `enable_noise_suppression` (RNNoise), `enable_echo_cancellation`, `aec_filter_blocks`, `aec_mu`, `aec_ref_delay_ms`. |
| `ToolsConfig` | `web_search_backend`, Tavily credentials, `enable_system_control`, `allow_power_off`, `enable_coding`, `command_timeout`, `enable_self_extend`, `plugins_dir`. |
| `VaultConfig` | Encrypted website-login vault (`vault.py`): `enabled`, `vault_path`, `use_dpapi`, `master_password`, `use_cdp_login` (Playwright DOM fill), `autofill_delay`. |
| `CodeAgentConfig` | `enable_code_agent`, `crew_venv`, `runner`, `timeout`, `output_dir`, `agent_role`/`agent_goal`/`agent_backstory`, `model` (`ATLAS_CREW_MODEL`, Gemini). |
| `AgentsConfig` | `enable_agents`, `max_iterations`, `confirm_risky`, `risky_tools`, the ReAct system prompt. |
| `MemoryConfig` | `enable_memory`, `recall_k`, `score_threshold`, store path. |
| `StateConfig` | `enable_state`, `load_recent`, Postgres DSN (from `ATLAS_PG_DSN`). |
| `CacheConfig` | Redis URL (from `ATLAS_REDIS_URL`), TTLs. |
| `RAGConfig` | `docs_dir`, `auto_ingest`, `chunk_chars`, `top_k`, `score_threshold`. |
| `VisionConfig` | `enable_vision`, model + mmproj paths, `n_gpu_layers`, `max_image_px`, `ground_image_px`. |
| `FaceConfig` | `enable_faces`, `model_pack`, `db_path`, `match_threshold`, `min_det_score`, `det_size`. |
| `AuthConfig` | `require_identity` (password+face+voice startup gate), `test_mode` (`ATLAS_TEST_MODE` in `.env` — skip all auth), `require_password`, `password_path`, `owner_name`, `face_shots`, `voice_samples`, `voice_seconds`, `auth_attempts`. |

Secrets (`ATLAS_TAVILY_MCP_URL` / `TAVILY_API_KEY`, `ATLAS_PG_DSN`,
`ATLAS_REDIS_URL`) go in a gitignored **`.env`** in the project root, along with
`ATLAS_TEST_MODE` (the dev bypass for the identity gate, default off).

---

## Privacy

- The wake word, speaker check, speech-to-text, language model, and
  text-to-speech all run **locally**. Audio is processed in memory and not sent
  anywhere.
- The network is used only by clearly network-bound, opt-in tools: `web_search`
  (your query → search provider), `play_music` (YouTube), and
  `create_github_repo` (GitHub via `gh`).
- The screen vision model, the webcam, and face recognition all run **locally** —
  screenshots, camera frames, and face embeddings are processed on-device and
  never uploaded. The camera is opened only to grab a frame, then released.
- Face recognition stores **biometric data** (face embeddings) in `faces.npz`,
  kept local and gitignored. It is personalization, not security (a photo can
  fool it).
- The startup password is stored only as a salted **PBKDF2-SHA256 hash**
  (`auth_secret.dat`) — never in plaintext — and is gitignored.
- Stored data stays on your machine: `qdrant_data/` (memory), the local Postgres
  DB (transcript), `docs/` and `qdrant_docs/` (your documents + index),
  `voiceprint.npy`, `faces.npz`, `auth_secret.dat`, `screenshots/`, and
  `photos/`. Delete any of them to wipe that data.
- One-time, on first run only: model downloads (ECAPA-TDNN, Whisper, and any you
  fetch manually). The wake-word model is a local file — no download.

---

## Troubleshooting

- **"No voiceprint found"** — run `python enroll.py`.
- **"GGUF not found"** — download the model to `models/Qwen3-8B-Q4_K_M.gguf`.
- **"Could not open the microphone"** — check your input device / OS permissions.
- **Wake word "wakes on every word" from idle** — usually your room's background
  noise is clearing the energy gate, so the decay-tail after any word triggers it.
  Raise **`wake_min_rms`** above your room's noise floor (measure it with
  `wake_pytorch/live_mic_test.py`); raise `wake_threshold` / `wake_consecutive` too
  if needed.
- **Wake word only responds to *your* voice** — the model was trained too heavily
  on your own recordings. Retrain speaker-independent: keep the diverse TTS
  positives and use a low `--real-frac` (≈0.1), not 0.5.
- **Wake word not triggering** — lower `wake_threshold`, or `wake_min_rms` if a
  soft/distant "Atlas" is missed.
- **Cuts you off / won't stop recording** — tune `silence_tail_ms`,
  `min_command_ms` (minimum listen time), and `vad_aggressiveness`.
- **GPU build loads as CPU** — rerun `python setup_gpu.py`; confirm
  `llama_supports_gpu_offload()` is `True`.
- **Memory/State/Cache disabled** — read the `DISABLED — <reason>` line; for
  memory, make sure no other `main.py` is still running.
- **Vision disabled** — download the two files into `models/vision/` (see
  [Screen vision](#screen-vision-local-vlm)).

---

## Project layout

| File | Role |
| --- | --- |
| `main.py` | Wires everything together; the per-turn loop. |
| `config.py` | All configuration dataclasses. |
| `audio_input.py` | Wake word + VAD recording. |
| `audio_dsp.py` | RNNoise noise suppression + echo cancellation (mic DSP). |
| `speaker_id.py` | ECAPA-TDNN speaker verification. |
| `enroll.py` | One-time voiceprint enrollment. |
| `stt.py` | faster-whisper transcription. |
| `llm.py` | Qwen3 brain, tool loop, context building. |
| `agents.py` | ReAct task agent + risk confirmation. |
| `tools.py` | Tool registry + JSON protocol + self-extension (plugins). |
| `vault.py` | Encrypted website-login vault (DPAPI/AES) + `--set`/`--list` CLI. |
| `calibrate_aec.py` | Measures audio round-trip latency to tune `aec_ref_delay_ms`. |
| `code_agent.py` / `crew_runner.py` / `setup_crew.py` | CrewAI coding agent: subprocess bridge + isolated-venv runner + setup. |
| `plugins/` | Custom tools Atlas wrote for itself (auto-loaded at startup). |
| `system_control.py` | OS actions (volume, media, apps, input, power). |
| `vision.py` | Local screen vision (Qwen2.5-VL) + webcam capture. |
| `meeting.py` | Background meeting recorder (mic) for transcription. |
| `face_id.py` / `enroll_face.py` | Face recognition (InsightFace) + enrollment. |
| `face_window.py` | Live face-recognition webcam window (boxes + names). |
| `auth.py` | Startup identity gate (password + face + voice onboarding/login). |
| `tts.py` | Piper text-to-speech. |
| `playback.py` | Streaming audio playback. |
| `memory.py` | Semantic memory (Qdrant + fastembed). |
| `state.py` | Durable state (PostgreSQL). |
| `cache.py` | Cache (Redis/Memurai). |
| `rag.py` / `ingest.py` | Document RAG + manual ingestion. |
| `setup_gpu.py` | Copies CUDA DLLs for the GPU build. |
| `wake_pytorch/` | Local "Atlas" wake-word training pipeline (MatchboxNet, Kokoro positives, MUSAN negatives) + runtime detector. |
