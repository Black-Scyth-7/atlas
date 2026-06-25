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
| **System control** | Set system/per-app volume, screen brightness, take screenshots, lock/sleep/shutdown. |
| **Media** | Play a song from YouTube; play/pause/next/previous of whatever's already playing; say what's now playing (Windows SMTC). |
| **Apps & windows** | Open and close applications and websites. |
| **Keyboard & mouse** | Type text, press hotkeys, click/move/scroll the mouse — full input control. |
| **Screen vision** | Look at the screen and answer questions about it (*"read this"*, *"what does this error say?"*) with a local vision model. |
| **Vision-guided control** | Find a UI element by description and click/type into it (*"click the Save button"*) — operating apps it can see, not just ones with named tools. Best on large, labeled targets. |
| **Camera** | Look through the webcam (*"look at me"*, *"what am I holding?"*) and take photos, interpreted by the same local vision model. |
| **Facial recognition** | Enroll people by name and identify who's at the camera (*"who is this?"*, *"remember my face as Alex"*) via local ArcFace embeddings. |
| **Identity gate** | First run registers your face + voice; every later startup requires both to match before Atlas will start. |
| **Coding** | Write, read, and edit source files, and run/compile/test code by voice — the agent can iterate on errors. |
| **Developer** | Create a GitHub repository by voice via the `gh` CLI. |
| **Memory** | Remembers across sessions (semantic recall) and keeps a durable transcript + profile. |
| **Your documents** | Answers grounded in your own files (RAG over `docs/`). |
| **Hands-free** | Wake-word activation, barge-in (interrupt mid-reply), and a follow-up window so you needn't repeat the wake word. |
| **Safety** | Pauses and asks for a spoken *"yes"* before irreversible actions (close app, create repo, shutdown, factory reset). |
| **Factory reset** | *"Reset yourself"* wipes everything — all memory, conversation history, cache, and your enrolled voice + face (after confirmation). |

Everything is **best-effort and degrades gracefully**: if an optional service
(memory, state, cache, vision) can't start, Atlas prints why on startup and keeps
running without it.

---

## How it works

### Per-turn pipeline

1. **Wake word** — an always-on mic listens for *"Atlas"* (openWakeWord, a custom
   locally-trained model: `models/atlas.onnx`).
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
| Wake word | **openWakeWord** (ONNX runtime) | Custom locally-trained "Atlas" model. |
| Recording | **webrtcvad** + **sounddevice** (PortAudio) | Voice-activity detection trims silence. |
| Speaker gate | **SpeechBrain** ECAPA-TDNN (PyTorch, CPU) | Cosine-similarity voiceprint match. |
| Speech-to-text | **faster-whisper** (CTranslate2) | `small` model, CPU, int8. |
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

With the [identity gate](#startup-identity-gate-face--voice-login) on by default,
the **first launch walks you through registering your face and voice** (look at
the camera, then repeat a few phrases). Every later launch then asks you to
verify both before Atlas starts. Say **"Atlas"**, speak a command, and it replies
aloud. Ctrl+C to quit.

*Optional manual enrollment / tuning* (also usable any time to add samples):

```powershell
python enroll.py            # voice only -> voiceprint.npy
python enroll_face.py Alex  # face only  -> faces.npz
python speaker_id.py        # test voice accept/reject scoring
```

Tune `speaker_threshold` (voice) and `FaceConfig.match_threshold` (face) in
`config.py`, or set `AuthConfig.require_identity = False` to skip the gate.

> First runs download the openWakeWord, ECAPA-TDNN, and Whisper models (internet
> needed once); afterwards the core AI pipeline is fully offline.

The startup banner reports the status of every subsystem, e.g.:

```text
Cache: enabled (Redis).
Documents: 42 chunks indexed (RAG active).
Vision: enabled (lazy-loaded on first screen question).
Memory: enabled (128 stored).
State: enabled (310 messages logged).
Faces: enabled (1 enrolled: owner).
Agents: ReAct task loop (max 8 steps, confirm risky actions).
Identity check — verify your face and voice to start Atlas.
Ready. Say the wake word ('atlas'). Ctrl+C to quit.
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

Startup prints `Faces: enabled (N enrolled: …)` or
`Faces: DISABLED — <reason>`. Tune via `FaceConfig`: `match_threshold` (cosine
similarity to count as a match — raise to be stricter), `min_det_score`,
`db_path`. Enrolled embeddings live in `faces.npz` (gitignored).

> **Personalization, not security.** Like the voice gate, face recognition can be
> fooled by a photo of someone. Don't use it to protect anything sensitive.

### Startup identity gate (face + voice login)

Atlas can require **you** — your face *and* your voice — before it will start
(`auth.py`, config in `AuthConfig`, on by default):

- **First ever run** → onboarding. Atlas speaks you through registering your
  **face** (a few webcam shots) and your **voice** (repeat a few short phrases),
  saving `faces.npz` and `voiceprint.npy`.
- **Every later run** → login. Atlas asks you to look at the camera and say
  something, and starts **only if both the face and the voice match** the owner.
  You get `auth_attempts` tries (default 3); otherwise it refuses to start.

```text
First-time setup: registering your face and voice...
Identity check — verify your face and voice to start Atlas.
  [auth] voice score 0.71 (ok).
Ready. Say the wake word ('atlas').
```

Tune via `AuthConfig`: `require_identity` (master switch), `owner_name`,
`face_shots`, `voice_samples`, `voice_seconds`, `auth_attempts`. The face match
uses `FaceConfig.match_threshold`; the voice match uses `Config.speaker_threshold`.
If face recognition or the camera is unavailable, the gate **degrades to
voice-only** rather than locking you out. After a [factory reset](#factory-reset)
(which clears both enrollments), the next startup automatically re-runs
onboarding.

> **Personalization, not security — important here.** This gate is convenience
> and personalization, **not** real authentication: a photo of you can pass the
> face check and a recording of your voice can pass the voice check. Do not treat
> it as protecting access to the machine. **If you ever get locked out** (bad
> lighting, broken camera, lost voiceprint), set `AuthConfig.require_identity =
> False` in `config.py` to disable it.

### Coding agent (write, edit, and run code)

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
(`~` and relative paths are resolved). Because that's powerful, the actions
that execute code or destroy data are gated by a spoken **"yes"**:
**`run_command`** and **`debug_python`** (both run code on your machine) and
**`write_file` when it would overwrite an existing file** (creating new files,
and the read-only `check_syntax` / `read_file` / `list_dir`, run freely). Turn
the whole feature off with `enable_coding = False`.

Example — *"write a Python script in C:\\tmp\\fib.py that prints the first ten
Fibonacci numbers, then run it"*: the agent calls `write_file`, then asks to
confirm `run_command`, and on "yes" runs it and reads you the output.

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

Toggle/tune via `MemoryConfig` (`enable_memory`, `recall_k`, `score_threshold`).
Test standalone with `python memory.py`. Delete `qdrant_data/` to wipe memories.

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
- your **voice enrollment** (`voiceprint.npy`), and
- your **face enrollment** (`faces.npz`).

It's irreversible, so it's gated by the agent's spoken confirmation (*"I'm about
to factory-reset everything … Should I go ahead?"* → say **"yes"**). Afterwards,
re-enroll with `python enroll.py` (voice) and `python enroll_face.py <name>`
(face) before those features work again. Each store is cleared independently, so
a reset still succeeds if, say, Postgres is down (it just skips that one).

### Hands-free follow-up

After Atlas replies it keeps listening for **`followup_window_ms`** (default 6 s)
so you can ask a follow-up **without repeating the wake word**. Stay silent and it
returns to wake-word mode. Toggle with `enable_followup`. (Leading silence is
trimmed from each clip so the speaker gate isn't thrown off by the pause before
you speak.)

### Barge-in

While Atlas speaks, saying the wake word interrupts it and starts a new command.
Assumes headphones — on open speakers Atlas's own voice can false-trigger it;
disable with `allow_barge_in = False`.

### Wake word

`Config.wake_model` points at **`models/atlas.onnx`** — a custom single-word
"Atlas" model trained locally with openWakeWord (pipeline in
`wake_training/train_atlas.py`: ~20k LibriTTS-generated positives + the
precomputed negative feature set, file-free augmentation, CPU). Set it back to a
bundled phrase like `"hey_jarvis"` (and `wake_framework`) to change it — nothing
else changes. Single short words false-trigger more easily, so tune
`wake_threshold` to your mic (synthetic validation: silence ~0.09, a real "atlas"
clip ~0.998).

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
| `Config` | Sample rate, VAD/silence timing, wake model + threshold, speaker threshold and `require_speaker_match`, STT model/device/language, follow-up window. |
| `LLMConfig` | Model path, `n_gpu_layers`, context size, system prompt, `disable_thinking` (Qwen3 `/no_think`), `enable_tools`, `max_tool_rounds`. |
| `TTSConfig` | Piper voice paths, output sample rate, `default_lang`, `voices` map. |
| `PlaybackConfig` | `allow_barge_in`. |
| `ToolsConfig` | `web_search_backend`, Tavily credentials, `enable_system_control`, `allow_power_off`, `enable_coding`, `command_timeout`. |
| `AgentsConfig` | `enable_agents`, `max_iterations`, `confirm_risky`, `risky_tools`, the ReAct system prompt. |
| `MemoryConfig` | `enable_memory`, `recall_k`, `score_threshold`, store path. |
| `StateConfig` | `enable_state`, `load_recent`, Postgres DSN (from `ATLAS_PG_DSN`). |
| `CacheConfig` | Redis URL (from `ATLAS_REDIS_URL`), TTLs. |
| `RAGConfig` | `docs_dir`, `auto_ingest`, `chunk_chars`, `top_k`, `score_threshold`. |
| `VisionConfig` | `enable_vision`, model + mmproj paths, `n_gpu_layers`, `max_image_px`, `ground_image_px`. |
| `FaceConfig` | `enable_faces`, `model_pack`, `db_path`, `match_threshold`, `min_det_score`, `det_size`. |
| `AuthConfig` | `require_identity` (face+voice startup gate), `owner_name`, `face_shots`, `voice_samples`, `voice_seconds`, `auth_attempts`. |

Secrets (`ATLAS_TAVILY_MCP_URL` / `TAVILY_API_KEY`, `ATLAS_PG_DSN`,
`ATLAS_REDIS_URL`) go in a gitignored **`.env`** in the project root.

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
- Stored data stays on your machine: `qdrant_data/` (memory), the local Postgres
  DB (transcript), `docs/` and `qdrant_docs/` (your documents + index),
  `voiceprint.npy`, `faces.npz`, `screenshots/`, and `photos/`. Delete any of
  them to wipe that data.
- One-time, on first run only: model downloads (wake word, ECAPA-TDNN, Whisper,
  and any you fetch manually).

---

## Troubleshooting

- **"No voiceprint found"** — run `python enroll.py`.
- **"GGUF not found"** — download the model to `models/Qwen3-8B-Q4_K_M.gguf`.
- **"Could not open the microphone"** — check your input device / OS permissions.
- **Wake word too sensitive / not triggering** — adjust `wake_threshold`.
- **Cuts you off / won't stop recording** — tune `silence_tail_ms` /
  `vad_aggressiveness`.
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
| `speaker_id.py` | ECAPA-TDNN speaker verification. |
| `enroll.py` | One-time voiceprint enrollment. |
| `stt.py` | faster-whisper transcription. |
| `llm.py` | Qwen3 brain, tool loop, context building. |
| `agents.py` | ReAct task agent + risk confirmation. |
| `tools.py` | Tool registry + JSON protocol. |
| `system_control.py` | OS actions (volume, media, apps, input, power). |
| `vision.py` | Local screen vision (Qwen2.5-VL) + webcam capture. |
| `face_id.py` / `enroll_face.py` | Face recognition (InsightFace) + enrollment. |
| `auth.py` | Startup identity gate (face + voice onboarding/login). |
| `tts.py` | Piper text-to-speech. |
| `playback.py` | Streaming audio playback. |
| `memory.py` | Semantic memory (Qdrant + fastembed). |
| `state.py` | Durable state (PostgreSQL). |
| `cache.py` | Cache (Redis/Memurai). |
| `rag.py` / `ingest.py` | Document RAG + manual ingestion. |
| `setup_gpu.py` | Copies CUDA DLLs for the GPU build. |
| `wake_training/` | Local "Atlas" wake-word training pipeline. |
