"""Stage 7: tool calling for Atlas — timers, time/date, and web search.

Protocol (works regardless of whether the GGUF chat template natively parses
tool calls): the system prompt tells the model that, to use a tool, it must
reply with ONLY a JSON object:

    {"tool": "<name>", "arguments": { ... }}

We detect and run it, feed the result back as a message, and the model then
answers the user in plain spoken language. If the model replies with normal
text instead, that's treated as the final answer (no tool needed).

This is the JSON-instruction fallback the spec calls for. To switch to Qwen3's
native tool tokens later, pass `tools=` to create_chat_completion in llm.py and
read response `tool_calls`; the registry/execution here stays the same.
"""

import datetime
import json
import os
import re
import sys
import threading
from typing import Callable, Optional
from urllib.parse import parse_qs, quote_plus, urlparse
from urllib.request import Request, urlopen


def _key_from_mcp_url(url: str) -> str:
    """Pull the tavilyApiKey query param out of a Tavily MCP URL, if present."""
    if not url:
        return ""
    return parse_qs(urlparse(url).query).get("tavilyApiKey", [""])[0]


def _extract_json_objects(text: str) -> list[str]:
    """Return all balanced {...} substrings found in the text."""
    objects: list[str] = []
    start = 0
    while True:
        start = text.find("{", start)
        if start == -1:
            break
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    objects.append(text[start : i + 1])
                    start = i + 1
                    break
        else:
            start += 1
    return objects


def _html_to_text(html: str) -> str:
    """Crude HTML -> readable text (drops scripts/styles/tags, unescapes)."""
    import html as _h

    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html,
                  flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", _h.unescape(html)).strip()


# Common crypto symbol -> CoinGecko id.
_COIN_IDS = {
    "btc": "bitcoin", "eth": "ethereum", "sol": "solana", "ada": "cardano",
    "xrp": "ripple", "doge": "dogecoin", "bnb": "binancecoin", "ltc": "litecoin",
    "dot": "polkadot", "matic": "matic-network", "avax": "avalanche-2",
    "link": "chainlink", "trx": "tron", "shib": "shiba-inu", "usdt": "tether",
    "usdc": "usd-coin", "bch": "bitcoin-cash", "xmr": "monero", "atom": "cosmos",
}

# Unit -> (dimension, factor to the base unit). Temperature handled separately.
_UNITS = {
    "m": ("len", 1.0), "meter": ("len", 1.0), "meters": ("len", 1.0),
    "km": ("len", 1000.0), "cm": ("len", 0.01), "mm": ("len", 0.001),
    "mi": ("len", 1609.344), "mile": ("len", 1609.344), "miles": ("len", 1609.344),
    "yd": ("len", 0.9144), "yard": ("len", 0.9144), "ft": ("len", 0.3048),
    "feet": ("len", 0.3048), "foot": ("len", 0.3048), "in": ("len", 0.0254),
    "inch": ("len", 0.0254), "inches": ("len", 0.0254),
    "kg": ("mass", 1000.0), "g": ("mass", 1.0), "gram": ("mass", 1.0),
    "grams": ("mass", 1.0), "mg": ("mass", 0.001), "lb": ("mass", 453.592),
    "lbs": ("mass", 453.592), "pound": ("mass", 453.592), "pounds": ("mass", 453.592),
    "oz": ("mass", 28.3495), "ounce": ("mass", 28.3495),
    "l": ("vol", 1.0), "liter": ("vol", 1.0), "litre": ("vol", 1.0),
    "ml": ("vol", 0.001), "gal": ("vol", 3.78541), "gallon": ("vol", 3.78541),
    "gallons": ("vol", 3.78541),
}


def _safe_eval(node):
    """Evaluate an arithmetic AST node safely (no eval/exec)."""
    import ast
    import math
    import operator as op

    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
           ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod,
           ast.FloorDiv: op.floordiv, ast.USub: op.neg, ast.UAdd: op.pos}
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in ops:
        return ops[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
        return ops[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = getattr(math, node.func.id, None)
        if callable(fn):
            return fn(*[_safe_eval(a) for a in node.args])
    if isinstance(node, ast.Name):
        import math as _m
        return {"pi": _m.pi, "e": _m.e, "tau": _m.tau}.get(node.id)
    raise ValueError("unsupported expression")


def _known_folders() -> dict:
    """Map common folder names to the user's REAL paths (handles OneDrive
    redirection). Used so the model never has to guess a username/Desktop path."""
    home = os.path.expanduser("~")
    out = {"home": home}
    for name in ("Desktop", "Documents", "Downloads"):
        plain = os.path.join(home, name)
        onedrive = os.path.join(home, "OneDrive", name)
        out[name.lower()] = (onedrive if (not os.path.isdir(plain)
                                          and os.path.isdir(onedrive)) else plain)
    return out


class Tools:
    """Registry + executor for the assistant's tools."""

    def __init__(
        self,
        on_timer_fire: Optional[Callable[[str], None]] = None,
        web_search_backend: str = "auto",
        tavily_api_key: str = "",
        tavily_mcp_url: str = "",
        cache=None,
        enable_system_control: bool = False,
        allow_power_off: bool = False,
        vision=None,
        faces=None,
        code_agent=None,
        transcriber=None,
        meeting_recorder=None,
        memory=None,
        store=None,
        voiceprint_path: str = "",
        password_path: str = "",
        enable_coding: bool = True,
        command_timeout: int = 60,
        enable_self_extend: bool = True,
        plugins_dir: str = "plugins",
        auto_install_deps: bool = True,
        code_generator=None,
        authority: str = "admin",
        test_mode: bool = False,
        admin_only_tools=None,
        register_user_fn=None,
    ):
        # Called when a timer elapses; main.py wires this to speak the alert.
        self._on_timer_fire = on_timer_fire or (lambda msg: print(f"\n{msg}"))
        self._timers: list[threading.Timer] = []
        self.cache = cache  # optional web-search cache (see cache.py)
        self._allow_power_off = allow_power_off
        self._vision = vision  # optional screen vision (see vision.py)
        self._faces = faces    # optional face recognition (see face_id.py)
        self._code_agent = code_agent  # optional CrewAI coding agent (code_agent.py)
        self._transcriber = transcriber       # faster-whisper, for meetings
        self._meeting = meeting_recorder      # MeetingRecorder (see meeting.py)
        self._memory = memory  # optional semantic memory (see memory.py)
        self._store = store    # optional durable state (see state.py)
        self._voiceprint_path = voiceprint_path
        self._password_path = password_path
        self._command_timeout = command_timeout
        self._plugins_dir = plugins_dir
        self._auto_install_deps = auto_install_deps
        # Callable(messages)->str used to write tool code when the model's
        # embedded code is unusable; wired to the LLM in main.py.
        self._code_gen = code_generator
        # Callable() that wipes the live in-RAM conversation; set in main.py so
        # reset_all also clears what was said this session.
        self._clear_session = None
        # reset_all sets this; main.py restarts the process after speaking.
        self.restart_requested = False

        # Login authority of the current session. Risky/system tools in
        # _admin_only are refused for non-admins. In test mode the gate is off
        # (everything unrestricted). set_authority() updates it after login.
        self._authority = authority or "admin"
        self._test_mode = bool(test_mode)
        self._admin_only = set(admin_only_tools or [])
        # Callable(name, authority)->str that registers a new person live;
        # main.py wires it with the camera/mic/verifier (see auth.register_new_user).
        self._register_user_fn = register_user_fn

        # web_search backend selection. The MCP URL embeds the API key, so it
        # also serves as a key source for the REST backend.
        self._backend = web_search_backend
        self._tavily_url = tavily_mcp_url
        self._tavily_key = tavily_api_key or _key_from_mcp_url(tavily_mcp_url)

        # name -> (handler, description, arguments-help)
        self._registry: dict[str, tuple[Callable[[dict], str], str, str]] = {
            "get_time": (self._get_time, "Get the current local time.", "{}"),
            "get_date": (self._get_date, "Get today's date.", "{}"),
            "set_timer": (
                self._set_timer,
                "Start a countdown timer that alerts when it elapses.",
                '{"seconds": <int>, "label": "<short name>"}',
            ),
            "web_search": (
                self._web_search,
                "Search the web for a short factual answer. Needs internet.",
                '{"query": "<search terms>"}',
            ),
            "remember": (
                self._remember,
                "Permanently remember a fact, preference, correction, or "
                "standing instruction the user gives you (e.g. 'remember "
                "that...', 'from now on...', 'call me...', 'actually it's...'). "
                "Saved and always applied in future conversations.",
                '{"text": "<the thing to remember, as a clear statement>"}',
            ),
            "forget": (
                self._forget,
                "Forget a specific thing the user previously told you to "
                "remember.",
                '{"text": "<what to forget>"}',
            ),
            "reset_all": (
                self._reset_all,
                "Factory-reset Atlas: erase ALL memory and conversation "
                "history, the cache, the enrolled voice and face, and the "
                "startup password. Irreversible. Only when the user clearly "
                "asks to reset everything / reset yourself / wipe everything.",
                "{}",
            ),
            "register_user": (
                self._register_user,
                "Register a NEW person so they can use Atlas: enrolls their face "
                "and voice and sets their authority. Use for 'register a new "
                "user', 'add a user', 'enroll my friend'. Admin-only; only one "
                "admin is allowed, so new people are 'user' or 'guest'.",
                '{"name": "<person\'s name>", "authority": "user|guest"}',
            ),
        }

        if enable_self_extend:
            self._registry.update({
                "create_tool": (
                    self._create_tool,
                    "Permanently add a NEW custom voice-assistant tool/capability to "
                    "yourself when the user asks to expand your abilities. Just give a "
                    "snake_case 'name', a clear 'description' of what it does, and an "
                    "'arguments' schema — you do NOT need to write the 'code'; leave it "
                    "out and Atlas writes the function for you. DO NOT use this to create "
                    "regular files, scripts, or user programs (use code_agent/write_file).",
                    '{"name": "<snake_case>", "description": "<what it should do, in detail>", "arguments": "<JSON schema, e.g. {}>"}',
                ),
                "remove_tool": (
                    self._remove_tool,
                    "Delete a custom tool that was previously added with "
                    "create_tool (cannot remove built-in tools).",
                    '{"name": "<tool name>"}',
                ),
            })

        # CrewAI coding agent (delegated to an isolated venv; see code_agent.py).
        if code_agent is not None and getattr(code_agent, "available", False):
            self._registry["code_agent"] = (
                self._code_agent_tool,
                "Delegate a programming / coding / software / debugging / "
                "architecture task to a senior engineer agent (CrewAI, cloud "
                "LLM). Use this for ANY request to write, fix, build, review, or "
                "design code. Pass the user's request as the task.",
                '{"task": "<the full coding request, verbatim>"}',
            )

        # Productivity: documents always; OCR needs the VLM; meeting needs STT.
        self._registry["read_pdf"] = (
            self._read_pdf,
            "Read a PDF (or txt/md) file and return its text, or answer a "
            "question about it. Use for 'read this PDF', 'what does <file> say'.",
            '{"path": "<file path>", "question": "<optional question>"}',
        )
        self._registry["summarize_document"] = (
            self._summarize_document,
            "Summarize a document (pdf/txt/md): a short summary plus key points. "
            "Use for 'summarize <file>'.",
            '{"path": "<file path>"}',
        )
        if vision is not None and getattr(vision, "available", False):
            self._registry["read_text"] = (
                self._read_text,
                "OCR: read and return ALL the text in the screen or an image "
                "file, verbatim. Use for 'read the text on screen', 'extract the "
                "text from this image'.",
                '{"source": "screen" | "<image file path>"}',
            )
        if transcriber is not None and meeting_recorder is not None:
            self._registry["start_meeting"] = (
                self._start_meeting,
                "Start recording a meeting (microphone) in the background. Use "
                "for 'start meeting notes', 'record this meeting'.",
                "{}",
            )
            self._registry["stop_meeting"] = (
                self._stop_meeting,
                "Stop the meeting recording, then transcribe it and produce "
                "notes/summary/action items (saved to a file). Use for 'stop "
                "meeting', 'end the meeting notes'.",
                "{}",
            )

        # Web & info (free endpoints / local model — no API keys).
        self._registry.update({
            "read_url": (
                self._read_url,
                "Fetch a web page and summarize it, or answer a question about "
                "it. Use for 'read this link', 'summarize <url>'.",
                '{"url": "<url>", "question": "<optional>"}',
            ),
            "get_weather": (
                self._get_weather,
                "Current weather for a place (or here if none given).",
                '{"location": "<city, optional>"}',
            ),
            "get_news": (
                self._get_news,
                "Top news headlines, optionally about a topic.",
                '{"topic": "<optional topic>"}',
            ),
            "get_stock": (
                self._get_stock,
                "Latest stock/ETF price and daily change for a ticker symbol.",
                '{"symbol": "<e.g. AAPL>"}',
            ),
            "get_crypto": (
                self._get_crypto,
                "Current cryptocurrency price (USD) and 24h change.",
                '{"coin": "<e.g. bitcoin or btc>"}',
            ),
            "translate": (
                self._translate,
                "Translate text into a target language.",
                '{"text": "<text>", "to": "<language>"}',
            ),
            "calculate": (
                self._calculate,
                "Safely evaluate a math expression (arithmetic + functions like "
                "sqrt, sin; constants pi, e). No code execution.",
                '{"expression": "<e.g. (2+3)*sqrt(16)>"}',
            ),
            "convert_currency": (
                self._convert_currency,
                "Convert an amount between currencies at the latest rate.",
                '{"amount": <number>, "from": "<USD>", "to": "<EUR>"}',
            ),
            "convert_units": (
                self._convert_units,
                "Convert between units of length, mass, volume, or temperature.",
                '{"amount": <number>, "from": "<km>", "to": "<mi>"}',
            ),
        })

        if enable_system_control:
            self._registry.update({
                "set_volume": (
                    self._set_volume,
                    "Set system volume to a percentage, or mute/unmute.",
                    '{"percent": <0-100>}  or  {"mute": true/false}',
                ),
                "set_app_volume": (
                    self._set_app_volume,
                    "Set volume or mute for a specific running app (e.g. Spotify).",
                    '{"app": "<name>", "percent": <0-100>}  or  {"app": "<name>", "mute": true/false}',
                ),
                "set_brightness": (
                    self._set_brightness,
                    "Set the display brightness to a percentage.",
                    '{"percent": <0-100>}',
                ),
                "take_screenshot": (
                    self._take_screenshot,
                    "Capture the screen and save it to a file.",
                    "{}",
                ),
                "media_control": (
                    self._media_control,
                    "Control playback of ALREADY-playing media (current session).",
                    '{"action": "play"|"pause"|"play_pause"|"next"|"previous"|"stop"}',
                ),
                "play_music": (
                    self._play_music,
                    "Play a specific song, music, or video on YouTube (opens the "
                    "top result and plays it). Use this to START playing something.",
                    '{"query": "<song / artist / video>"}',
                ),
                "now_playing": (
                    self._now_playing,
                    "Say what media (song/video) is currently playing.",
                    "{}",
                ),
                "open_app": (
                    self._open_app,
                    "Open/launch an application on this computer.",
                    '{"name": "<app name, e.g. notepad>"}',
                ),
                "close_app": (
                    self._close_app,
                    "Close or quit a desktop application by name. ALWAYS use this "
                    "whenever the user asks to close/quit/exit an app (e.g. "
                    "'close Notepad', 'quit Chrome'). Do NOT assume whether it's "
                    "running or claim it isn't — this tool checks and closes it "
                    "and reports the real result.",
                    '{"name": "<app name, e.g. notepad>"}',
                ),
                "type_text": (
                    self._type_text,
                    "Type/write text into the currently focused window (e.g. "
                    "after opening Notepad). Use for 'write/type X'.",
                    '{"text": "<text to type>"}',
                ),
                "press_keys": (
                    self._press_keys,
                    "Press a key or key combination (hotkey).",
                    '{"keys": "<e.g. enter | ctrl+s | alt+tab | win+d>"}',
                ),
                "mouse_click": (
                    self._mouse_click,
                    "Click the mouse (optionally at screen x,y; supports double).",
                    '{"button": "left"|"right"|"middle", "x": <int?>, "y": <int?>, "double": <bool?>}',
                ),
                "move_mouse": (
                    self._move_mouse,
                    "Move the mouse cursor to screen coordinates.",
                    '{"x": <int>, "y": <int>}',
                ),
                "mouse_scroll": (
                    self._mouse_scroll,
                    "Scroll the mouse wheel (positive=up, negative=down).",
                    '{"amount": <int, e.g. -3>}',
                ),
                "open_website": (
                    self._open_website,
                    "Open a website in the browser.",
                    '{"url": "<url or domain>"}',
                ),
                "create_github_repo": (
                    self._create_github_repo,
                    "Create a new GitHub repository (uses the gh CLI; needs gh "
                    "to be authenticated).",
                    '{"name": "<repo-name>", "private": <bool?>, "description": "<text?>"}',
                ),
                "system_power": (
                    self._system_power,
                    "Lock, sleep, shut down, restart, or cancel a pending "
                    "shutdown (shutdown/restart may be disabled).",
                    '{"action": "lock"|"sleep"|"shutdown"|"restart"|"cancel"}',
                ),
            })

        # Screen vision (only if a VLM is available — see vision.py).
        if vision is not None and getattr(vision, "available", False):
            self._registry["see_screen"] = (
                self._see_screen,
                "Look at the user's screen and answer a question about what is "
                "shown (read text, describe it, find something). Use for "
                "'what's on my screen', 'read this', 'what does this say'.",
                '{"question": "<what to look for; empty = describe the screen>"}',
            )
            # Vision-guided control: locate an on-screen element by description
            # and act on it. Use these to operate apps that have no dedicated
            # tool ("click the Submit button", "type into the search box").
            self._registry["click_element"] = (
                self._click_element,
                "Find a UI element on screen by description and click it "
                "(buttons, links, fields, icons). Use when there's no specific "
                "tool for the target.",
                '{"description": "<element, e.g. the Save button>", "button": "left"|"right"?, "double": <bool?>}',
            )
            self._registry["type_into"] = (
                self._type_into,
                "Click an on-screen element by description, then type text into "
                "it (e.g. a search or text box).",
                '{"description": "<the field>", "text": "<text to type>", "enter": <bool?>}',
            )
            self._registry["find_on_screen"] = (
                self._find_on_screen,
                "Locate a UI element on screen by description and report where "
                "it is, WITHOUT clicking. Use to check before acting.",
                '{"description": "<element to find>"}',
            )
            # Camera: look through the webcam (the same VLM, a webcam frame).
            self._registry["see_camera"] = (
                self._see_camera,
                "Look through the webcam and answer a question about what the "
                "camera sees (the user, an object they hold up, the room). Use "
                "for 'look at me', 'what am I holding', 'what do you see'.",
                '{"question": "<what to look for; empty = describe>", "camera": <index?>}',
            )
            self._registry["take_photo"] = (
                self._take_photo,
                "Take a photo with the webcam and save it to photos/.",
                '{"camera": <index, default 0>}',
            )

        # Face recognition (needs the camera + an enrolled face database).
        if faces is not None and getattr(faces, "available", False):
            self._registry["recognize_face"] = (
                self._recognize_face,
                "Look at the webcam and identify who is there by matching "
                "enrolled faces. Use for 'who is this', 'who am I', 'do you "
                "recognize me'.",
                '{"camera": <index?>}',
            )
            self._registry["enroll_face"] = (
                self._enroll_face,
                "Save the face currently at the webcam under a name, so it can "
                "be recognized later. Use for 'remember my face as X', 'this is "
                "<name>'.",
                '{"name": "<person\'s name>", "camera": <index?>}',
            )
            self._registry["open_face_window"] = (
                self._open_face_window,
                "Open a live face-recognition window showing the webcam with "
                "names/boxes drawn on recognized faces. Use for 'open the face "
                "recognition window', 'show me the camera with faces'.",
                '{"camera": <index?>}',
            )

        # Coding: read/write/edit files and run commands. write_file overwriting
        # an existing file and run_command are gated by spoken confirmation in
        # the agent (see agents.Orchestrator). These local coding tools are a
        # FALLBACK — when the CrewAI coding agent is available it owns all coding,
        # so they're only registered when CrewAI is absent.
        crew_active = code_agent is not None and getattr(
            code_agent, "available", False)
        if enable_coding:
            self._registry.update({
                "write_file": (
                    self._write_file,
                    "Write, create, or overwrite any local text or source code file on disk "
                    "(including Python scripts like helloworld.py, text documents, or logs). "
                    "Always use this for creating files.",
                    '{"path": "<file path>", "content": "<full file text>"}',
                ),
                "read_file": (
                    self._read_file,
                    "Read and return the contents of a text/code file.",
                    '{"path": "<file path>"}',
                ),
                "edit_file": (
                    self._edit_file,
                    "Replace exact text in an existing file (find -> replace). "
                    "Read the file first so 'find' matches exactly.",
                    '{"path": "<file path>", "find": "<exact text>", "replace": "<new text>", "all": <bool?>}',
                ),
                "list_dir": (
                    self._list_dir,
                    "List the files and folders in a directory.",
                    '{"path": "<directory, default current>"}',
                ),
                "run_command": (
                    self._run_command,
                    "Run a shell command (e.g. run/compile/test code) and "
                    "return its output. Executes on the machine.",
                    '{"command": "<command line>", "cwd": "<working dir?>"}',
                ),
                "check_syntax": (
                    self._check_syntax,
                    "Check a Python file for syntax errors WITHOUT running it "
                    "(safe). Returns the first error's line, or that it's clean.",
                    '{"path": "<.py file>"}',
                ),
                "debug_python": (
                    self._debug_python,
                    "Run a Python script and, if it crashes, return a parsed "
                    "traceback: the error type/message, the exact file:line, and "
                    "the offending source line. Use this to find and fix bugs.",
                    '{"path": "<.py file>", "args": ["<cli arg>", ...]?}',
                ),
            })

        # Everything registered so far is built-in and protected from being
        # overwritten/removed by the self-extension tools. Then load any custom
        # tools the user has previously had Atlas create.
        self._builtin_names = set(self._registry)
        if enable_self_extend:
            self._load_plugins()

    # ---- prompt help -----------------------------------------------------
    def prompt_block(self, allowed: Optional[set] = None,
                     instructions: bool = True) -> str:
        """Human-readable tool list to embed in the system prompt.

        If `allowed` is given, only those tool names are listed (role scoping);
        otherwise every registered tool is listed. With `instructions=False`,
        only the tool list is returned (the caller supplies the protocol — used
        by the ReAct agent, which has its own loop instructions).
        """
        names = [n for n in self._registry if allowed is None or n in allowed]
        if not names:
            return "You have no tools available; answer directly."
        lines = ["You can use these tools:"]
        for name in names:
            _, desc, args = self._registry[name]
            lines.append(f'- {name}: {desc} arguments: {args}')
        if not instructions:
            return "\n".join(lines)
        lines.append(
            "When a tool is needed, your ENTIRE reply must be a single JSON "
            'object and nothing else: {"tool": "<name>", "arguments": {...}}. '
            "Decide immediately. Never say you will look something up, never ask "
            "the user to wait, and never ask a clarifying question when "
            "web_search could answer it — search instead. "
            'Example — user: "weather in Tokyo" you: '
            '{"tool": "web_search", "arguments": {"query": "current weather in Tokyo"}} '
            'Example — user: "F1 champion" you: '
            '{"tool": "web_search", "arguments": {"query": "current Formula 1 world champion"}} '
            "After you receive the tool result, give the user a direct one-"
            "sentence spoken answer. If no tool is needed, just answer directly."
        )
        return "\n".join(lines)

    @staticmethod
    def _normalize_arguments(value) -> dict:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return {}
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    def _normalize_tool_call(self, obj: object) -> Optional[dict]:
        if not isinstance(obj, dict):
            return None

        if isinstance(obj.get("tool_call"), dict):
            obj = obj["tool_call"]
        elif isinstance(obj.get("function_call"), dict):
            obj = obj["function_call"]

        name = None
        for key in ("tool", "name", "function", "tool_name"):
            value = obj.get(key)
            if isinstance(value, str) and value in self._registry:
                name = value
                break

        if name is None:
            return None

        arguments = None
        for key in ("arguments", "args", "parameters", "input", "payload"):
            if key in obj:
                arguments = self._normalize_arguments(obj.get(key))
                break
        if arguments is None:
            arguments = {}
        return {"name": name, "arguments": arguments}

    # ---- detection + dispatch -------------------------------------------
    def parse(self, text: str) -> Optional[dict]:
        """Return {"name", "arguments"} if `text` is a tool call, else None."""
        for blob in _extract_json_objects(text):
            try:
                obj = json.loads(blob)
            except json.JSONDecodeError:
                continue
            parsed = self._normalize_tool_call(obj)
            if parsed is not None:
                return parsed
        # Fallback: small models often emit slightly-broken JSON when the call
        # embeds code (a stray/missing brace, an unescaped quote). If it looks
        # like an attempted tool call, repair the JSON and try once more.
        if '"tool"' in text or "'tool'" in text:
            try:
                from json_repair import repair_json

                obj = repair_json(text, return_objects=True)
                for cand in (obj if isinstance(obj, list) else [obj]):
                    parsed = self._normalize_tool_call(cand)
                    if parsed is not None:
                        return parsed
            except Exception:
                pass
        return None

    def set_authority(self, authority: str) -> None:
        """Set the current session's login authority (admin/user/guest)."""
        self._authority = authority or "admin"

    def execute(self, call: dict) -> str:
        """Run a parsed tool call and return a short text result."""
        name = call["name"]
        # Authority gate: non-admins can't run admin-only tools. Test mode is
        # fully unrestricted (it bypasses auth entirely).
        if (not self._test_mode and self._authority != "admin"
                and name in self._admin_only):
            return (f"Sorry, '{name.replace('_', ' ')}' is an admin-only action "
                    f"and you're signed in as {self._authority}. Ask an admin.")
        handler = self._registry[name][0]
        try:
            return handler(call.get("arguments", {}))
        except Exception as e:  # never let a tool crash the loop
            return f"Tool '{name}' failed: {e}"

    # ---- tool implementations -------------------------------------------
    def _get_time(self, args: dict) -> str:
        return "The current time is " + datetime.datetime.now().strftime("%I:%M %p").lstrip("0")

    def _get_date(self, args: dict) -> str:
        return "Today is " + datetime.datetime.now().strftime("%A, %B %d, %Y")

    def _remember(self, args: dict) -> str:
        if self._memory is None or not getattr(self._memory, "enabled", False):
            return "My long-term memory is off right now, so I can't save that."
        text = str(args.get("text", "")).strip()
        if not text:
            return "What would you like me to remember?"
        return ("Got it — I'll remember that." if self._memory.note(text)
                else "I couldn't save that to memory.")

    def _forget(self, args: dict) -> str:
        if self._memory is None or not getattr(self._memory, "enabled", False):
            return "My long-term memory is off right now."
        text = str(args.get("text", "")).strip()
        if not text:
            return "What should I forget?"
        removed = self._memory.forget(text)
        return (f"Okay, I've forgotten that: {removed}" if removed
                else "I couldn't find anything like that to forget.")

    def _code_agent_tool(self, args: dict) -> str:
        if self._code_agent is None or not getattr(self._code_agent,
                                                    "available", False):
            return "The CrewAI coding agent isn't available right now."
        task = str(args.get("task", "")).strip()
        if not task:
            return "What coding task should I hand to the engineer?"
        return self._code_agent.run(task)

    # ---- productivity: OCR / PDF / summaries / meetings -----------------
    def _summarize_text(self, text: str, instruction: str) -> str:
        """Summarize/answer over `text` with the local LLM, map-reduce for long
        text. Returns '' if no LLM is wired."""
        if self._code_gen is None:
            return ""
        text = (text or "").strip()
        if not text:
            return ""

        def llm(chunk: str) -> str:
            try:
                return self._code_gen([
                    {"role": "system", "content": "You are a concise, accurate "
                     "assistant."},
                    {"role": "user", "content": f"{instruction}\n\n{chunk} "
                     "/no_think"}]).strip()
            except Exception:
                return ""

        if len(text) <= 6000:
            return llm(text)
        from rag import chunk_text
        chunks = chunk_text(text, 6000, 200)[:8]   # cap work on the local model
        partials = [p for p in (llm(c) for c in chunks) if p]
        if not partials:
            return ""
        return llm("Combine these section notes into one cohesive result:\n\n"
                   + "\n\n".join(partials))

    def _read_text(self, args: dict) -> str:
        if self._vision is None or not getattr(self._vision, "available", False):
            return "Screen/image reading is unavailable (no vision model)."
        from PIL import Image, ImageGrab

        source = str(args.get("source", "screen")).strip()
        if source.lower() in ("", "screen"):
            img = ImageGrab.grab()
        else:
            path = self._resolve_path(source)
            if not os.path.isfile(path):
                return f"No such image file: {path}"
            try:
                img = Image.open(path)
            except Exception as e:
                return f"Couldn't open the image: {e}"
        return self._vision.look(
            img, "Transcribe ALL text in this image exactly, preserving line "
            "breaks. Output only the text.")

    def _read_pdf(self, args: dict) -> str:
        from rag import read_document

        path = self._resolve_path(args.get("path", ""))
        if not os.path.isfile(path):
            return f"No such file: {path}"
        text = read_document(path)
        if not text.strip():
            return ("I couldn't extract any text — it may be a scanned/image-"
                    "only PDF. Try 'read the text' with it open on screen.")
        question = str(args.get("question", "")).strip()
        if question:
            ans = self._summarize_text(text, f"Answer this about the document: "
                                       f"{question}")
            return ans or text[:2000]
        return text if len(text) <= 6000 else text[:6000] + "\n... (truncated)"

    def _summarize_document(self, args: dict) -> str:
        from rag import read_document

        path = self._resolve_path(args.get("path", ""))
        if not os.path.isfile(path):
            return f"No such file: {path}"
        text = read_document(path)
        if not text.strip():
            return "I couldn't read any text from that file."
        summary = self._summarize_text(
            text, "Summarize this document concisely: a 2-3 sentence overview, "
            "then the key points as a short bullet list.")
        return summary or "I couldn't summarize that document."

    def _start_meeting(self, args: dict) -> str:
        if self._meeting is None:
            return "Meeting recording isn't available."
        if getattr(self._meeting, "active", False):
            return "I'm already recording the meeting."
        if self._meeting.start():
            return ("Recording the meeting. Say 'stop meeting' when you're done "
                    "and I'll write up the notes.")
        return "I couldn't start recording (microphone busy?)."

    def _stop_meeting(self, args: dict) -> str:
        if self._meeting is None or not getattr(self._meeting, "active", False):
            return "No meeting is being recorded."
        audio = self._meeting.stop()
        if audio.size == 0:
            return "I didn't capture any audio."
        if self._transcriber is None:
            return "I recorded it but transcription isn't available."
        text, _ = self._transcriber.transcribe(audio)
        if not text.strip():
            return "I recorded the meeting but couldn't make out any speech."
        notes = self._summarize_text(
            text, "This is a meeting transcript. Write: a short summary, the key "
            "points/decisions, and a list of action items (with who/what if "
            "stated).") or text
        path = ""
        try:
            os.makedirs("meetings", exist_ok=True)
            name = "meeting_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".md"
            path = os.path.join("meetings", name)
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(f"# Meeting notes\n\n{notes}\n\n---\n\n## Transcript\n\n{text}\n")
        except Exception:
            path = ""
        mins = audio.size / max(self._transcriber.cfg.sample_rate, 1) / 60
        head = notes.splitlines()[0][:200] if notes else ""
        return (f"Meeting saved ({mins:.0f} min) to {path}. {head}" if path
                else notes[:1500])

    # ---- web & info -----------------------------------------------------
    @staticmethod
    def _http_get(url: str, timeout: int = 12) -> str:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (Atlas)"})
        with urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")

    def _read_url(self, args: dict) -> str:
        url = str(args.get("url", "")).strip()
        if not url:
            return "What page should I read?"
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        try:
            html = self._http_get(url, timeout=15)
        except Exception as e:
            return f"Couldn't fetch that page: {e}"
        text = _html_to_text(html)
        if not text:
            return "That page had no readable text."
        q = str(args.get("question", "")).strip()
        instr = (f"Answer this about the page: {q}" if q else
                 "Summarize this web page concisely: a 2-3 sentence overview "
                 "then the key points.")
        return self._summarize_text(text[:20000], instr) or text[:1500]

    def _get_weather(self, args: dict) -> str:
        loc = str(args.get("location", "")).strip()
        url = "https://wttr.in/" + quote_plus(loc) + "?format=j1"
        try:
            data = json.loads(self._http_get(url, timeout=12))
            cur = data["current_condition"][0]
            place = loc
            try:
                place = data["nearest_area"][0]["areaName"][0]["value"]
            except Exception:
                pass
            desc = cur["weatherDesc"][0]["value"]
            return (f"{place or 'Here'}: {desc}, {cur['temp_C']}°C "
                    f"(feels {cur['FeelsLikeC']}°C), humidity {cur['humidity']}%, "
                    f"wind {cur['windspeedKmph']} km/h.")
        except Exception as e:
            return f"Couldn't get the weather: {e}"

    def _get_news(self, args: dict) -> str:
        topic = str(args.get("topic", "")).strip()
        if topic:
            url = ("https://news.google.com/rss/search?q=" + quote_plus(topic)
                   + "&hl=en-US&gl=US&ceid=US:en")
        else:
            url = "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"
        try:
            xml = self._http_get(url, timeout=12)
        except Exception as e:
            return f"Couldn't get the news: {e}"
        import html as _h
        # Only item titles (avoids the channel/image <title> e.g. "Google News").
        titles = [_h.unescape(t).strip() for t in
                  re.findall(r"<item>.*?<title>(.*?)</title>", xml, re.DOTALL)]
        titles = [t for t in titles if t][:5]
        if not titles:
            return "I couldn't find any headlines."
        lead = f"Top '{topic}' headlines: " if topic else "Top headlines: "
        return lead + " | ".join(titles)

    def _get_stock(self, args: dict) -> str:
        sym = str(args.get("symbol", "")).strip().upper()
        if not sym:
            return "Which stock symbol?"
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote_plus(sym)}"
        try:
            meta = json.loads(self._http_get(url, timeout=12))["chart"]["result"][0]["meta"]
            price = meta.get("regularMarketPrice")
            prev = meta.get("previousClose") or meta.get("chartPreviousClose")
            cur = meta.get("currency", "")
            if price is None:
                return f"I couldn't find a price for {sym}."
            out = f"{sym}: {price:g} {cur}".strip()
            if prev:
                out += f" ({(price - prev) / prev * 100:+.2f}% today)"
            return out
        except Exception as e:
            return f"Couldn't get the stock price: {e}"

    def _get_crypto(self, args: dict) -> str:
        coin = str(args.get("coin", "")).strip().lower()
        if not coin:
            return "Which cryptocurrency?"
        cid = _COIN_IDS.get(coin, coin.replace(" ", "-"))
        url = ("https://api.coingecko.com/api/v3/simple/price?ids="
               + quote_plus(cid) + "&vs_currencies=usd&include_24hr_change=true")
        try:
            data = json.loads(self._http_get(url, timeout=12))
        except Exception as e:
            return f"Couldn't get the price: {e}"
        if cid not in data:
            return f"I couldn't find '{coin}'."
        price = data[cid].get("usd")
        chg = data[cid].get("usd_24h_change")
        out = f"{cid.replace('-', ' ').title()}: ${price:,}"
        if chg is not None:
            out += f" ({chg:+.1f}% 24h)"
        return out

    def _translate(self, args: dict) -> str:
        text = str(args.get("text", "")).strip()
        to = str(args.get("to", "")).strip() or "English"
        if not text:
            return "What should I translate?"
        if self._code_gen is None:
            return "Translation isn't available right now."
        try:
            out = self._code_gen([
                {"role": "system", "content": "You are a translator. Output "
                 "ONLY the translation, nothing else."},
                {"role": "user", "content": f"Translate to {to}:\n{text} "
                 "/no_think"}]).strip()
        except Exception as e:
            return f"Couldn't translate: {e}"
        return out or "I couldn't translate that."

    def _calculate(self, args: dict) -> str:
        import ast

        expr = str(args.get("expression", "")).strip()
        if not expr:
            return "What should I calculate?"
        try:
            val = _safe_eval(ast.parse(expr, mode="eval").body)
            if val is None:
                raise ValueError
            return f"{expr} = {val:g}" if isinstance(val, float) else f"{expr} = {val}"
        except Exception:
            return f"I couldn't evaluate '{expr}'."

    def _convert_currency(self, args: dict) -> str:
        try:
            amt = float(args.get("amount", 1) or 1)
        except Exception:
            amt = 1.0
        frm = str(args.get("from", "")).strip().upper()
        to = str(args.get("to", "")).strip().upper()
        if not frm or not to:
            return "Which currencies? (e.g. USD to EUR)"
        url = (f"https://api.frankfurter.app/latest?amount={amt}"
               f"&from={quote_plus(frm)}&to={quote_plus(to)}")
        try:
            rate = json.loads(self._http_get(url, timeout=12))["rates"].get(to)
        except Exception as e:
            return f"Couldn't convert: {e}"
        if rate is None:
            return f"I couldn't convert {frm} to {to}."
        return f"{amt:g} {frm} = {rate:g} {to}"

    def _convert_units(self, args: dict) -> str:
        try:
            amt = float(args.get("amount", 1) or 1)
        except Exception:
            amt = 1.0
        frm = str(args.get("from", "")).strip().lower()
        to = str(args.get("to", "")).strip().lower()
        temp = {"c", "celsius", "f", "fahrenheit", "k", "kelvin"}
        if frm in temp and to in temp:
            c = amt if frm[0] == "c" else (amt - 32) * 5 / 9 if frm[0] == "f" else amt - 273.15
            out = c if to[0] == "c" else c * 9 / 5 + 32 if to[0] == "f" else c + 273.15
            return f"{amt:g}°{frm[0].upper()} = {out:g}°{to[0].upper()}"
        a, b = _UNITS.get(frm), _UNITS.get(to)
        if not a or not b or a[0] != b[0]:
            return f"I can't convert '{frm}' to '{to}'."
        return f"{amt:g} {frm} = {amt * a[1] / b[1]:g} {to}"

    def _reset_all(self, args: dict) -> str:
        """Factory reset: wipe memory, history, cache, voiceprint, and faces.

        Irreversible. The agent confirms (spoken 'yes') before this runs.
        """
        done: list[str] = []
        if self._memory is not None and getattr(self._memory, "enabled", False):
            done.append("memory cleared" if self._memory.reset()
                        else "memory could not be cleared")
        if self._store is not None and getattr(self._store, "enabled", False):
            done.append("conversation history cleared" if self._store.reset()
                        else "history could not be cleared")
        if self.cache is not None and getattr(self.cache, "enabled", False):
            n = self.cache.reset()
            done.append(f"cache cleared ({n} keys)")
        if self._voiceprint_path and os.path.exists(self._voiceprint_path):
            try:
                os.remove(self._voiceprint_path)
                done.append("voice enrollment removed")
            except Exception:
                done.append("voice enrollment could not be removed")
        if self._password_path and os.path.exists(self._password_path):
            try:
                os.remove(self._password_path)
                done.append("startup password removed")
            except Exception:
                done.append("startup password could not be removed")
        if self._faces is not None and getattr(self._faces, "available", False):
            done.append("face enrollments removed" if self._faces.reset()
                        else "faces could not be removed")
        # Wipe the live in-RAM conversation too, so I forget what was said this
        # session immediately (not just the persistent stores).
        if self._clear_session is not None:
            try:
                self._clear_session()
                done.append("current conversation forgotten")
            except Exception:
                pass
        if not done:
            return "There was nothing to reset."
        # Ask main to restart so onboarding (face/voice/password) runs fresh.
        self.restart_requested = True
        return ("Done — I reset everything: " + "; ".join(done) + ". "
                "I no longer know who you are. Restarting now to set up again.")

    # ---- self-extension: create / load / remove custom tools ------------
    _PLUGIN_TEMPLATE = (
        '"""Atlas custom tool \'{name}\' — created via voice on {date}."""\n\n'
        "TOOL = {{\n"
        "    \"name\": {name!r},\n"
        "    \"description\": {description!r},\n"
        "    \"arguments\": {arguments!r},\n"
        "}}\n\n\n"
        "{code}\n"
    )

    @staticmethod
    def _safe_tool_name(name: str) -> str:
        name = str(name or "").strip().lower().replace(" ", "_").replace("-", "_")
        return name if re.match(r"^[a-z][a-z0-9_]{0,39}$", name) else ""

    def _load_plugins(self) -> None:
        """Load previously-created custom tools from the plugins directory."""
        d = self._plugins_dir
        if not d or not os.path.isdir(d):
            return
        loaded = 0
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".py") or fn.startswith("_"):
                continue
            path = os.path.join(d, fn)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    src = f.read()
                ns: dict = {}
                try:
                    exec(compile(src, path, "exec"), ns)
                except ImportError:
                    # A declared dependency is missing — install it and retry
                    # once, so tools keep working across restarts / new machines.
                    reqs = self._extract_requirements(src)
                    if reqs and self._auto_install_deps and self._install_deps(reqs)[0]:
                        print(f"[tools] installed deps for {fn}: {', '.join(reqs)}")
                        ns = {}
                        exec(compile(src, path, "exec"), ns)
                    else:
                        raise
                run = ns.get("run")
                meta = ns.get("TOOL") or {}
                name = self._safe_tool_name(meta.get("name") or fn[:-3])
                if callable(run) and name and name not in self._registry:
                    self._registry[name] = (
                        run, meta.get("description", "Custom tool."),
                        meta.get("arguments", "{}"))
                    loaded += 1
            except Exception as e:
                print(f"[tools] skipped plugin {fn}: {e}")
        if loaded:
            print(f"Custom tools: loaded {loaded} from {d}/.")

    @staticmethod
    def _syntax_ok(code: str) -> bool:
        """True if `code` parses and defines run() — WITHOUT importing anything,
        so code that uses not-yet-installed third-party packages still passes."""
        if not code or "def run" not in code:
            return False
        try:
            return "run" in compile(code, "<tool>", "exec").co_names or True
        except SyntaxError:
            return False

    @staticmethod
    def _tool_code_ok(code: str) -> bool:
        """True if `code` defines a callable run(args) and executes cleanly
        (imports resolve — call only after dependencies are installed)."""
        if not code or "def run" not in code:
            return False
        try:
            ns: dict = {}
            exec(compile(code, "<tool-check>", "exec"), ns)
            return callable(ns.get("run"))
        except Exception:
            return False

    @staticmethod
    def _extract_requirements(code: str) -> list:
        """Pull pip package names from `# pip: pkg1 pkg2` lines in the code."""
        reqs: list[str] = []
        for line in (code or "").splitlines():
            m = re.match(r"\s*#\s*pip:\s*(.+)", line, re.IGNORECASE)
            if m:
                reqs += m.group(1).replace(",", " ").split()
        ok = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[\w,.-]+\])?"
                        r"([=<>!~]=?[\w.*+-]+)?$")
        # de-dup, keep order, reject anything that isn't a plain requirement spec
        seen, out = set(), []
        for r in reqs:
            r = r.strip()
            if r and r.lower() not in seen and ok.match(r):
                seen.add(r.lower())
                out.append(r)
        return out

    def _install_deps(self, reqs: list) -> tuple:
        """pip-install `reqs` into Atlas's venv (where plugins run). Returns
        (ok, installed_list)."""
        import subprocess

        if not reqs:
            return True, []
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "install", *reqs],
                capture_output=True, text=True, timeout=600)
            return proc.returncode == 0, reqs
        except Exception:
            return False, reqs

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        text = (text or "").strip()
        m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return text

    def _generate_tool_code(self, name: str, description: str,
                            arguments: str) -> str:
        """Write the tool's `run(args)` function. Prefers the CrewAI code agent
        (cloud) for reliable code; falls back to the local model. Returns code
        that compiles, or ''."""
        prompt = (
            f"Write one Python function for a tool called '{name}'.\n"
            f"What it must do: {description}\n"
            f"Signature: def run(args):  where args is a dict with these "
            f"arguments: {arguments}. Read inputs from args (e.g. "
            "args['x']). Return a short human-readable string (do not print).\n"
            "Prefer the standard library, but you MAY use well-known third-party "
            "packages. If you use any, put their pip install names on the FIRST "
            "line as a comment, e.g.  # pip: requests beautifulsoup4\n"
            "Output ONLY the function code (with that optional first comment) — "
            "no markdown fences, no commentary, no example calls.")

        # 1) CrewAI writes the code if it's set up. Validate by SYNTAX only here
        # (deps may not be installed yet — they're installed in _create_tool).
        if self._code_agent is not None and getattr(
                self._code_agent, "available", False):
            try:
                code = self._strip_code_fences(self._code_agent.complete(prompt))
                if self._syntax_ok(code):
                    return code
            except Exception:
                pass

        # 2) Fall back to the local model (two attempts).
        if self._code_gen is not None:
            for _ in range(2):
                try:
                    text = self._code_gen([
                        {"role": "system", "content": "You write small, correct "
                         "Python functions. Output only valid code."},
                        {"role": "user", "content": prompt + " /no_think"}])
                except Exception:
                    break
                code = self._strip_code_fences(text)
                if self._syntax_ok(code):
                    return code
        return ""

    def _create_tool(self, args: dict) -> str:
        name = self._safe_tool_name(args.get("name", ""))
        if not name:
            return ("Invalid tool name. Use a short snake_case name like "
                    "'tell_joke'.")
        if name in getattr(self, "_builtin_names", set()):
            return f"'{name}' is a built-in tool and can't be replaced."
        if name in self._registry:
            return (f"A custom tool '{name}' already exists. Remove it first if "
                    "you want to replace it.")
        description = str(args.get("description", "")).strip() or "Custom tool."
        arguments = str(args.get("arguments", "") or "{}")
        code = str(args.get("code", "")).strip()

        # The model's JSON-embedded code is fragile (brace/quote escaping). Use
        # it only if it parses; otherwise generate the function from the
        # description with a clean, non-JSON LLM call (far more reliable).
        if not self._syntax_ok(code):
            code = self._generate_tool_code(name, description, arguments)
        if not self._syntax_ok(code):
            return ("I couldn't produce working code for that tool. Try "
                    "describing it more simply.")

        # Auto-install any third-party packages the tool declares (`# pip: ...`).
        note = ""
        reqs = self._extract_requirements(code)
        if reqs and self._auto_install_deps:
            ok, installed = self._install_deps(reqs)
            if not ok:
                return (f"The tool needs {', '.join(reqs)}, but I couldn't "
                        "install it. Check the package name or your internet.")
            if installed:
                note = f" (installed: {', '.join(installed)})"

        # Now that deps are present, the code must actually run + define run().
        if not self._tool_code_ok(code):
            return ("The tool code didn't run cleanly even after installing its "
                    "dependencies. Try describing it more simply.")

        module_text = self._PLUGIN_TEMPLATE.format(
            name=name, description=description, arguments=arguments, code=code,
            date=datetime.datetime.now().strftime("%Y-%m-%d"))
        ns: dict = {}
        exec(compile(module_text, f"<plugin {name}>", "exec"), ns)
        run = ns.get("run")
        try:
            os.makedirs(self._plugins_dir, exist_ok=True)
            with open(os.path.join(self._plugins_dir, f"{name}.py"), "w",
                      encoding="utf-8", newline="\n") as f:
                f.write(module_text)
        except Exception as e:
            return f"Couldn't save the new tool: {e}"
        # Register live so it's usable immediately.
        self._registry[name] = (run, description, arguments)
        return (f"Added a new tool '{name}'{note}. You can use it now and after "
                "restart.")

    def _remove_tool(self, args: dict) -> str:
        name = self._safe_tool_name(args.get("name", ""))
        if not name:
            return "Which custom tool should I remove?"
        if name in getattr(self, "_builtin_names", set()):
            return f"'{name}' is a built-in tool and can't be removed."
        path = os.path.join(self._plugins_dir, f"{name}.py")
        if not os.path.exists(path):
            return f"There's no custom tool called '{name}'."
        try:
            os.remove(path)
        except Exception as e:
            return f"Couldn't remove '{name}': {e}"
        self._registry.pop(name, None)
        return f"Removed the custom tool '{name}'."

    def _set_timer(self, args: dict) -> str:
        seconds = int(args.get("seconds", 0))
        label = str(args.get("label") or "timer")
        if seconds <= 0:
            return "Invalid timer duration."

        def fire() -> None:
            self._on_timer_fire(f"Timer '{label}' is done.")

        timer = threading.Timer(seconds, fire)
        timer.daemon = True
        timer.start()
        self._timers.append(timer)
        return f"Timer '{label}' set for {seconds} seconds."

    def _web_search(self, args: dict) -> str:
        query = str(args.get("query", "")).strip()
        if not query:
            return "No search query provided."

        # Serve from cache if we've searched this recently.
        ck = None
        if self.cache is not None and self.cache.enabled:
            import cache as cache_mod

            ck = cache_mod.key("web", query.lower())
            cached = self.cache.get(ck)
            if cached is not None:
                return cached

        result = self._run_web_search(query)

        if ck is not None and not result.startswith("Web search is unavailable"):
            self.cache.set(ck, result, ttl=self.cache.cfg.web_ttl)
        return result

    # ---- system control (delegates to system_control.py) ----------------
    def _set_volume(self, args: dict) -> str:
        import system_control as sc

        if "mute" in args:
            return sc.set_mute(bool(args["mute"]))
        if "percent" in args:
            return sc.set_volume(args["percent"])
        return "Specify a volume percent or mute true/false."

    def _set_app_volume(self, args: dict) -> str:
        import system_control as sc

        app = str(args.get("app", "")).strip()
        if not app:
            return "Specify which app."
        return sc.set_app_volume(app, args.get("percent"), args.get("mute"))

    def _set_brightness(self, args: dict) -> str:
        import system_control as sc

        if "percent" not in args:
            return "Specify a brightness percent."
        return sc.set_brightness(args["percent"])

    def _take_screenshot(self, args: dict) -> str:
        import system_control as sc

        return sc.take_screenshot()

    def _media_control(self, args: dict) -> str:
        import system_control as sc

        return sc.media(str(args.get("action", "")))

    def _play_music(self, args: dict) -> str:
        import system_control as sc

        return sc.play_youtube(str(args.get("query", "")))

    def _now_playing(self, args: dict) -> str:
        import system_control as sc

        return sc.now_playing()

    def _open_app(self, args: dict) -> str:
        import system_control as sc

        name = str(args.get("name", "")).strip()
        return sc.open_app(name) if name else "No app name given."

    def _close_app(self, args: dict) -> str:
        import system_control as sc

        name = str(args.get("name", "")).strip()
        return sc.close_app(name) if name else "No app name given."

    def _type_text(self, args: dict) -> str:
        import system_control as sc

        return sc.type_text(str(args.get("text", "")))

    def _press_keys(self, args: dict) -> str:
        import system_control as sc

        keys = str(args.get("keys", "")).strip()
        return sc.press_keys(keys) if keys else "No keys given."

    def _mouse_click(self, args: dict) -> str:
        import system_control as sc

        return sc.mouse_click(
            str(args.get("button", "left")), args.get("x"), args.get("y"),
            bool(args.get("double", False)))

    def _move_mouse(self, args: dict) -> str:
        import system_control as sc

        if "x" not in args or "y" not in args:
            return "Need x and y coordinates."
        return sc.move_mouse(args["x"], args["y"])

    def _mouse_scroll(self, args: dict) -> str:
        import system_control as sc

        return sc.mouse_scroll(int(args.get("amount", -3)))

    def _see_screen(self, args: dict) -> str:
        if self._vision is None or not getattr(self._vision, "available", False):
            return "Screen vision is unavailable."
        from PIL import ImageGrab

        question = str(args.get("question", "")).strip() \
            or "Describe what is on the screen, concisely."
        return self._vision.look(ImageGrab.grab(), question)

    # ---- vision-guided control ------------------------------------------
    def _locate_on_screen(self, description: str):
        """Capture the screen, locate `description`, return (x_px, y_px, shot)
        in screenshot pixels — or (None, None, reason)."""
        if self._vision is None or not getattr(self._vision, "available", False):
            return None, None, "Screen vision is unavailable."
        from PIL import ImageGrab

        if not description:
            return None, None, "What should I look for?"
        shot = ImageGrab.grab()
        loc = self._vision.locate(shot, description)
        if not loc:
            return None, None, f"I couldn't find '{description}' on the screen."
        x = round(loc["x"] * shot.size[0])
        y = round(loc["y"] * shot.size[1])
        return x, y, None

    def _click_element(self, args: dict) -> str:
        import system_control as sc

        desc = str(args.get("description", "")).strip()
        x, y, err = self._locate_on_screen(desc)
        if err:
            return err
        button = str(args.get("button", "left")).lower()
        double = bool(args.get("double", False))
        sc.mouse_click(button, x, y, double)
        kind = "Double-clicked" if double else f"{button.capitalize()}-clicked"
        return f"{kind} '{desc}' at {x}, {y}."

    def _type_into(self, args: dict) -> str:
        import time

        import system_control as sc

        desc = str(args.get("description", "")).strip()
        text = str(args.get("text", ""))
        x, y, err = self._locate_on_screen(desc)
        if err:
            return err
        sc.mouse_click("left", x, y, False)
        time.sleep(0.15)
        sc.type_text(text)
        if args.get("enter"):
            sc.press_keys("enter")
        return f"Typed into '{desc}' at {x}, {y}."

    def _find_on_screen(self, args: dict) -> str:
        desc = str(args.get("description", "")).strip()
        x, y, err = self._locate_on_screen(desc)
        if err:
            return err
        return f"'{desc}' is at about {x}, {y} on screen."

    # ---- camera ----------------------------------------------------------
    def _see_camera(self, args: dict) -> str:
        if self._vision is None or not getattr(self._vision, "available", False):
            return "Vision is unavailable, so I can't interpret the camera."
        from vision import capture_camera

        try:
            frame = capture_camera(int(args.get("camera", 0) or 0))
        except Exception as e:
            return f"Couldn't access the camera: {e}"
        question = str(args.get("question", "")).strip() \
            or "Describe what the camera sees, concisely."
        return self._vision.look(frame, question)

    def _take_photo(self, args: dict) -> str:
        from vision import capture_camera

        try:
            frame = capture_camera(int(args.get("camera", 0) or 0))
        except Exception as e:
            return f"Couldn't access the camera: {e}"
        os.makedirs("photos", exist_ok=True)
        name = "photo_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
        path = os.path.join("photos", name)
        try:
            frame.save(path)
        except Exception as e:
            return f"Couldn't save the photo: {e}"
        return f"Saved a photo to {path}."

    # ---- face recognition -----------------------------------------------
    def _grab_for_face(self, args: dict):
        if self._faces is None or not getattr(self._faces, "available", False):
            return None, "Face recognition is unavailable."
        from vision import capture_camera

        try:
            return capture_camera(int(args.get("camera", 0) or 0)), None
        except Exception as e:
            return None, f"Couldn't access the camera: {e}"

    def _recognize_face(self, args: dict) -> str:
        frame, err = self._grab_for_face(args)
        if err:
            return err
        results = self._faces.identify(frame)
        if not results:
            return "I don't see anyone at the camera."
        known = [r["name"] for r in results if r["name"] != "unknown"]
        if len(results) == 1:
            r = results[0]
            if r["name"] == "unknown":
                return "I see someone, but I don't recognize them."
            return f"I recognize {r['name']}."
        msg = f"I see {len(results)} people"
        if known:
            msg += ": " + ", ".join(known)
        unknown = len(results) - len(known)
        if unknown:
            msg += (f", and {unknown} I don't recognize" if known
                    else ", none of whom I recognize")
        return msg + "."

    def _enroll_face(self, args: dict) -> str:
        name = str(args.get("name", "")).strip()
        if not name:
            return "What name should I remember this face as?"
        frame, err = self._grab_for_face(args)
        if err:
            return err
        return self._faces.enroll(frame, name)

    def _register_user(self, args: dict) -> str:
        if self._register_user_fn is None:
            return ("I can't register new users right now — face/voice enrollment "
                    "isn't wired up.")
        name = str(args.get("name", "")).strip()
        role = str(args.get("authority", "user")).strip().lower() or "user"
        if not name:
            return "What's the new user's name?"
        try:
            return self._register_user_fn(name, role)
        except Exception as e:
            return f"Couldn't register {name}: {e}"

    def _open_face_window(self, args: dict) -> str:
        if self._faces is None or not getattr(self._faces, "available", False):
            return "Face recognition isn't available."
        import subprocess
        import sys
        import time

        here = os.path.dirname(os.path.abspath(__file__))
        script = os.path.join(here, "face_window.py")
        cam = str(int(args.get("camera", 0) or 0))
        log_path = os.path.join(here, "face_window.log")
        try:
            # Launch as a child process so the assistant keeps running; the
            # window owns its own GUI loop + camera until you close it (press q).
            # Send its output to a log so a crash is diagnosable instead of
            # silently vanishing while we falsely claim success.
            logf = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen([sys.executable, script, cam],
                                    stdout=logf, stderr=subprocess.STDOUT,
                                    cwd=here)
        except Exception as e:
            return f"Couldn't open the face window: {e}"

        # Give it a moment to fail fast (bad import, camera busy). If it's still
        # alive it's loading the model — don't claim success it didn't earn.
        time.sleep(1.2)
        if proc.poll() is not None:
            err = ""
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    lines = [ln for ln in f.read().splitlines() if ln.strip()]
                err = lines[-1] if lines else ""
            except Exception:
                pass
            return ("The face window opened but closed right away"
                    + (f": {err}" if err else
                       " (check the camera isn't in use by another app)."))
        return ("Opening the face recognition window — the camera shows right "
                "away and names appear once the face model finishes loading. "
                "Press q to close it.")

    def _open_website(self, args: dict) -> str:
        import system_control as sc

        url = str(args.get("url", "")).strip()
        return sc.open_website(url) if url else "No website given."

    def _create_github_repo(self, args: dict) -> str:
        import re
        import subprocess

        name = str(args.get("name", "")).strip()
        if not name:
            return "What should the repository be called?"
        if not re.match(r"^[A-Za-z0-9._-]+$", name):
            return f"Invalid repository name: {name!r}."
        private = bool(args.get("private", True))
        desc = str(args.get("description", "")).strip()
        cmd = ["gh", "repo", "create", name,
               "--private" if private else "--public"]
        if desc:
            cmd += ["--description", desc]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
        except FileNotFoundError:
            return "The GitHub CLI (gh) isn't installed."
        except Exception as e:
            return f"Couldn't create the repository: {e}"
        out = ((res.stdout or "") + (res.stderr or "")).strip()
        if res.returncode == 0:
            url = next((ln.strip() for ln in out.splitlines() if "github.com/" in ln), "")
            vis = "private" if private else "public"
            return f"Created the {vis} GitHub repository '{name}'." + (f" {url}" if url else "")
        low = out.lower()
        if "auth" in low and any(k in low for k in ("login", "logged", "token")):
            return "The GitHub CLI isn't authenticated. Run 'gh auth login' first."
        if "already exists" in low:
            return f"A repository named '{name}' already exists."
        return f"Couldn't create the repository: {out.splitlines()[-1] if out else 'unknown error'}"

    # ---- coding: files + command execution ------------------------------
    @staticmethod
    def _resolve_path(path: str) -> str:
        """Resolve a path to an absolute one, rebasing any Desktop/Documents/
        Downloads/home reference onto the user's REAL folder. This fixes the
        model guessing a username or a Unix-style /Users/... path (which on
        Windows would otherwise land on the wrong drive)."""
        raw = str(path or "").strip()
        if not raw:
            return ""
        folders = _known_folders()
        parts = [p for p in raw.replace("\\", "/").split("/") if p not in ("", ".")]
        for i, comp in enumerate(parts):
            low = comp.lower()
            if low in ("desktop", "documents", "downloads"):
                return os.path.abspath(os.path.join(folders[low], *parts[i + 1:]))
        if parts and (parts[0] == "~" or parts[0].lower() == "home"):
            return os.path.abspath(os.path.join(folders["home"], *parts[1:]))
        return os.path.abspath(os.path.expanduser(raw))

    def folders_note(self) -> str:
        """A system-prompt line giving the model the user's real folder paths."""
        f = _known_folders()
        return ("\n\nThis machine's real folders — use these EXACT paths for file "
                "operations and NEVER guess a username or use Unix-style "
                f"/Users/... paths: home={f['home']}; Desktop={f['desktop']}; "
                f"Documents={f['documents']}; Downloads={f['downloads']}.")

    def _write_file(self, args: dict) -> str:
        path = self._resolve_path(args.get("path", ""))
        if not path:
            return "No file path given."
        content = args.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            existed = os.path.exists(path)
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(content)
        except Exception as e:
            return f"Couldn't write {path}: {e}"
        lines = content.count("\n") + 1 if content else 0
        verb = "Updated" if existed else "Wrote"
        return f"{verb} {path} ({lines} lines)."

    def _read_file(self, args: dict) -> str:
        path = self._resolve_path(args.get("path", ""))
        if not os.path.isfile(path):
            return f"No such file: {path}"
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except Exception as e:
            return f"Couldn't read {path}: {e}"
        if len(text) > 6000:  # protect the model's context window
            text = text[:6000] + "\n... (truncated)"
        return f"{path}:\n{text}"

    def _edit_file(self, args: dict) -> str:
        path = self._resolve_path(args.get("path", ""))
        if not os.path.isfile(path):
            return f"No such file: {path}"
        find = args.get("find", "")
        if not isinstance(find, str) or find == "":
            return "Provide the exact 'find' text to replace."
        replace = args.get("replace", "")
        replace = replace if isinstance(replace, str) else str(replace)
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            return f"Couldn't read {path}: {e}"
        if find not in text:
            return f"Couldn't find that text in {path}."
        count = -1 if args.get("all") else 1
        new = text.replace(find, replace, count)
        try:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(new)
        except Exception as e:
            return f"Couldn't write {path}: {e}"
        n = text.count(find) if count == -1 else 1
        return f"Edited {path} ({n} replacement{'s' if n != 1 else ''})."

    def _list_dir(self, args: dict) -> str:
        path = self._resolve_path(args.get("path", ".") or ".")
        if not os.path.isdir(path):
            return f"No such directory: {path}"
        try:
            entries = sorted(os.listdir(path))
        except Exception as e:
            return f"Couldn't list {path}: {e}"
        if not entries:
            return f"{path} is empty."
        marked = [e + ("/" if os.path.isdir(os.path.join(path, e)) else "")
                  for e in entries[:100]]
        return f"{path}:\n" + ", ".join(marked)

    def _run_command(self, args: dict) -> str:
        import subprocess

        command = str(args.get("command", "")).strip()
        if not command:
            return "No command given."
        cwd = args.get("cwd")
        cwd = self._resolve_path(cwd) if cwd else None
        try:
            res = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=self._command_timeout, cwd=cwd)
        except subprocess.TimeoutExpired:
            return f"Command timed out after {self._command_timeout}s."
        except Exception as e:
            return f"Couldn't run the command: {e}"
        out = ((res.stdout or "") + (res.stderr or "")).strip()
        if len(out) > 4000:
            out = out[:4000] + "\n... (truncated)"
        status = "ok" if res.returncode == 0 else f"exit {res.returncode}"
        return f"[{status}]\n{out}" if out else f"[{status}] (no output)"

    def _check_syntax(self, args: dict) -> str:
        path = self._resolve_path(args.get("path", ""))
        if not os.path.isfile(path):
            return f"No such file: {path}"
        if not path.endswith(".py"):
            return "check_syntax currently supports Python (.py) files only."
        try:
            with open(path, "r", encoding="utf-8") as f:
                src = f.read()
        except Exception as e:
            return f"Couldn't read {path}: {e}"
        try:
            compile(src, path, "exec")  # parses only; does NOT run the code
        except SyntaxError as e:
            loc = f"{path}:{e.lineno}" + (f":{e.offset}" if e.offset else "")
            line = (e.text or "").rstrip("\n")
            return (f"SyntaxError at {loc}: {e.msg}"
                    + (f"\n    {line.strip()}" if line.strip() else ""))
        except Exception as e:
            return f"Couldn't check {path}: {e}"
        return f"No syntax errors in {path}."

    def _debug_python(self, args: dict) -> str:
        import subprocess

        path = self._resolve_path(args.get("path", ""))
        if not os.path.isfile(path):
            return f"No such file: {path}"
        extra = args.get("args") or []
        if isinstance(extra, str):
            extra = extra.split()
        cmd = [sys.executable, path, *[str(a) for a in extra]]
        try:
            res = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self._command_timeout,
                cwd=os.path.dirname(path) or None)
        except subprocess.TimeoutExpired:
            return (f"Script timed out after {self._command_timeout}s "
                    "(possible infinite loop).")
        except Exception as e:
            return f"Couldn't run {path}: {e}"
        out = (res.stdout or "").strip()
        err = (res.stderr or "").strip()
        if res.returncode == 0:
            body = out if out else "(no output)"
            if len(body) > 4000:
                body = body[:4000] + "\n... (truncated)"
            return f"[ran cleanly] {body}"
        msg = f"[crashed] {self._summarize_traceback(err)}"
        if out:
            tail = out[-1000:]
            msg += f"\n--- stdout before crash ---\n{tail}"
        return msg

    @staticmethod
    def _summarize_traceback(stderr: str) -> str:
        """Turn a Python traceback into error + deepest file:line + source line."""
        if not stderr:
            return "exited with an error (no traceback)."
        lines = stderr.splitlines()
        exc = next((ln.strip() for ln in reversed(lines) if ln.strip()), "")
        frame_re = re.compile(r'File "(.+?)", line (\d+)(?:, in (.+))?')
        frames = [m for ln in lines for m in (frame_re.search(ln),) if m]
        if not frames:
            return exc or stderr[-500:]
        last = frames[-1]
        fpath, lineno, func = last.group(1), int(last.group(2)), last.group(3) or "?"
        src = ""
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                flines = f.readlines()
            if 1 <= lineno <= len(flines):
                src = flines[lineno - 1].strip()
        except Exception:
            pass
        summary = f"{exc}\n  at {fpath}:{lineno} in {func}"
        if src:
            summary += f"\n    {lineno} | {src}"
        return summary

    def _system_power(self, args: dict) -> str:
        import system_control as sc

        return sc.power(str(args.get("action", "")), self._allow_power_off)

    def _run_web_search(self, query: str) -> str:
        # Try the preferred backend, then fall back to any others that are
        # available, so a bad key or a flaky endpoint never dead-ends a search.
        handlers = {
            "tavily_rest": (self._tavily_rest_search, bool(self._tavily_key)),
            "tavily_mcp": (self._tavily_mcp_search, bool(self._tavily_url)),
            "duckduckgo": (self._duckduckgo_search, True),
        }
        order = ["tavily_rest", "tavily_mcp", "duckduckgo"]
        if self._backend != "auto" and self._backend in handlers:
            order = [self._backend] + [b for b in order if b != self._backend]

        last_error = None
        for name in order:
            handler, available = handlers[name]
            if not available:
                continue
            try:
                return handler(query)
            except Exception as e:  # 401, timeout, etc. — try the next backend
                last_error = f"{name}: {e}"
        return f"Web search is unavailable right now ({last_error})."

    def _tavily_rest_search(self, query: str) -> str:
        """Search via the Tavily REST API; returns its synthesized answer."""
        body = json.dumps(
            {
                "query": query,
                "include_answer": "basic",
                "max_results": 3,
                "search_depth": "basic",
            }
        ).encode("utf-8")
        req = Request(
            "https://api.tavily.com/search",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._tavily_key}",
            },
        )
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data.get("answer"):
            return str(data["answer"])
        return self._summarize_results(data.get("results"), query)

    def _tavily_mcp_search(self, query: str) -> str:
        """Search via the Tavily MCP server. Returns a short spoken-friendly answer."""
        import asyncio

        async def run() -> str:
            # Imported lazily so Tavily is optional (no hard dep on `mcp`).
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client

            async with streamablehttp_client(self._tavily_url) as (read, write, *_):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    res = await session.call_tool(
                        "tavily_search", {"query": query, "max_results": 3}
                    )
                    for chunk in res.content:
                        text = getattr(chunk, "text", None)
                        if text:
                            return text
            return ""

        raw = asyncio.run(run())
        if not raw:
            return f"No results for '{query}'."
        data = json.loads(raw)
        if data.get("answer"):
            return str(data["answer"])
        # No synthesized answer from MCP: hand the LLM the top snippets so it can
        # summarize them into one spoken sentence (see the post-tool reprompt).
        return self._summarize_results(data.get("results"), query)

    @staticmethod
    def _summarize_results(results: Optional[list], query: str) -> str:
        """Join the top result snippets for the LLM to condense into one answer."""
        snippets = []
        for result in (results or [])[:3]:
            content = " ".join((result.get("content") or "").split())
            if content:
                snippets.append(content)
        if snippets:
            return " ".join(snippets)[:800]
        return f"No results for '{query}'."

    def _duckduckgo_search(self, query: str) -> str:
        """Fallback search: DuckDuckGo Instant Answer API (no key needed)."""
        url = (
            "https://api.duckduckgo.com/?q="
            + quote_plus(query)
            + "&format=json&no_html=1&skip_disambig=1"
        )
        req = Request(url, headers={"User-Agent": "Atlas/1.0"})
        with urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data.get("AbstractText"):
            return data["AbstractText"]
        if data.get("Answer"):
            return str(data["Answer"])
        for topic in data.get("RelatedTopics", []):
            if isinstance(topic, dict) and topic.get("Text"):
                return topic["Text"]
        return f"No quick answer found for '{query}'."

    def cancel_all(self) -> None:
        """Cancel pending timers (call on shutdown)."""
        for t in self._timers:
            t.cancel()
