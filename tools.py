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


def _extract_json_object(text: str) -> Optional[str]:
    """Return the first balanced {...} substring, or None."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


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
        memory=None,
        store=None,
        voiceprint_path: str = "",
        enable_coding: bool = True,
        command_timeout: int = 60,
    ):
        # Called when a timer elapses; main.py wires this to speak the alert.
        self._on_timer_fire = on_timer_fire or (lambda msg: print(f"\n{msg}"))
        self._timers: list[threading.Timer] = []
        self.cache = cache  # optional web-search cache (see cache.py)
        self._allow_power_off = allow_power_off
        self._vision = vision  # optional screen vision (see vision.py)
        self._faces = faces    # optional face recognition (see face_id.py)
        self._memory = memory  # optional semantic memory (see memory.py)
        self._store = store    # optional durable state (see state.py)
        self._voiceprint_path = voiceprint_path
        self._command_timeout = command_timeout

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
            "reset_all": (
                self._reset_all,
                "Factory-reset Atlas: erase ALL memory and conversation "
                "history, the cache, and the enrolled voice and face. "
                "Irreversible. Only when the user clearly asks to reset "
                "everything / reset yourself / wipe everything.",
                "{}",
            ),
        }

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
                    "Close/quit a running application on this computer.",
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

        # Coding: read/write/edit files and run commands. write_file overwriting
        # an existing file and run_command are gated by spoken confirmation in
        # the agent (see agents.Orchestrator).
        if enable_coding:
            self._registry.update({
                "write_file": (
                    self._write_file,
                    "Create or overwrite a text/code file with the given "
                    "content (parent folders are created). Use this to write "
                    "code to disk.",
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

    # ---- detection + dispatch -------------------------------------------
    def parse(self, text: str) -> Optional[dict]:
        """Return {"name", "arguments"} if `text` is a tool call, else None."""
        blob = _extract_json_object(text)
        if not blob:
            return None
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            return None
        if (
            isinstance(obj, dict)
            and isinstance(obj.get("tool"), str)
            and obj["tool"] in self._registry
        ):
            args = obj.get("arguments")
            return {"name": obj["tool"], "arguments": args if isinstance(args, dict) else {}}
        return None

    def execute(self, call: dict) -> str:
        """Run a parsed tool call and return a short text result."""
        handler = self._registry[call["name"]][0]
        try:
            return handler(call.get("arguments", {}))
        except Exception as e:  # never let a tool crash the loop
            return f"Tool '{call['name']}' failed: {e}"

    # ---- tool implementations -------------------------------------------
    def _get_time(self, args: dict) -> str:
        return "The current time is " + datetime.datetime.now().strftime("%I:%M %p").lstrip("0")

    def _get_date(self, args: dict) -> str:
        return "Today is " + datetime.datetime.now().strftime("%A, %B %d, %Y")

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
        if self._faces is not None and getattr(self._faces, "available", False):
            done.append("face enrollments removed" if self._faces.reset()
                        else "faces could not be removed")
        if not done:
            return "There was nothing to reset."
        return ("Done — I reset everything: " + "; ".join(done) + ". "
                "Re-enroll your voice with enroll.py and your face with "
                "enroll_face.py before those features work again.")

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
        return os.path.abspath(os.path.expanduser(str(path).strip()))

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
