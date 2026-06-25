"""Stage 12: system control (Windows) — volume, media, apps, websites, power.

Dependency-light wrappers around Windows APIs so Atlas can act on the machine.
Each function returns a short status string for the assistant to speak.

Destructive power actions (shutdown/restart) are gated by the caller; lock/sleep
are reversible and always allowed.
"""

import ctypes
import datetime
import os
import re
import subprocess
import webbrowser
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

# Virtual-key codes for media/volume keys (tapped via keybd_event).
_MEDIA_VK = {
    "play_pause": 0xB3,
    "next": 0xB0,
    "previous": 0xB1,
    "stop": 0xB2,
}
_KEYEVENTF_KEYUP = 0x0002

# Friendly names -> launch target (exe on PATH, or shell/UWP URI).
_APPS = {
    "notepad": "notepad", "calculator": "calc", "calc": "calc",
    "paint": "mspaint", "explorer": "explorer", "file explorer": "explorer",
    "files": "explorer", "task manager": "taskmgr", "cmd": "cmd",
    "command prompt": "cmd", "powershell": "powershell",
    "settings": "ms-settings:", "camera": "microsoft.windows.camera:",
    "chrome": "chrome", "edge": "msedge", "browser": "msedge",
    "spotify": "spotify", "vs code": "code", "code": "code",
}

_SAFE_NAME = re.compile(r"^[\w .:\-\\/]+$")  # block shell metacharacters


def _tap_key(vk: int) -> None:
    ctypes.windll.user32.keybd_event(vk, 0, 0, 0)
    ctypes.windll.user32.keybd_event(vk, 0, _KEYEVENTF_KEYUP, 0)


# --- SendInput plumbing for typing arbitrary Unicode text into the focused window ---
from ctypes import wintypes  # noqa: E402

_ULONG_PTR = ctypes.POINTER(ctypes.c_ulong)
_KEYEVENTF_UNICODE = 0x0004
_INPUT_KEYBOARD = 1
_VK_RETURN = 0x0D


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", _ULONG_PTR)]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", _ULONG_PTR)]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT), ("hi", _HARDWAREINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


def _key_input(scan: int, vk: int, unicode: bool, keyup: bool) -> _INPUT:
    flags = (_KEYEVENTF_UNICODE if unicode else 0) | (_KEYEVENTF_KEYUP if keyup else 0)
    return _INPUT(type=_INPUT_KEYBOARD,
                  u=_INPUTUNION(ki=_KEYBDINPUT(vk, scan, flags, 0, None)))


def _send(*inputs: _INPUT) -> None:
    n = len(inputs)
    ctypes.windll.user32.SendInput(n, (_INPUT * n)(*inputs), ctypes.sizeof(_INPUT))


def type_text(text: str) -> str:
    """Type text into the currently-focused window (keyboard simulation)."""
    import time

    if not text:
        return "Nothing to type."
    time.sleep(0.4)  # let the target window take focus/settle
    for ch in text:
        if ch in ("\n", "\r"):
            _send(_key_input(0, _VK_RETURN, False, False),
                  _key_input(0, _VK_RETURN, False, True))
        else:
            code = ord(ch)
            _send(_key_input(code, 0, True, False), _key_input(code, 0, True, True))
        time.sleep(0.004)
    preview = text if len(text) <= 50 else text[:50] + "…"
    return f'Typed "{preview}".'


# Named virtual-key codes for hotkeys/keypresses (single a-z/0-9 derive from ord).
_VK_NAMES = {
    "enter": 0x0D, "return": 0x0D, "tab": 0x09, "esc": 0x1B, "escape": 0x1B,
    "space": 0x20, "backspace": 0x08, "delete": 0x2E, "del": 0x2E, "insert": 0x2D,
    "home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "ctrl": 0x11, "control": 0x11, "alt": 0x12, "shift": 0x10,
    "win": 0x5B, "windows": 0x5B, "cmd": 0x5B, "capslock": 0x14, "printscreen": 0x2C,
    **{f"f{i}": 0x6F + i for i in range(1, 13)},
}


def _vk_for(name: str):
    name = name.strip().lower()
    if len(name) == 1 and name.isalnum():
        return ord(name.upper())
    return _VK_NAMES.get(name)


def press_keys(combo: str) -> str:
    """Press a key or key combination, e.g. 'enter', 'ctrl+s', 'alt+tab', 'win+d'."""
    parts = [p for p in combo.replace(" ", "").split("+") if p]
    if not parts:
        return "No keys given."
    vks = []
    for p in parts:
        vk = _vk_for(p)
        if vk is None:
            return f"Unknown key: {p}"
        vks.append(vk)
    # Press all down (modifiers first), release in reverse order.
    inputs = [_key_input(0, vk, False, False) for vk in vks]
    inputs += [_key_input(0, vk, False, True) for vk in reversed(vks)]
    _send(*inputs)
    return f"Pressed {combo}."


# mouse_event flags
_MOUSE = {
    "left": (0x0002, 0x0004), "right": (0x0008, 0x0010), "middle": (0x0020, 0x0040),
}
_MOUSEEVENTF_WHEEL = 0x0800


def move_mouse(x: int, y: int) -> str:
    ctypes.windll.user32.SetCursorPos(int(x), int(y))
    return f"Moved the cursor to {int(x)}, {int(y)}."


def mouse_click(button: str = "left", x=None, y=None, double: bool = False) -> str:
    import time

    button = (button or "left").lower()
    down_up = _MOUSE.get(button)
    if down_up is None:
        return f"Unknown mouse button: {button}"
    if x is not None and y is not None:
        ctypes.windll.user32.SetCursorPos(int(x), int(y))
        time.sleep(0.05)
    down, up = down_up
    for _ in range(2 if double else 1):
        ctypes.windll.user32.mouse_event(down, 0, 0, 0, 0)
        ctypes.windll.user32.mouse_event(up, 0, 0, 0, 0)
        if double:
            time.sleep(0.05)
    where = f" at {int(x)}, {int(y)}" if x is not None and y is not None else ""
    return f"{'Double-' if double else ''}{button} click{where}."


def mouse_scroll(amount: int = -3) -> str:
    """Scroll the wheel; positive = up, negative = down (in notches)."""
    ctypes.windll.user32.mouse_event(_MOUSEEVENTF_WHEEL, 0, 0, int(amount) * 120, 0)
    return f"Scrolled {'up' if amount > 0 else 'down'}."


def _endpoint_volume():
    """Return the Core Audio master-volume interface (via pycaw)."""
    from pycaw.pycaw import AudioUtilities

    speakers = AudioUtilities.GetSpeakers()
    # Newer pycaw exposes the (already-cast) interface as .EndpointVolume.
    if hasattr(speakers, "EndpointVolume"):
        return speakers.EndpointVolume
    # Older pycaw: activate the COM interface manually.
    from ctypes import POINTER, cast

    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import IAudioEndpointVolume

    interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(interface, POINTER(IAudioEndpointVolume))


def set_volume(percent: int) -> str:
    p = max(0, min(100, int(percent)))
    _endpoint_volume().SetMasterVolumeLevelScalar(p / 100.0, None)
    return f"Volume set to {p} percent."


def set_mute(mute: bool) -> str:
    _endpoint_volume().SetMute(bool(mute), None)
    return "Muted." if mute else "Unmuted."


# SMTC (System Media Transport Controls) session methods per action. These act
# on the actual current media session — more reliable than global media keys.
_SMTC_METHODS = {
    "play_pause": "try_toggle_play_pause_async",
    "play": "try_play_async",
    "pause": "try_pause_async",
    "next": "try_skip_next_async",
    "previous": "try_skip_previous_async",
    "stop": "try_stop_async",
}


async def _current_session():
    from winsdk.windows.media.control import (
        GlobalSystemMediaTransportControlsSessionManager as _Mgr,
    )

    manager = await _Mgr.request_async()
    return manager.get_current_session()


def media(action: str) -> str:
    """Control playback via SMTC (the real session); fall back to media keys."""
    import asyncio

    method = _SMTC_METHODS.get(action)
    if method:
        try:
            async def _go():
                session = await _current_session()
                if session is None:
                    return False
                return await getattr(session, method)()

            if asyncio.run(_go()):
                return f"Done: {action.replace('_', ' ')}."
        except Exception:
            pass  # fall through to media keys

    # Fallback: global media keys (play/pause share one key).
    vk = _MEDIA_VK.get(action) or (
        _MEDIA_VK["play_pause"] if action in ("play", "pause") else None
    )
    if vk is None:
        return f"Unknown media action '{action}'."
    _tap_key(vk)
    return f"Done: {action.replace('_', ' ')}."


def now_playing() -> str:
    """Report the currently playing track via SMTC."""
    import asyncio

    async def _go():
        session = await _current_session()
        if session is None:
            return None
        info = await session.try_get_media_properties_async()
        return (info.title or "").strip(), (info.artist or "").strip()

    try:
        result = asyncio.run(_go())
    except Exception as e:
        return f"Couldn't read media info: {e}"
    if not result or not result[0]:
        return "Nothing is playing right now."
    title, artist = result
    return f"Now playing: {title}" + (f" by {artist}." if artist else ".")


def set_brightness(percent: int) -> str:
    import screen_brightness_control as sbc

    p = max(0, min(100, int(percent)))
    sbc.set_brightness(p)  # all detected displays
    return f"Brightness set to {p} percent."


def take_screenshot(directory: str = "screenshots") -> str:
    from PIL import ImageGrab

    os.makedirs(directory, exist_ok=True)
    name = "screenshot_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
    path = os.path.join(directory, name)
    ImageGrab.grab().save(path)
    return f"Saved a screenshot to {path}."


def set_app_volume(app: str, percent=None, mute=None) -> str:
    """Set volume/mute for a running app's audio session, matched by name."""
    from pycaw.pycaw import AudioUtilities

    target = app.strip().lower()
    matched = None
    for session in AudioUtilities.GetAllSessions():
        proc = session.Process
        if proc and target in proc.name().lower():
            vol = session.SimpleAudioVolume
            if mute is not None:
                vol.SetMute(bool(mute), None)
            if percent is not None:
                vol.SetMasterVolume(max(0, min(100, int(percent))) / 100.0, None)
            matched = proc.name()
            break
    if matched is None:
        return f"No running app matching '{app}' is playing audio."
    if mute is not None:
        return f"{'Muted' if mute else 'Unmuted'} {matched}."
    return f"Set {matched} volume to {int(percent)} percent."


def open_app(name: str) -> str:
    key = name.strip().lower()
    target = _APPS.get(key, name.strip())
    if not _SAFE_NAME.match(target):
        return f"Refusing to open an unsafe name: {name!r}."
    try:
        if target.endswith(":") or target.startswith("microsoft."):
            os.startfile(target)  # protocol / UWP app  # noqa: S606
        else:
            subprocess.Popen(target, shell=True)  # resolves PATH / App Paths
        return f"Opening {key}."
    except Exception as e:
        return f"Couldn't open {key}: {e}"


# Friendly name -> Windows image name for closing. Distinct from _APPS (launch
# targets) since UWP/launch URIs don't map to process images.
_CLOSE_NAMES = {
    "notepad": "notepad.exe", "calculator": "Calculator.exe", "calc": "Calculator.exe",
    "paint": "mspaint.exe", "chrome": "chrome.exe", "edge": "msedge.exe",
    "browser": "msedge.exe", "spotify": "Spotify.exe", "vs code": "Code.exe",
    "code": "Code.exe", "task manager": "Taskmgr.exe", "cmd": "cmd.exe",
    "powershell": "powershell.exe",
}
# Never kill these — would break the desktop/session or Atlas itself.
_CLOSE_DENYLIST = {"explorer.exe", "svchost.exe", "winlogon.exe", "csrss.exe",
                   "system.exe", "python.exe", "pythonw.exe"}


def close_app(name: str) -> str:
    """Close a running application by name (graceful taskkill by image name)."""
    key = name.strip().lower()
    img = _CLOSE_NAMES.get(key, name.strip())
    if not img.lower().endswith(".exe"):
        img += ".exe"
    if not _SAFE_NAME.match(img):
        return f"Refusing to close an unsafe name: {name!r}."
    if img.lower() in _CLOSE_DENYLIST:
        return f"I won't close {img} — it's a critical system process."
    result = subprocess.run(["taskkill", "/IM", img], capture_output=True, text=True)
    if result.returncode == 0:
        return f"Closed {key}."
    return f"{key} doesn't appear to be running."


def _open_url(url: str) -> None:
    """Open a URL in Microsoft Edge (falls back to the default browser)."""
    try:
        os.startfile(f"microsoft-edge:{url}")  # protocol forces Edge
    except Exception:
        webbrowser.open(url)


def _youtube_top_video(query: str):
    """Return the first videoId for a YouTube search query, or None."""
    search_url = "https://www.youtube.com/results?search_query=" + quote_plus(query)
    try:
        req = Request(search_url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"})
        html = urlopen(req, timeout=10).read().decode("utf-8", "ignore")
        m = re.search(r'"videoId":"([A-Za-z0-9_-]{11})"', html)
        return m.group(1) if m else None
    except Exception:
        return None


def _force_media_play() -> None:
    """After a watch page opens, wait for its media session and send a 'play'.

    Browsers block autoplay-with-sound, so a freshly opened YouTube video loads
    paused. A system media 'play' command (SMTC try_play) counts as user intent
    and starts it with sound. We give the page a moment to load + register, then
    issue play.
    """
    import asyncio
    import time

    async def _go():
        await asyncio.sleep(3.0)  # let the watch page load + register a session
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            session = await _current_session()
            if session is not None:
                try:
                    await session.try_play_async()
                except Exception:
                    pass
                return
            await asyncio.sleep(0.5)

    try:
        asyncio.run(_go())
    except Exception:
        pass


def play_youtube(query: str) -> str:
    """Search YouTube, open the top result's watch page, and force playback."""
    query = query.strip()
    if not query:
        return "What should I play?"
    vid = _youtube_top_video(query)
    if vid:
        _open_url(f"https://www.youtube.com/watch?v={vid}")
        _force_media_play()  # bypass the browser's autoplay block
        return f"Playing {query} on YouTube."
    # Fallback: open the search results page.
    _open_url("https://www.youtube.com/results?search_query=" + quote_plus(query))
    return f"Opened a YouTube search for {query}."


def open_website(url: str) -> str:
    url = url.strip()
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url
    _open_url(url)
    return f"Opening {url}."


def lock() -> str:
    ctypes.windll.user32.LockWorkStation()
    return "Locked the screen."


def power(action: str, allow_power_off: bool = False) -> str:
    action = action.strip().lower()
    if action == "lock":
        return lock()
    if action == "sleep":
        subprocess.Popen(
            "rundll32.exe powrprof.dll,SetSuspendState 0,1,0", shell=True
        )
        return "Going to sleep."
    if action == "cancel":
        subprocess.run(["shutdown", "/a"], capture_output=True)
        return "Cancelled the pending shutdown."
    if action in ("shutdown", "restart"):
        if not allow_power_off:
            return ("Shutdown and restart are disabled. Enable "
                    "allow_power_off in the config to use them.")
        flag = "/s" if action == "shutdown" else "/r"
        subprocess.Popen(["shutdown", flag, "/t", "15"])
        return (f"{action.capitalize()} in 15 seconds. "
                "Say 'cancel shutdown' to abort.")
    return f"Unknown power action '{action}'."
