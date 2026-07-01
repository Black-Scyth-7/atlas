"""Startup identity gate: password + face + voice registration and login.

On the first ever run, Atlas registers the owner's face (webcam, InsightFace),
voice (voiceprint, SpeechBrain), and a typed password (stored as a salted
PBKDF2-SHA256 hash). On every later run it requires all three to match before
the assistant starts (password first, then face, then voice).

NOTE: the PASSWORD is a real secret (a salted hash). The face and voice checks
are PERSONALIZATION, not security — both are spoofable (a photo of the owner, a
recording of their voice). If you get locked out, set
`AuthConfig.require_identity = False` in config.py, or delete the password file
to reset just the password.

The functions take their dependencies explicitly (verifier, face recognizer, the
shared mic stream, a `speak` callback) so main.py wires them to the already-
loaded components.
"""

from __future__ import annotations

import getpass
import hashlib
import hmac
import json
import os
import re
import secrets

import audio_input

_PBKDF2_ITERATIONS = 200_000
_VOICEPRINTS_DIR = "voiceprints"   # per-user voiceprints: voiceprints/<name>.npy

# Spoken prompts to read back during voice enrollment (content is irrelevant to
# the text-independent ECAPA model; they just give the user something to say).
_PHRASES = [
    "The quick brown fox jumps over the lazy dog.",
    "I would like to check the weather and my schedule today.",
    "Please set a timer for ten minutes from now.",
    "Tell me something interesting about the solar system.",
    "Atlas, this is my voice for verification.",
]


# ---- user registry: name + authority (role); exactly one admin ----------
def _load_users(auth) -> dict:
    """Map of registered name -> {"role": ...}. Empty if nothing registered."""
    try:
        with open(auth.users_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_users(auth, users: dict) -> None:
    with open(auth.users_path, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def admin_name(auth) -> str | None:
    """The single admin's name, or None if no admin is registered yet."""
    for name, rec in _load_users(auth).items():
        if rec.get("role") == "admin":
            return name
    return None


def _first_user(auth) -> str | None:
    return next(iter(_load_users(auth)), None)


def register(auth, name: str, role: str) -> tuple[bool, str]:
    """Add/update a registered person. Enforces a single admin.

    Returns (ok, info): on success info is the stored role; if the requested
    role was 'admin' but another admin exists, returns (False, <admin name>).
    """
    role = (role or "user").strip().lower()
    if role not in tuple(auth.authorities):
        role = "user"
    existing = admin_name(auth)
    if role == "admin" and existing and existing != name:
        return False, existing
    users = _load_users(auth)
    users[name] = {"role": role}
    _save_users(auth, users)
    return True, role


def _user_vp_path(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "user"
    return os.path.join(_VOICEPRINTS_DIR, f"{safe}.npy")


def _load_user_voiceprint(verifier, cfg, name: str):
    """This user's voiceprint, falling back to the legacy single-owner file."""
    p = _user_vp_path(name)
    try:
        if os.path.exists(p):
            return verifier.load_voiceprint(p)
        if os.path.exists(cfg.voiceprint_path):
            return verifier.load_voiceprint()
    except Exception:
        pass
    return None


def register_new_user(cfg, auth, verifier, faces, stream, speak,
                      name: str, role: str) -> str:
    """Register an ADDITIONAL person at runtime (admin action): capture their
    face + voice and store their authority. Enforces the single-admin rule."""
    name = (name or "").strip()
    if not name:
        return "I need a name to register someone."
    requested = (role or "user").strip().lower()
    existing = admin_name(auth)
    if requested == "admin" and existing and existing != name:
        speak(f"There's already an admin, {existing}. I'll register {name} as a user.")
        requested = "user"

    # --- Face ---
    if faces is not None and getattr(faces, "available", False):
        from vision import capture_camera
        speak(f"{name}, please look at the camera.")
        got, attempts = 0, 0
        while got < auth.face_shots and attempts < auth.face_shots * 3:
            attempts += 1
            try:
                msg = faces.enroll(capture_camera(0), name)
            except Exception as e:
                msg = f"camera error: {e}"
            if msg.startswith("Saved"):
                got += 1
                print(f"  face {got}/{auth.face_shots} captured.")
            else:
                speak("I couldn't see the face clearly. Hold still and look at the camera.")
        if got:
            speak(f"{name}'s face is registered.")

    # --- Voice (saved to a per-user voiceprint) ---
    if verifier is not None:
        print(f"\n=== Voice registration for {name} ===")
        speak(f"{name}, please repeat each phrase after I say it.")
        samples = []
        for i in range(auth.voice_samples):
            phrase = _PHRASES[i % len(_PHRASES)]
            print(f"  [{i + 1}/{auth.voice_samples}] Say:  \"{phrase}\"")
            speak(f"Say: {phrase}")
            clip = audio_input.record_fixed(stream, auth.voice_seconds, cfg)
            if clip.size:
                samples.append(clip)
        if samples:
            os.makedirs(_VOICEPRINTS_DIR, exist_ok=True)
            verifier.save_voiceprint(verifier.build_voiceprint(samples),
                                     _user_vp_path(name))
            speak(f"{name}'s voice is registered.")

    ok, info = register(auth, name, requested)
    if not ok:
        register(auth, name, "user")
        info = "user"
    speak(f"Registered {name} as {info}.")
    return f"Registered {name} as {info}."


def _prompt_identity(auth, speak) -> tuple[str, str]:
    """Ask (typed, in the terminal) for the new user's name and authority.

    Only one admin is allowed: if an admin already exists, 'admin' is shown as
    unavailable and can't be picked.
    """
    speak("Let's register you. Please type your name in the terminal.")
    print("\n=== Registration ===")
    name = ""
    while not name:
        name = input("  Your name: ").strip()
        if not name:
            print("  Name can't be empty.")

    existing = admin_name(auth)
    levels = list(auth.authorities)
    print("\n  Authority level:")
    for i, level in enumerate(levels, 1):
        taken = (level == "admin" and existing and existing != name)
        note = (f"  (already assigned to {existing} — unavailable)"
                if taken else "")
        print(f"    {i}) {level}{note}")
    speak("Now choose your authority level.")

    role = ""
    while not role:
        choice = input(f"  Choose 1-{len(levels)}: ").strip()
        if not choice.isdigit() or not (1 <= int(choice) <= len(levels)):
            print(f"  Please type a number 1 to {len(levels)}.")
            continue
        level = levels[int(choice) - 1]
        if level == "admin" and existing and existing != name:
            print("  Only one admin is allowed and it's already taken — "
                  "pick another level.")
            continue
        role = level
    return name, role


# ---- password (typed; stored only as a salted PBKDF2 hash) ---------------
def _hash_password(password: str, salt: bytes | None = None,
                   iterations: int = _PBKDF2_ITERATIONS) -> dict:
    if salt is None:
        salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return {"salt": salt.hex(), "hash": dk.hex(), "iter": iterations}


def password_is_set(auth) -> bool:
    return bool(auth.require_password) and os.path.exists(auth.password_path)


def set_password(auth) -> bool:
    """Prompt (hidden) for a new password twice and store its salted hash."""
    print("Set a startup password (typed in this terminal; input is hidden).")
    for _ in range(3):
        pw = getpass.getpass("  New password: ")
        if not pw:
            print("  Password can't be empty.")
            continue
        if pw != getpass.getpass("  Confirm password: "):
            print("  Passwords didn't match — try again.")
            continue
        try:
            with open(auth.password_path, "w", encoding="utf-8") as f:
                json.dump(_hash_password(pw), f)
        except Exception as e:
            print(f"  Could not save the password: {e}")
            return False
        print("  Password set.")
        return True
    print("  Could not set a password.")
    return False


def verify_password(auth, attempts: int = 3) -> bool:
    """Prompt for the password and check it against the stored hash."""
    if not password_is_set(auth):
        return True  # no password configured -> nothing to check
    try:
        with open(auth.password_path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        salt = bytes.fromhex(rec["salt"])
        expected = bytes.fromhex(rec["hash"])
        iterations = int(rec.get("iter", _PBKDF2_ITERATIONS))
    except Exception:
        print("  Password file is unreadable. Delete "
              f"'{auth.password_path}' to reset it (you'll set a new one).")
        return False  # fail closed
    for i in range(attempts):
        pw = getpass.getpass("  Password: ")
        dk = hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), salt, iterations)
        if hmac.compare_digest(dk, expected):
            return True
        remaining = attempts - i - 1
        print("  Incorrect password." + ("  Try again." if remaining else ""))
    return False


def needs_onboarding(cfg, auth, faces) -> bool:
    """True if nobody is registered yet, or voice/password isn't set up."""
    no_users = not _load_users(auth)
    voice_missing = not os.path.exists(cfg.voiceprint_path)
    password_missing = bool(auth.require_password) and not os.path.exists(
        auth.password_path)
    return no_users or voice_missing or password_missing


def onboard(cfg, auth, verifier, faces, stream, speak) -> None:
    """First-run registration: name + authority, then face and voice."""
    speak("Welcome to Atlas. Let's set up your identity so only you can start me.")

    # --- Name + authority (only one admin allowed) ---
    name, role = _prompt_identity(auth, speak)
    auth.owner_name = name           # rest of setup + this run's login use it
    ok, info = register(auth, name, role)
    if not ok:                       # admin taken (belt-and-suspenders)
        register(auth, name, "user")
        info = "user"
        print(f"  Admin is already assigned; registered '{name}' as user.")
    print(f"  Registered '{name}' as {info}.")
    speak(f"Registered {name} as {info}.")

    # --- Face ---
    if (faces is not None and getattr(faces, "available", False)
            and auth.owner_name not in faces.names()):
        from vision import capture_camera

        speak("First, your face. Please look at the camera.")
        got, attempts = 0, 0
        while got < auth.face_shots and attempts < auth.face_shots * 3:
            attempts += 1
            try:
                msg = faces.enroll(capture_camera(0), auth.owner_name)
            except Exception as e:
                msg = f"camera error: {e}"
            if msg.startswith("Saved"):
                got += 1
                print(f"  face {got}/{auth.face_shots} captured.")
            else:
                speak("I couldn't see your face clearly. Hold still and look at "
                      "the camera.")
        if got:
            speak("Your face is registered.")
        else:
            speak("I couldn't register your face, so I'll use your voice only.")
    elif faces is None or not getattr(faces, "available", False):
        print("[auth] face recognition unavailable — voice-only enrollment.")

    # --- Voice ---
    if not os.path.exists(cfg.voiceprint_path):
        print("\n=== Voice registration ===")
        print("Repeat each phrase out loud after I say it (also shown here):\n")
        speak("Now your voice. Please repeat each phrase after I say it.")
        samples = []
        for i in range(auth.voice_samples):
            phrase = _PHRASES[i % len(_PHRASES)]
            # Show the phrase in text too, so it can be read, not just heard.
            print(f"  [{i + 1}/{auth.voice_samples}] Say:  \"{phrase}\"")
            speak(f"Say: {phrase}")
            clip = audio_input.record_fixed(stream, auth.voice_seconds, cfg)
            if clip.size:
                samples.append(clip)
            print(f"      recorded ({len(samples)}/{auth.voice_samples}).")
        if samples:
            verifier.save_voiceprint(verifier.build_voiceprint(samples))
            speak("Your voice is registered.")
        else:
            speak("I didn't catch your voice, but you can enroll later with the "
                  "enrollment script.")

    # --- Password ---
    if bool(auth.require_password) and not os.path.exists(auth.password_path):
        speak("Finally, set a startup password. Please type it in the terminal.")
        set_password(auth)

    speak("Setup complete.")


def authenticate(cfg, auth, verifier, voiceprint, faces, stream, vad, speak):
    """Identify the person at startup and verify them. Returns a dict
    {"name", "role", "voiceprint"} on success, or None on failure.

    Multi-user: the face identifies WHO is logging in; their authority (role)
    comes from the registry, voice is checked against THEIR voiceprint, and the
    typed password is required only for the admin (it's the admin's secret).
    """
    from vision import capture_camera

    users = _load_users(auth)
    if not users:
        print("[auth] no registered users — allowing startup as admin.")
        return {"name": auth.owner_name, "role": "admin", "voiceprint": voiceprint}

    have_faces = faces is not None and getattr(faces, "available", False)

    for attempt in range(auth.auth_attempts):
        # --- Identify WHO is present (by face), else fall back to the admin. ---
        candidate, face_ok = None, True
        if have_faces:
            speak("Please look at the camera to verify your identity.")
            try:
                results = faces.identify(capture_camera(0))
            except Exception as e:
                print(f"[auth] camera error: {e}")
                results = []
            candidate = next((r["name"] for r in results if r["name"] in users),
                             None)
            face_ok = candidate is not None
        if candidate is None:                 # no face match / faces disabled
            candidate = admin_name(auth) or _first_user(auth)
        role = users.get(candidate, {}).get("role", "user")
        auth.owner_name = candidate

        # --- Password — only the admin has one, and only the admin must type it.
        if face_ok and role == "admin" and password_is_set(auth):
            speak("Please type your admin password in the terminal.")
            if not verify_password(auth, 1):
                remaining = auth.auth_attempts - attempt - 1
                speak("Incorrect password." + (" Let's try again." if remaining
                                               else " Atlas will not start."))
                continue

        # --- Voice — against THIS user's voiceprint (skipped if they have none).
        voice_ok = True
        vp = _load_user_voiceprint(verifier, cfg, candidate) if verifier else None
        if face_ok and vp is not None:
            phrase = _PHRASES[attempt % len(_PHRASES)]
            print("\n=== Voice login ===")
            print(f"  Say this to verify your voice:  \"{phrase}\"")
            speak(f"To verify your voice, say: {phrase}")
            audio = audio_input.record_until_silence(stream, vad, cfg)
            voice_ok, score = verifier.verify(audio, vp)
            print(f"  voice score {score:.2f} ({'ok' if voice_ok else 'reject'}).")

        if face_ok and voice_ok:
            if candidate and candidate != "owner":
                speak(f"Welcome back, {candidate}."
                      + (f" Authority: {role}." if role else ""))
            else:
                speak("Welcome back.")
            return {"name": candidate, "role": role, "voiceprint": vp or voiceprint}

        reason = ("your face or voice" if not face_ok and not voice_ok
                  else "your face" if not face_ok else "your voice")
        remaining = auth.auth_attempts - attempt - 1
        speak(f"I couldn't verify {reason}."
              + (" Let's try again." if remaining else ""))

    return None
