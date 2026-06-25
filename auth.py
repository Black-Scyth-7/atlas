"""Startup identity gate: face + voice registration and login.

On the first ever run, Atlas registers the owner's face (webcam, InsightFace)
and voice (voiceprint, SpeechBrain). On every later run it requires BOTH the
owner's face AND voice to match before the assistant starts.

NOTE: this is PERSONALIZATION, not security. Both checks are spoofable (a photo
of the owner, a recording of their voice). It keeps Atlas personal to you; it is
not an access-control mechanism. If you get locked out, set
`AuthConfig.require_identity = False` in config.py.

The functions take their dependencies explicitly (verifier, face recognizer, the
shared mic stream, a `speak` callback) so main.py wires them to the already-
loaded components.
"""

from __future__ import annotations

import os

import audio_input

# Spoken prompts to read back during voice enrollment (content is irrelevant to
# the text-independent ECAPA model; they just give the user something to say).
_PHRASES = [
    "The quick brown fox jumps over the lazy dog.",
    "I would like to check the weather and my schedule today.",
    "Please set a timer for ten minutes from now.",
    "Tell me something interesting about the solar system.",
    "Atlas, this is my voice for verification.",
]


def needs_onboarding(cfg, auth, faces) -> bool:
    """True if the owner's voice or face isn't registered yet (first run)."""
    voice_missing = not os.path.exists(cfg.voiceprint_path)
    face_missing = (faces is not None and getattr(faces, "available", False)
                    and auth.owner_name not in faces.names())
    return voice_missing or face_missing


def onboard(cfg, auth, verifier, faces, stream, speak) -> None:
    """First-run registration of the owner's face and voice."""
    speak("Welcome to Atlas. Let's set up your identity so only you can start me.")

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
        speak("Now your voice. Please repeat each phrase after I say it.")
        samples = []
        for i in range(auth.voice_samples):
            phrase = _PHRASES[i % len(_PHRASES)]
            speak(f"Say: {phrase}")
            clip = audio_input.record_fixed(stream, auth.voice_seconds, cfg)
            if clip.size:
                samples.append(clip)
            print(f"  voice {len(samples)}/{auth.voice_samples} recorded.")
        if samples:
            verifier.save_voiceprint(verifier.build_voiceprint(samples))
            speak("Your voice is registered.")
        else:
            speak("I didn't catch your voice, but you can enroll later with the "
                  "enrollment script.")

    speak("Setup complete.")


def authenticate(cfg, auth, verifier, voiceprint, faces, stream, vad, speak) -> bool:
    """Verify the owner's face AND voice. Returns True only if both pass."""
    from vision import capture_camera

    use_face = (faces is not None and getattr(faces, "available", False)
                and auth.owner_name in faces.names())
    use_voice = verifier is not None and voiceprint is not None

    if not use_face and not use_voice:
        print("[auth] nothing enrolled to verify against — allowing startup.")
        return True

    for attempt in range(auth.auth_attempts):
        # --- Face ---
        face_ok = True
        if use_face:
            speak("Please look at the camera to verify your identity.")
            try:
                results = faces.identify(capture_camera(0))
            except Exception as e:
                print(f"[auth] camera error: {e}")
                results = []
            face_ok = any(r["name"] == auth.owner_name for r in results)

        # --- Voice ---
        voice_ok = True
        if use_voice:
            speak("Now say something so I can verify your voice.")
            audio = audio_input.record_until_silence(stream, vad, cfg)
            voice_ok, score = verifier.verify(audio, voiceprint)
            print(f"[auth] voice score {score:.2f} "
                  f"({'ok' if voice_ok else 'reject'}).")

        if face_ok and voice_ok:
            speak("Welcome back.")
            return True

        if not face_ok and not voice_ok:
            reason = "your face or voice"
        elif not face_ok:
            reason = "your face"
        else:
            reason = "your voice"
        remaining = auth.auth_attempts - attempt - 1
        if remaining:
            speak(f"I couldn't verify {reason}. Let's try again.")
        else:
            speak(f"I couldn't verify {reason}.")

    return False
