"""Stage 4: speech-to-text with faster-whisper (small model, CPU).

Bridges a recorded float32 clip to text for the LLM.

Run this file directly for a standalone Step 3 test: it waits for the wake word,
records, applies the speaker gate (if enabled), and prints the transcript.
    python stt.py
"""

import os

import numpy as np
from faster_whisper import WhisperModel

from config import Config


class Transcriber:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = WhisperModel(
            cfg.stt_model,
            device=cfg.stt_device,
            compute_type=cfg.stt_compute_type,
        )

    def transcribe(self, audio: np.ndarray) -> tuple[str, str]:
        """Transcribe a float32 16 kHz mono clip. Returns (text, language).

        Whisper is multilingual; with stt_language "" it auto-detects the spoken
        language per utterance (e.g. 'en', 'hi') so downstream can reply in the
        same language and pick the matching TTS voice.
        """
        if audio.size == 0:
            return "", ""
        segments, info = self.model.transcribe(
            audio, language=self.cfg.stt_language or None, beam_size=1
        )
        text = " ".join(seg.text for seg in segments).strip()
        return text, info.language


if __name__ == "__main__":
    # Standalone Step 3 test: wake -> record -> (speaker gate) -> transcribe.
    import webrtcvad
    import audio_input
    from speaker_id import SpeakerVerifier

    cfg = Config()

    print("Loading models...")
    oww = audio_input.load_wake_model(cfg)
    vad = webrtcvad.Vad(cfg.vad_aggressiveness)
    transcriber = Transcriber(cfg)

    verifier = None
    voiceprint = None
    if cfg.require_speaker_match:
        if not os.path.exists(cfg.voiceprint_path):
            raise SystemExit("No voiceprint found. Run `python enroll.py` first.")
        verifier = SpeakerVerifier(cfg)
        voiceprint = verifier.load_voiceprint()

    try:
        stream = audio_input.open_stream(cfg)
    except Exception as e:
        raise SystemExit(f"Could not open the microphone: {e}")

    print(f"Ready. Say '{cfg.wake_model}', then a command. Ctrl+C to quit.\n")
    try:
        while True:
            audio_input.wait_for_wake_word(stream, oww, cfg)
            print("Wake word detected — recording...")
            audio = audio_input.record_until_silence(stream, vad, cfg)
            if audio.size == 0:
                print("  No speech detected.\n")
                continue

            if verifier is not None:
                is_match, score = verifier.verify(audio, voiceprint)
                if not is_match:
                    print(f"  REJECT (score {score:.3f}) — not transcribing.\n")
                    continue
                print(f"  ACCEPT (score {score:.3f})")

            text, lang = transcriber.transcribe(audio)
            if not text:
                print("  (no speech recognized)\n")
                continue
            print(f"  transcript [{lang}]: {text}\n")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stream.stop()
        stream.close()
