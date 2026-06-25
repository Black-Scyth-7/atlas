"""Stage 6a: text-to-speech with Piper.

Wraps a Piper voice and exposes two paths:
  - synthesize(text): the whole reply as one float32 clip.
  - stream(text):     one float32 clip per sentence, so playback can begin
                      speaking the first sentence while the rest still synthesize.

Run this file directly for a standalone Step 5 test: it synthesizes a line and
saves it to tts_test.wav.
    python tts.py
"""

import os
import re

import numpy as np
from piper import PiperVoice

from config import TTSConfig

# Split on sentence-ending punctuation followed by whitespace. Includes the
# Devanagari danda (।॥) so Hindi replies chunk correctly too.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।॥])\s+")


def split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENTENCE_SPLIT.split(text.strip())]
    return [s for s in parts if s]


class TTS:
    def __init__(self, cfg: TTSConfig):
        self.cfg = cfg
        self._voices: dict[str, object] = {}  # lang -> PiperVoice (lazy-loaded)
        self.default_lang = cfg.default_lang
        # Load the default voice eagerly so a missing model fails fast, and to
        # fix the output sample rate (all voices must share it).
        default = self._load(self.default_lang)
        if default is None:
            model = cfg.voices.get(self.default_lang, ("?",))[0]
            raise FileNotFoundError(
                f"Default Piper voice not found ({model}). Download a voice "
                "(.onnx + .onnx.json) — see README."
            )
        self.sample_rate: int = default.config.sample_rate

    def _load(self, lang: str):
        """Lazy-load the Piper voice for a language; None if not available."""
        if lang in self._voices:
            return self._voices[lang]
        spec = self.cfg.voices.get(lang)
        if not spec:
            return None
        model, config = spec
        if not os.path.exists(model):
            return None
        voice = PiperVoice.load(model, config, use_cuda=False)
        self._voices[lang] = voice
        return voice

    def _voice_for(self, lang: str | None):
        """Voice for a language code, falling back to the default voice."""
        return (self._load(lang) if lang else None) or self._voices[self.default_lang]

    def _synth(self, text: str, lang: str | None = None) -> np.ndarray:
        """Synthesize one piece of text in `lang` to float32 [-1, 1] mono."""
        text = text.strip()
        if not text:
            return np.empty(0, dtype=np.float32)
        voice = self._voice_for(lang)
        chunks = [c.audio_int16_array for c in voice.synthesize(text)]
        if not chunks:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32) / 32768.0

    def synthesize(self, text: str, lang: str | None = None) -> np.ndarray:
        """Whole reply as a single float32 clip in the given language."""
        return self._synth(text, lang)

    def stream(self, text: str, lang: str | None = None):
        """Yield one float32 clip per sentence for early/streaming playback."""
        for sentence in split_sentences(text):
            audio = self._synth(sentence, lang)
            if audio.size:
                yield audio


if __name__ == "__main__":
    import wave

    cfg = TTSConfig()
    tts = TTS(cfg)
    line = "Hello, I am Atlas. I can hear you, think, and speak back."
    print(f"Synthesizing at {tts.sample_rate} Hz: {line!r}")
    audio = tts.synthesize(line)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open("tts_test.wav", "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(tts.sample_rate)
        w.writeframes(pcm.tobytes())
    print(f"Wrote {audio.size / tts.sample_rate:.1f}s to tts_test.wav. Play it back.")
