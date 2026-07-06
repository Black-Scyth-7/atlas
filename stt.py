"""Stage 4: speech-to-text with faster-whisper (small model, GPU).

Bridges a recorded float32 clip to text for the LLM.

Run this file directly for a standalone Step 3 test: it waits for the wake word,
records, applies the speaker gate (if enabled), and prints the transcript.
    python stt.py
"""

import glob
import os
import re

import numpy as np
from faster_whisper import WhisperModel

from config import Config


def _norm(s: str) -> str:
    """Normalize for hallucination matching: lowercase, strip punctuation/
    apostrophes, collapse whitespace. "Yep, that's it!" -> "yep thats it"."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", "", s.lower())).strip()


def _preload_cuda_dlls() -> None:
    """Pre-load the CUDA runtime DLLs faster-whisper/ctranslate2 needs (Windows).

    The pip `nvidia-*-cu12` packages drop cublas/cudnn/cudart under
    site-packages/nvidia/*/bin, but ctranslate2 loads them BY NAME and does NOT
    search that directory — so a plain GPU load fails with "cublas64_12.dll is
    not found or cannot be loaded". Loading each by FULL PATH here (dependencies
    first) puts them in the process so ctranslate2's later by-name loads resolve
    to the already-resident modules. No-op off Windows or if the libs are absent.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        import nvidia  # namespace package from the nvidia-*-cu12 wheels
    except ImportError:
        return
    # cudart -> cublasLt -> cublas -> cudnn: load dependencies before dependents.
    for name in ("cudart64_12.dll", "cublasLt64_12.dll",
                 "cublas64_12.dll", "cudnn64_9.dll"):
        for base in list(getattr(nvidia, "__path__", [])):
            hits = glob.glob(os.path.join(base, "**", name), recursive=True)
            if hits:
                try:
                    ctypes.WinDLL(hits[0])
                except OSError:
                    pass
                break


class Transcriber:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Pre-normalize the caption-hallucination blocklist once for fast lookup.
        self._halluc = {_norm(p) for p in
                        getattr(cfg, "stt_hallucination_phrases", ()) if _norm(p)}
        self.model = None
        self.on_gpu = False
        self._load_model()

    def _load_model(self) -> None:
        """(Re)load the Whisper model on the configured device; fall back to CPU
        if the GPU load fails. Sets self.model and self.on_gpu."""
        cfg = self.cfg
        if cfg.stt_device == "cuda":
            _preload_cuda_dlls()
        try:
            self.model = WhisperModel(cfg.stt_model, device=cfg.stt_device,
                                      compute_type=cfg.stt_compute_type)
            self.on_gpu = (cfg.stt_device == "cuda")
        except Exception as e:
            if cfg.stt_device == "cuda":
                # GPU unavailable (missing CUDA libs / OOM / driver) — degrade to
                # CPU rather than failing startup. Slower but keeps Atlas working.
                print(f"STT: GPU load failed ({e}); falling back to CPU int8.")
                self.model = WhisperModel(cfg.stt_model, device="cpu",
                                          compute_type="int8")
                self.on_gpu = False
            else:
                raise

    def release(self) -> bool:
        """Unload the model to free its VRAM (e.g. while the vision model runs on
        a small GPU). Returns True if a GPU model was actually freed. The next
        transcribe() reloads it lazily."""
        if self.model is None or not self.on_gpu:
            return False
        import gc

        self.model = None            # CTranslate2 frees the GPU memory on destruct
        self.on_gpu = False
        gc.collect()
        return True

    def ensure_loaded(self) -> None:
        """Reload the model if it was released (no-op if already loaded)."""
        if self.model is None:
            self._load_model()

    def transcribe(self, audio: np.ndarray) -> tuple[str, str]:
        """Transcribe a float32 16 kHz mono clip. Returns (text, language).

        Whisper is multilingual; with stt_language "" it auto-detects the spoken
        language per utterance (e.g. 'en', 'hi') so downstream can reply in the
        same language and pick the matching TTS voice.

        Whisper hallucinates confident-sounding text on silence/noise (often
        repeated phrases, or an echo of the initial_prompt). We guard against it:
          - vad_filter drops non-speech regions before decoding (biggest win);
          - condition_on_previous_text=False stops one bad segment seeding a
            runaway repetition loop;
          - the *_threshold knobs make Whisper mark low-confidence / silent
            segments as no-speech so they're dropped;
          - we post-filter any surviving segment whose no_speech_prob is high
            and avg_logprob is very low (a hallucination fingerprint);
          - an RMS floor rejects near-silent clips (a breath / click / room
            noise) before decoding, since those are what Whisper turns into
            confident caption phrases like "Thank you." / "Bye.";
          - a caption-phrase blocklist drops a transcript whose every sentence
            is a known Whisper hallucination ("thanks for watching", etc.) even
            when the model reports it confidently.
        Rejected clips return "" so the caller speaks not_understood_phrase and
        re-listens — better than acting on a phantom command.
        """
        if audio.size == 0:
            return "", ""
        self.ensure_loaded()   # reload if it was parked to free VRAM for vision
        # Energy floor: real speech is rms ~0.05-0.2; near-silent clips (rms well
        # under this) are the ones Whisper hallucinates captions from.
        min_rms = getattr(self.cfg, "stt_min_rms", 0.0)
        if min_rms > 0.0 and float(np.sqrt(np.mean(audio.astype(np.float32) ** 2))) < min_rms:
            return "", ""
        segments, info = self.model.transcribe(
            audio, language=self.cfg.stt_language or None,
            beam_size=getattr(self.cfg, "stt_beam_size", 5),
            initial_prompt=(getattr(self.cfg, "stt_initial_prompt", "") or None),
            condition_on_previous_text=False,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            no_speech_threshold=getattr(self.cfg, "stt_no_speech_threshold", 0.6),
            log_prob_threshold=getattr(self.cfg, "stt_log_prob_threshold", -1.0),
            compression_ratio_threshold=getattr(
                self.cfg, "stt_compression_ratio_threshold", 2.4),
            vad_filter=getattr(self.cfg, "stt_vad_filter", True),
            vad_parameters={"min_silence_duration_ms": 500},
        )
        no_speech_max = getattr(self.cfg, "stt_no_speech_threshold", 0.6)
        logprob_min = getattr(self.cfg, "stt_log_prob_threshold", -1.0)
        kept = [
            seg.text for seg in segments
            # Drop segments that look like hallucinations: model itself is fairly
            # sure there's no speech AND the decode was low-probability.
            if not (seg.no_speech_prob > no_speech_max
                    and seg.avg_logprob < logprob_min)
        ]
        text = " ".join(kept).strip()
        if self._is_hallucination(text):
            return "", info.language
        return text, info.language

    def _is_hallucination(self, text: str) -> bool:
        """True if EVERY sentence of `text` is a known caption hallucination.

        Whole-phrase match only, so a real command that merely contains such a
        word survives: "thank you atlas" and "bye for now atlas" get through
        (extra words), while bare "Thank you." / "Yep, that's all there is to
        it. Bye." are dropped.
        """
        if not self._halluc:
            return False
        parts = [p for p in re.split(r"[.!?]+", text) if _norm(p)]
        return bool(parts) and all(_norm(p) in self._halluc for p in parts)


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
