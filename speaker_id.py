"""Stage 3: speaker verification with SpeechBrain's ECAPA-TDNN model (CPU).

Turns a voice clip into a 192-d embedding, averages several enrollment clips
into one reference "voiceprint", and verifies new clips by cosine similarity
against a tunable threshold.

IMPORTANT: this is PERSONALIZATION, not security. It only judges whether a clip
sounds like the enrolled owner and is trivially spoofable by a recording. Do not
use it as an access-control mechanism.

Run this file directly for a standalone Step 2 test (requires a voiceprint from
enroll.py): it waits for the wake word, records each command, and prints
accept/reject plus the cosine score. No transcription.
    python speaker_id.py
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

from config import Config


class SpeakerVerifier:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        # ECAPA-TDNN trained on VoxCeleb. Downloads once (~80 MB), then cached.
        # local_strategy=COPY copies the model files into savedir instead of
        # symlinking them — Windows symlinks need Developer Mode/admin, and the
        # symlink default otherwise prints a noisy warning on every startup.
        self.encoder = EncoderClassifier.from_hparams(
            source=cfg.speaker_model,
            savedir="pretrained/spkrec-ecapa",
            run_opts={"device": cfg.speaker_device},
            local_strategy=LocalStrategy.COPY,
        )

    def embed(self, audio: np.ndarray) -> torch.Tensor:
        """Return a single L2-normalized 192-d embedding for a float32 clip."""
        wav = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # [1, samples]
        with torch.no_grad():
            emb = self.encoder.encode_batch(wav)  # [1, 1, 192]
        emb = emb.squeeze()                       # [192]
        return F.normalize(emb, dim=0)

    def build_voiceprint(self, samples: list[np.ndarray]) -> np.ndarray:
        """Average several enrollment clips into one reference voiceprint."""
        embs = torch.stack([self.embed(s) for s in samples])
        centroid = F.normalize(embs.mean(dim=0), dim=0)
        return centroid.cpu().numpy()

    def save_voiceprint(self, voiceprint: np.ndarray) -> None:
        np.save(self.cfg.voiceprint_path, voiceprint)

    def load_voiceprint(self) -> torch.Tensor:
        """Load the enrolled voiceprint. Raises FileNotFoundError if missing."""
        arr = np.load(self.cfg.voiceprint_path)
        return torch.tensor(arr, dtype=torch.float32)

    def verify(self, audio: np.ndarray, voiceprint: torch.Tensor) -> tuple[bool, float]:
        """Compare a clip to the stored voiceprint. Returns (is_match, score)."""
        if audio.size == 0:
            return False, 0.0
        emb = self.embed(audio)
        score = F.cosine_similarity(emb, voiceprint, dim=0).item()
        return score >= self.cfg.speaker_threshold, score


if __name__ == "__main__":
    # Standalone Step 2 test: wake word -> record -> verify speaker -> print.
    import webrtcvad
    import audio_input

    cfg = Config()

    if not os.path.exists(cfg.voiceprint_path):
        raise SystemExit("No voiceprint found. Run `python enroll.py` first.")

    print("Loading models...")
    verifier = SpeakerVerifier(cfg)
    voiceprint = verifier.load_voiceprint()
    oww = audio_input.load_wake_model(cfg)
    vad = webrtcvad.Vad(cfg.vad_aggressiveness)

    try:
        stream = audio_input.open_stream(cfg)
    except Exception as e:
        raise SystemExit(f"Could not open the microphone: {e}")

    print(f"Ready. Say '{cfg.wake_model}', then a command. Ctrl+C to quit.")
    print(f"(threshold = {cfg.speaker_threshold:.2f})\n")
    try:
        while True:
            audio_input.wait_for_wake_word(stream, oww, cfg)
            print("Wake word detected — recording...")
            audio = audio_input.record_until_silence(stream, vad, cfg)
            if audio.size == 0:
                print("  No speech detected.\n")
                continue
            is_match, score = verifier.verify(audio, voiceprint)
            verdict = "ACCEPT" if is_match else "REJECT"
            print(f"  {verdict}  (cosine score {score:.3f})\n")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stream.stop()
        stream.close()
