# Atlas Wake Word — PyTorch / MatchboxNet Training Plan

Replaces the old openWakeWord pipeline with a self-contained, from-scratch
PyTorch keyword spotter for the single wake word **"Atlas"**.

- **Model:** MatchboxNet (1D time-channel separable conv ResNet), implemented
  from scratch in PyTorch — no NeMo dependency.
- **Features:** **64 MFCC** (locked; matches the MatchboxNet paper). Identical
  transform at train and runtime (the one contract that must never drift).
- **Positives:** synthesized with **Kokoro TTS** (Kokoro-82M, ~50 voices).
- **Negatives + noise augmentation:** **MUSAN** (music / speech / noise).
- **Hardware:** RTX 5050 (Blackwell), `wake_pytorch/.venv` (torch 2.11+cu128).

---

## 1. The feature contract (shared train ⇄ runtime)

Defined once in `wake_pytorch/features.py` and imported by BOTH training and the
runtime detector, so they can never diverge.

| Param | Value |
|---|---|
| sample rate | 16 000 Hz, mono |
| window (n_fft / win_length) | 400 samples (25 ms) |
| hop_length | 160 samples (10 ms) |
| n_mels | 64 |
| n_mfcc | 64 |
| f_min / f_max | 20 / 8000 Hz |
| mel→dB | `AmplitudeToDB` before DCT (torchaudio MFCC default) |
| normalization | per-feature (per coefficient) mean/std over the window |
| detection window | 1.5 s = 24 000 samples → ~150 frames |
| inference hop (streaming) | 1280 samples (80 ms) — same cadence as before |

`torchaudio.transforms.MFCC` (n_mfcc=64, melkwargs n_fft=400/hop=160/n_mels=64)
+ per-feature normalize. At runtime we keep a rolling 1.5 s int16 ring buffer and
re-run every 80 ms.

---

## 2. Positives — Kokoro TTS  (`gen_positives.py`)

Kokoro-82M via the `kokoro` pip package (`KPipeline`), 24 kHz output → resample
to 16 kHz.

- Loop over **all ~50 voices** (af_*, am_*, bf_*, bm_* … American + British).
- Text variants for prosody diversity: `"Atlas"`, `"Atlas."`, `"Atlas?"`,
  `"Atlas!"`, `"...Atlas."` (leading pause).
- Speed sweep per voice: `speed ∈ {0.85, 1.0, 1.15}`.
- Post: trim leading/trailing silence, peak-normalize, pad/center into 1.5 s.
- Target: **~4–8k** clean positive clips → `data/positive/`.
- **First pass: Kokoro-only** (no own-voice recording yet). `record_positives.py`
  is deferred to a later iteration once the Kokoro-only model is validated.

## 3. Negatives — MUSAN + hard words  (`prepare_negatives.py`)

Download MUSAN (openslr.org/17, ~11 GB) → `data/musan/{music,noise,speech}`.

- **MUSAN speech** → the key **hard negatives** (continuous speech / babble that
  is NOT "Atlas"). Sliced into 1.5 s windows.
- **MUSAN music + noise** → both standalone negatives AND the background pool for
  augmentation (below).
- **Synthetic quiet/ambient** clips (silence, mic floor, pink/brown noise, mains
  hum) — port the openWakeWord-free `_synth_quiet_clip` generator; teaches the
  model that a quiet idle mic is not a wake.
- (Optional) short common words as extra hard negatives.

## 4. Augmentation  (`dataset.py`, applied on-the-fly)

- **MUSAN mixing:** add random noise/music/babble to positives (and some
  negatives) at random **SNR 0–20 dB**. This is what makes it robust in a real room.
- **SpecAugment:** time + frequency masking on the MFCC features (MatchboxNet standard).
- **Time shift / random crop** of the 1.5 s window.
- (Optional later) RIR reverberation.

---

## 5. Model — MatchboxNet  (`model.py`)

MatchboxNet-**3×2×64** (B=3 blocks, R=2 sub-blocks, C=64), binary head.

- **Prologue:** TCSConv(k=11, C=64) → BN → ReLU.
- **3 residual blocks**, each 2 sub-blocks of `[TCSConv(depthwise, k) → pointwise
  1×1 → BN → ReLU → Dropout]`; kernel sizes per block `[13, 15, 17]`; residual
  1×1 conv + BN added before the block's final activation.
- **Epilogue:** TCSConv(k=29, dilation=2, C=128) → 1×1 conv(128) → 1×1 conv(2).
- **Global average pool over time** → 2-class logits (`atlas` / `not-atlas`).

TCSConv = depthwise 1D conv over time (grouped) + pointwise 1×1 mixing channels —
the efficiency trick that makes MatchboxNet tiny (~93k params) and fast on CPU.
Global pooling means it tolerates variable-length input (streaming-friendly).

## 6. Training  (`train.py`)

- Loss: cross-entropy (or BCE on 1 logit); class-balanced sampling of pos/neg.
- Optimizer: AdamW + cosine LR; SpecAugment + MUSAN aug in the loader.
- Steps/epochs env-tunable; runs on the 5050 GPU (`.venv`).
- Track **val accuracy** AND **false-positives-per-hour** on a held-out MUSAN
  speech+noise stream — the metric that actually matters for a wake word.
- Checkpoint best model → `checkpoints/atlas_matchboxnet.pt`.

## 7. Eval + threshold  (`eval.py`)

- Sweep detection threshold; report ROC, FP/hour vs true-accept rate.
- Pick the threshold giving the target (e.g. ≤ 0.5 FP/hour at ≥ 95% accept) →
  becomes `config.wake_threshold`.

## 8. Export  (`export_onnx.py`)

- `torch.onnx.export` the MatchboxNet → `wake_pytorch/atlas.onnx`
  (self-contained, opset 17), copy to `models/atlas.onnx`.
- Verify parity: same MFCC input → PyTorch logits == onnxruntime logits.

---

## 9. Runtime wiring (replaces openWakeWord in the app)

New `WakeDetector` class (in `wake_pytorch/detector.py`, imported by the app):

- Loads `models/atlas.onnx` via **onnxruntime** (already a runtime dep).
- Holds the rolling 1.5 s buffer; computes MFCC via the SAME `features.py`.
- Exposes an openWakeWord-compatible surface so app changes stay minimal:
  - `predict(chunk_int16) -> float`  (wake probability 0..1)
  - `reset()`
- Same `wake_threshold` / `wake_consecutive` / `wake_warmup_chunks` gating logic
  already in `audio_input.wait_for_wake_word` and `main._bargein_watcher`.

Edits to existing files:
- `audio_input.py`: drop `import openwakeword`; `load_wake_model()` returns a
  `WakeDetector`; `wait_for_wake_word` uses `det.predict(chunk)` (float) instead
  of `max(scores.values())`.
- `main.py`: `_bargein_watcher` uses the same float `predict`.
- `config.py`: rewrite the "Wake word" section — keep `wake_threshold`,
  `wake_consecutive`, `wake_warmup_chunks`, `bargein_*`, `wake_chunk`; point
  `wake_model` at the new `.onnx`; drop `wake_framework`.
- `stt.py` / `speaker_id.py`: update their `__main__` smoke tests.

---

## 10. File layout (`wake_pytorch/`)

```
wake_pytorch/
  PLAN.md              <- this file
  .venv/               <- GPU env (done)
  features.py          <- MFCC contract (shared)
  gen_positives.py     <- Kokoro "Atlas" synthesis
  record_positives.py  <- (optional) capture your own voice
  prepare_negatives.py <- MUSAN download/index + synthetic quiet
  dataset.py           <- dataset + MUSAN/SpecAugment augmentation
  model.py             <- MatchboxNet
  train.py             <- training loop
  eval.py              <- threshold + FP/hour
  export_onnx.py       <- -> atlas.onnx
  detector.py          <- runtime WakeDetector (imported by the app)
  data/                <- (gitignored) positives, musan, negatives
  checkpoints/         <- (gitignored) .pt
```

## 11. Extra venv deps (installed when we build each stage)

- `kokoro` (+ `misaki[en]`, `soundfile`) — Kokoro TTS.
- `matplotlib` — ROC/threshold plots in eval (optional).
- torch/torchaudio/numpy/onnx/onnxscript/onnxruntime/scipy/tqdm — already installed.

## 12. Build order

1. `features.py` (contract) → 2. `model.py` + a shape smoke test →
3. `gen_positives.py` (Kokoro) → 4. `prepare_negatives.py` (MUSAN) →
5. `dataset.py` → 6. `train.py` → 7. `eval.py` → 8. `export_onnx.py` →
9. `detector.py` + wire into `audio_input.py`/`main.py`/`config.py` →
10. end-to-end mic test.
