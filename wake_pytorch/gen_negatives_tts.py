"""Synthesize ISOLATED single-word NEGATIVES with Kokoro TTS.

Why this exists: every positive ("Atlas") is a single short word centered in a
quiet 1.5 s window, while the MUSAN negatives are *continuous* speech/music. With
nothing else, the model takes a shortcut — "one short isolated word in quiet =
Atlas" — and false-wakes on "hello", "hey", "yellow", "help", etc. These clips
close that gap: hundreds of OTHER single words, in the identical format to the
positives, so the model must key on the actual sound of "Atlas".

Same voices/speeds/post-processing as gen_positives.py (so the only difference
from a positive is the word itself). Output -> data/negative_words/.

    wake_pytorch/.venv/Scripts/python wake_pytorch/gen_negatives_tts.py            # all
    wake_pytorch/.venv/Scripts/python wake_pytorch/gen_negatives_tts.py --limit 20 # smoke test

Then retrain:  wake_pytorch/.venv/Scripts/python wake_pytorch/train.py
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from gen_positives import _postprocess, VOICES_A, VOICES_B, SPEEDS
from features import SAMPLE_RATE

BASE = Path(__file__).resolve().parent
OUT = BASE / "data" / "negative_words"

# Common single words a user actually says near the assistant — greetings,
# interjections, one-word commands — plus rhyme families around the observed
# false wakes ("hello"/"yellow"/"help"). NONE of these is the wake word.
WORDS = [
    # greetings / interjections
    "hello", "hey", "hi", "hiya", "howdy", "yo", "yeah", "yes", "no", "nope",
    "nah", "okay", "oh", "ah", "hmm", "wow", "oops", "alright", "thanks",
    "please", "sorry", "bye", "goodbye", "cheers", "hooray", "aha",
    # one-word commands
    "play", "stop", "pause", "next", "previous", "back", "open", "close",
    "start", "wait", "cancel", "help", "search", "mute", "unmute", "louder",
    "quieter", "volume", "brighter", "resume", "repeat", "done", "ready",
    "listen", "enough", "never", "again",
    # -ello / -ellow / -allow family (near "hello"/"yellow")
    "yellow", "mellow", "fellow", "hollow", "follow", "pillow", "willow",
    "bellow", "shallow", "swallow", "hallow", "cello",
    # -all / -al family
    "hall", "call", "ball", "tall", "wall", "fall", "small", "all", "also",
    # everyday nouns
    "water", "phone", "music", "coffee", "morning", "evening", "weather",
    "time", "timer", "light", "lights", "computer", "laptop", "window",
    "table", "kitchen", "message", "email", "reminder", "alarm",
    # short adjectives / directions / digits
    "good", "great", "cool", "nice", "right", "left", "up", "down",
    "one", "two", "three", "four", "five", "hundred",
    # HARD phonetic near-twins of "Atlas" (/ˈæt.ləs/) — the sound-alikes from
    # record_negatives.py. TTS across many voices covers what a few personal
    # recordings can't. NONE is the wake word.
    "at last", "outlast", "ballast", "at us", "add less", "a class", "at less",
    "hatless", "artless", "flatlands", "cutlass",
    "callous", "Dallas", "Wallace", "Pallas", "gallows", "palace",
    "malice", "chalice", "Alice", "trellis", "necklace",
    "solace", "jealous", "zealous", "careless", "reckless", "restless",
    "cactus", "lettuce", "lattice", "practice", "notice", "novice", "office",
    "atlantic", "atlanta", "athlete", "atmosphere", "actress",
    "at least", "hopeless", "helpless", "homeless", "harmless", "aimless",
    "endless", "clueless", "hapless", "a list",
    "arctic", "antarctica", "antarctic", "atlantis", "attic",
    "access", "matches",
]

# Two speeds is plenty of prosody variety here (augmentation adds the rest);
# keeps the count reasonable across ~27 voices.
NEG_SPEEDS = [0.9, 1.1]


def main(limit: int | None = None, voices_per_word: int = 8, seed: int = 0) -> None:
    from kokoro import KPipeline

    OUT.mkdir(parents=True, exist_ok=True)
    pipelines = {"a": KPipeline(lang_code="a"), "b": KPipeline(lang_code="b")}
    all_voices = [("a", v) for v in VOICES_A] + [("b", v) for v in VOICES_B]
    rng = random.Random(seed)

    made = 0
    for word in WORDS:
        # A random subset of voices per word — full word coverage with enough
        # voice variety, without synthesizing every word in all ~27 voices
        # (augmentation supplies the rest). Each pick gets a random speed.
        for lang, voice in rng.sample(all_voices, min(voices_per_word, len(all_voices))):
            if limit is not None and made >= limit:
                print(f"\nReached --limit {limit}; stopping at {made} clips.")
                return
            speed = rng.choice(NEG_SPEEDS)
            fn = f"{word}_{voice}_s{int(speed * 100)}.wav"
            if (OUT / fn).exists():         # resumable: skip already-made clips
                made += 1
                continue
            chunks = [a for _, _, a in pipelines[lang](word, voice=voice, speed=speed)]
            if not chunks:
                continue
            audio = np.concatenate([
                (c.detach().cpu().numpy() if isinstance(c, torch.Tensor)
                 else np.asarray(c)).astype(np.float32) for c in chunks])
            pcm = _postprocess(audio)
            sf.write(str(OUT / fn), pcm, SAMPLE_RATE)
            made += 1
            if made % 100 == 0:
                print(f"  {made} word-negatives...", flush=True)

    print(f"Done: {made} isolated-word negative clips in {OUT}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Generate at most N clips (quick smoke test).")
    ap.add_argument("--voices-per-word", type=int, default=8,
                    help="how many random voices to synthesize each word in")
    ap.add_argument("--seed", type=int, default=0,
                    help="change to add a fresh batch (new voice/speed picks) "
                         "on top of an existing run — filenames differ, so "
                         "resumable skip only skips true duplicates")
    args = ap.parse_args()
    main(limit=args.limit, voices_per_word=args.voices_per_word, seed=args.seed)
