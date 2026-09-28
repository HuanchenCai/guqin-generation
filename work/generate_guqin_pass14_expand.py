"""Build on the pass-14 model after the user's ratings.

1. Extend the favourite take (渔舟晚唱, seed 31) to about 3.5 minutes: each
   round feeds the last 15 s as the opening and adds 30 s; two chains with
   different seeds.
2. (Dropped by the user: twenty more takes of the same scene.)
3. Showcase v1: the five scenes x four seeds, to re-pick with pass 14.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from stable_audio_3 import StableAudioModel

from build_guqin_captions import HEAD_TEXT
from generate_guqin_piece_comparison import export
from generate_guqin_showcase_v0 import SCENES

WORK = Path(__file__).resolve().parent
OUTPUTS = WORK.parent / "outputs"
EXPAND = OUTPUTS / "古琴渔舟晚唱拓展"
SHOW = OUTPUTS / "古琴展示v1"
RATE = 44100
FAVOURITE = OUTPUTS / "古琴第14遍对比" / "audio" / "s3_pass14.flac"
FISHER = f"{HEAD_TEXT} {dict(SCENES)['渔舟晚唱']}"
CONTEXT, NEW, ROUNDS, FADE = 15, 30, 6, 0.1
CHAINS = {"A": 101, "B": 202}
TAKE_SEEDS = []  # the user dropped the 20 extra takes
SHOW_SEEDS = (2027, 31, 808, 4242)


def generate(model, prompt, seed, seconds, context=None):
    kwargs = {}
    if context is not None:
        kwargs = {"inpaint_audio": (RATE, torch.from_numpy(context.T.copy())),
                  "inpaint_mask_start_seconds": CONTEXT, "inpaint_mask_end_seconds": seconds}
    with torch.inference_mode():
        out = model.generate(prompt=prompt, duration=seconds, steps=8, seed=seed, chunked_decode=True, **kwargs)
    return out[0].detach().float().cpu().T.numpy()


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-9)


def extend(model, seed_base: int, lock_db: float | None = None) -> tuple[np.ndarray, list]:
    """Chain continuations; join each new part with a short crossfade.

    Chained continuation drifts louder (each round continues a slightly
    louder opening). With lock_db, the whole piece starts at that RMS level,
    every opening is fed at that level, and each new part is matched to the
    loudness of the context before it.
    """
    audio, _ = sf.read(FAVOURITE, dtype="float32", always_2d=True)
    target = 10 ** (lock_db / 20) if lock_db is not None else None
    if target:
        audio = audio * (target / rms(audio))
    log = []
    fade = int(FADE * RATE)
    ramp = np.linspace(0, 1, fade)[:, None]
    for k in range(ROUNDS):
        context = audio[-CONTEXT * RATE:]
        gain = target / rms(context) if target else 1.0
        out = generate(model, FISHER, seed_base + k, CONTEXT + NEW, context * gain) / gain
        cut = CONTEXT * RATE
        if target:
            out = out * (rms(context) / rms(out[cut:]))
        # The first seconds of `out` re-decode the context; crossfade into the new part.
        audio[-fade:] = audio[-fade:] * (1 - ramp) + out[cut - fade:cut] * ramp
        audio = np.concatenate([audio, out[cut:]])
        new = out[cut:]
        log.append({"round": k + 1, "seed": seed_base + k,
                    "new_rms_dbfs": round(float(20 * np.log10(np.sqrt(np.mean(new ** 2)) + 1e-9)), 1)})
    return audio, log


def lock_only(checkpoint: Path, lock_db: float = -16.0) -> None:
    """Add loudness-locked versions of the two chains to the existing records."""
    model = StableAudioModel.from_pretrained("medium", device="cuda")
    model.load_lora([str(export(checkpoint))])
    model.set_lora_strength(1.0)
    path = EXPAND / "样本记录.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    for name, seed_base in CHAINS.items():
        audio, log = extend(model, seed_base, lock_db)
        file = f"extend_{name}_lock.flac"
        sf.write(EXPAND / "audio" / file, audio, RATE, format="FLAC", subtype="PCM_16")
        records["chains"] = [c for c in records["chains"] if c["file"] != file]
        records["chains"].append({"file": file, "seconds": round(len(audio) / RATE, 1), "rounds": log,
                                  "lock_db": lock_db})
        print(json.dumps(records["chains"][-1], ensure_ascii=False), flush=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    checkpoint = Path(sys.argv[1])
    if "--lock-only" in sys.argv:
        return lock_only(checkpoint)
    (EXPAND / "audio").mkdir(parents=True, exist_ok=True)
    (SHOW / "audio").mkdir(parents=True, exist_ok=True)
    model = StableAudioModel.from_pretrained("medium", device="cuda")
    model.load_lora([str(export(checkpoint))])
    model.set_lora_strength(1.0)
    records = {"checkpoint": str(checkpoint), "prompt": FISHER, "chains": [], "takes": [], "showcase": []}
    for name, seed_base in CHAINS.items():
        started = time.monotonic()
        audio, log = extend(model, seed_base)
        file = f"extend_{name}.flac"
        sf.write(EXPAND / "audio" / file, audio, RATE, format="FLAC", subtype="PCM_16")
        records["chains"].append({"file": file, "seconds": round(len(audio) / RATE, 1), "rounds": log,
                                  "elapsed": round(time.monotonic() - started, 1)})
        print(json.dumps(records["chains"][-1], ensure_ascii=False), flush=True)
    for seed in TAKE_SEEDS:
        file = f"take_{seed:02d}.flac"
        sf.write(EXPAND / "audio" / file, generate(model, FISHER, seed, 35), RATE, format="FLAC", subtype="PCM_16")
        records["takes"].append({"file": file, "seed": seed})
        print(file, flush=True)
    for i, (scene, text) in enumerate(SCENES, 1):
        for seed in SHOW_SEEDS:
            file = f"s{i}_{seed}.flac"
            sf.write(SHOW / "audio" / file, generate(model, f"{HEAD_TEXT} {text}", seed, 35), RATE,
                     format="FLAC", subtype="PCM_16")
            records["showcase"].append({"file": file, "scene": scene, "seed": seed, "prompt": f"{HEAD_TEXT} {text}"})
            print(file, flush=True)
    (EXPAND / "样本记录.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (SHOW / "样本记录.json").write_text(json.dumps(records["showcase"], ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
