"""Measure inference speed and memory of the final adapter, and training time
per pass from checkpoint timestamps; writes paper/data/timing.json."""

import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import soundfile as sf
import torch
from stable_audio_3 import StableAudioModel

from build_guqin_captions import HEAD_TEXT
from generate_guqin_piece_comparison import export

WORK = Path(__file__).resolve().parent
RUN = WORK / "guqin_aligned_412" / "story_run"
FINAL = RUN / "pass14" / "seg_00000" / "epoch=1-step=2500.ckpt"
OPENING = WORK.parent / "outputs" / "古琴第14遍对比" / "audio" / "open_01.flac"
PROMPT = f"{HEAD_TEXT} Sunset over a misty lake; an old fisherman rows home, singing quietly to himself."


def timed(fn, runs=5):
    fn()  # warm-up (compilation, caches)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return {"median_seconds": round(statistics.median(times), 2), "runs": runs,
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}


def main() -> None:
    model = StableAudioModel.from_pretrained("medium", device="cuda")
    model.load_lora([str(export(FINAL))])
    context, rate = sf.read(OPENING, dtype="float32", always_2d=True)
    prefix = torch.from_numpy(context.T.copy())

    def continuation():
        with torch.inference_mode():
            model.generate(prompt=PROMPT, duration=35, steps=8, seed=1, inpaint_audio=(rate, prefix),
                           inpaint_mask_start_seconds=10, inpaint_mask_end_seconds=35, chunked_decode=True)

    def text_only():
        with torch.inference_mode():
            model.generate(prompt=PROMPT, duration=35, steps=8, seed=1, chunked_decode=True)

    cont, text = timed(continuation), timed(text_only)
    stamps = sorted(datetime.fromtimestamp(p.stat().st_mtime) for p in RUN.glob("pass*/seg_*/*.ckpt"))
    gaps = [(b - a).total_seconds() / 60 for a, b in zip(stamps, stamps[1:])]
    per_pass = statistics.median(g for g in gaps if g < 90)  # skip pauses between sessions
    out = {
        "gpu": torch.cuda.get_device_name(0),
        "continuation_10s_plus_25s": cont,
        "text_only_35s": text,
        "real_time_factor_text_only": round(35 / text["median_seconds"], 1),
        "minutes_per_training_pass_including_validation": round(per_pass, 1),
        "training_steps_per_pass": 2500,
        "observed_training_vram_gb": 10.7,
        "note": "Training VRAM is the nvidia-smi reading during training (10941 MiB), not a peak measurement.",
    }
    path = WORK.parent / "paper" / "data" / "timing.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
