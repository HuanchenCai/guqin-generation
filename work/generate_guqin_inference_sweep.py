"""Inference-side sweep for the 412-piece guqin adapter on unseen openings.

User ratings showed seed variance dwarfs adapter differences, and some
generations overuse left-hand friction noise until it sounds like sawing.
This sweep keeps the trained adapter fixed and varies only inference knobs,
plus the earlier 222-clip adapter as a reference. All openings come from
蔡德允, who was held out of every training set.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from stable_audio_3 import StableAudioModel

from guqin_noise_features import clip_report

WORK = Path(__file__).resolve().parent
ROOT = WORK / "guqin_aligned_412"
OUT = WORK.parent / "outputs" / "古琴推理调参"
AUDIO = OUT / "audio"
LIBRARY = Path("Y:/Music/古琴曲/CD第3册")
OPENINGS = ["42/蔡德允-关山月.wav", "42/蔡德允-平沙落雁1.wav", "42/蔡德允-梅花三弄.wav",
            "42/蔡德允-良宵引.wav", "43/蔡德允-潇湘水云1.wav", "43/蔡德允-阳关三叠.wav"]
SEEDS = (84917, 94017, 20260925)
FULL = ROOT / "full_412" / "epoch=1-step=2764.safetensors"
OLD = WORK / "sa3_feedback_bulk" / "medium_run" / "epoch=4-step=1000.safetensors"
# (code, description, model, adapter, strength, lora_interval, steps)
VARIANTS = [
    ("C10", "412 首 · 强度 1.0（现版本）", "medium", FULL, 1.0, None, 8),
    ("C07", "412 首 · 强度 0.7", "medium", FULL, 0.7, None, 8),
    ("CEarly", "412 首 · 只在前半去噪起作用", "medium", FULL, 1.0, (0.5, 1.0), 8),
    ("CBase", "412 首 · base 模型 50 步", "medium-base", FULL, 1.0, None, 50),
    ("D222", "旧版 222 段 1000 步", "medium", OLD, 1.0, None, 8),
]


def prepare_openings() -> list[dict]:
    AUDIO.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, rel in enumerate(OPENINGS, 1):
        audio, rate = sf.read(LIBRARY / rel, dtype="float32", always_2d=True)
        start = 40
        while start + 10 < len(audio) / rate:
            window = audio[start * rate:(start + 10) * rate]
            if 20 * np.log10(np.sqrt(np.mean(window ** 2)) + 1e-9) > -32:
                break
            start += 10
        name = f"open_{index:02d}.flac"
        sf.write(AUDIO / name, window, rate, subtype="PCM_16")
        rows.append({"file": name, "piece": Path(rel).stem.split("-", 1)[1],
                     "artist": "蔡德允", "source": rel, "start_seconds": start})
    return rows


def main() -> None:
    only = set(sys.argv[1:])
    openings = prepare_openings()
    records_path = OUT / "样本记录.json"
    records = json.loads(records_path.read_text(encoding="utf-8")) if records_path.exists() else []
    done = {r["file"] for r in records}
    loaded = None
    model = None
    for code, title, base, adapter, strength, interval, steps in VARIANTS:
        if only and code not in only:
            continue
        if loaded != (base, adapter):
            del model
            torch.cuda.empty_cache()
            model = StableAudioModel.from_pretrained(base, device="cuda")
            model.load_lora([str(adapter)])
            loaded = (base, adapter)
        model.set_lora_strength(strength)
        extra = {"lora_interval": interval} if interval else {}
        for index, opening in enumerate(openings, 1):
            context, rate = sf.read(AUDIO / opening["file"], dtype="float32", always_2d=True)
            prefix = torch.from_numpy(context.T.copy())
            for seed in SEEDS:
                name = f"o{index:02d}_{seed}_{code}.flac"
                if name in done:
                    continue
                started = time.monotonic()
                with torch.inference_mode():
                    result = model.generate(
                        prompt="", duration=35, steps=steps, seed=seed,
                        inpaint_audio=(rate, prefix), inpaint_mask_start_seconds=10,
                        inpaint_mask_end_seconds=35, chunked_decode=True, **extra)
                audio = result[0].detach().float().cpu().T.numpy()
                sf.write(AUDIO / name, audio, rate, format="FLAC", subtype="PCM_16")
                record = {"variant": code, "title": title, "opening": opening["file"],
                          "piece": opening["piece"], "seed": seed, "file": name,
                          "strength": strength, "interval": interval, "steps": steps,
                          "model": base, "adapter": adapter.name,
                          "seconds": round(time.monotonic() - started, 1),
                          **clip_report(AUDIO / name)}
                records.append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
                records_path.write_text(json.dumps(records, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    (OUT / "openings.json").write_text(json.dumps(openings, ensure_ascii=False, indent=2),
                                       encoding="utf-8")


if __name__ == "__main__":
    main()
