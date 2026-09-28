"""Pass 14 vs pass 2 (and C) with identical prompts and seeds.

Reuses the pass-2 and C continuations from 古琴故事版试听 and the pass-2
showcase takes the user picked; only pass 14 is generated here.
"""

import json
import os
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from stable_audio_3 import StableAudioModel

from generate_guqin_piece_comparison import export

WORK = Path(__file__).resolve().parent
OUTPUTS = WORK.parent / "outputs"
STORY = OUTPUTS / "古琴故事版试听"
SHOW = OUTPUTS / "古琴展示v0"
OUT = OUTPUTS / "古琴第14遍对比"
AUDIO = OUT / "audio"
PICKS = {"山居夜雨": 808, "边关月夜": 31, "渔舟晚唱": 31, "松下观云": 808, "雪夜访友": 2027}


def link(src: Path, name: str) -> None:
    target = AUDIO / name
    if not target.exists():
        os.link(src, target)


def main() -> None:
    checkpoint = Path(sys.argv[1])
    AUDIO.mkdir(parents=True, exist_ok=True)
    story = json.loads((STORY / "样本记录.json").read_text(encoding="utf-8"))
    show = json.loads((SHOW / "样本记录.json").read_text(encoding="utf-8"))
    prompts_new = {r["file"]: r["prompt"] for r in story["records"] if r["model"] == "new"}
    jobs, cont, scenes = [], [], []
    for i, item in enumerate(story["openings"], 1):
        link(STORY / "audio" / item["file"], item["file"])
        link(STORY / "audio" / f"o{i:02d}_new.flac", f"o{i:02d}_pass2.flac")
        link(STORY / "audio" / f"o{i:02d}_C.flac", f"o{i:02d}_C.flac")
        jobs.append((f"o{i:02d}_pass14.flac", prompts_new[f"o{i:02d}_new.flac"], item["file"], 2027))
        cont.append({**item, "index": i})
    for k, (scene, seed) in enumerate(PICKS.items(), 1):
        take = next(r for r in show if r["scene"] == scene and r["seed"] == seed)
        link(SHOW / "audio" / take["file"], f"s{k}_pass2.flac")
        jobs.append((f"s{k}_pass14.flac", take["prompt"], None, seed))
        scenes.append({"index": k, "scene": scene, "seed": seed, "prompt": take["prompt"]})
    model = StableAudioModel.from_pretrained("medium", device="cuda")
    model.load_lora([str(export(checkpoint))])
    model.set_lora_strength(1.0)
    for name, prompt, opening, seed in jobs:
        started = time.monotonic()
        kwargs = {}
        if opening:
            context, rate = sf.read(AUDIO / opening, dtype="float32", always_2d=True)
            kwargs = {"inpaint_audio": (rate, torch.from_numpy(context.T.copy())),
                      "inpaint_mask_start_seconds": 10, "inpaint_mask_end_seconds": 35}
        with torch.inference_mode():
            result = model.generate(prompt=prompt, duration=35, steps=8, seed=seed, chunked_decode=True, **kwargs)
        sf.write(AUDIO / name, result[0].detach().float().cpu().T.numpy(), 44100, format="FLAC", subtype="PCM_16")
        print(name, round(time.monotonic() - started, 1), flush=True)
    (OUT / "样本记录.json").write_text(json.dumps({"checkpoint": str(checkpoint), "continuations": cont,
                                                "scenes": scenes}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
