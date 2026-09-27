"""Showcase v0: free-written scene prompts, no opening, like the best-rated
text-only clip of the story run (t06, the autumn-river night mooring).

Five new scenes x three seeds; the user picks one take per scene.
"""

import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from stable_audio_3 import StableAudioModel

from build_guqin_captions import HEAD_TEXT
from generate_guqin_piece_comparison import export

WORK = Path(__file__).resolve().parent
OUT = WORK.parent / "outputs" / "古琴展示v0"
AUDIO = OUT / "audio"
SEEDS = (2027, 31, 808)
SCENES = [
    ("山居夜雨", "Rain on a mountain hut at night; a lone lamp, incense, a hermit sitting alone listening to the rain. "
             "山居夜雨，孤灯焚香，独坐听雨。"),
    ("边关月夜", "Moonlight over a frontier fortress; a soldier hears a distant flute and thinks of home. "
             "边关月夜，戍客闻笛思乡。"),
    ("渔舟晚唱", "Sunset over a misty lake; an old fisherman rows home, singing quietly to himself. "
             "夕阳烟波，渔翁归舟独唱。"),
    ("松下观云", "Sitting under an ancient pine on a high peak, watching clouds drift through the valleys. "
             "高山古松下，看云卷云舒。"),
    ("雪夜访友", "A snowy night; rowing a small boat to visit a friend, then turning back at the door, content. "
             "雪夜乘舟访友，兴尽而返。"),
]


def main() -> None:
    checkpoint = Path(sys.argv[1])
    AUDIO.mkdir(parents=True, exist_ok=True)
    model = StableAudioModel.from_pretrained("medium", device="cuda")
    model.load_lora([str(export(checkpoint))])
    model.set_lora_strength(1.0)
    records = []
    for i, (title, scene) in enumerate(SCENES, 1):
        for seed in SEEDS:
            name = f"s{i}_{seed}.flac"
            started = time.monotonic()
            with torch.inference_mode():
                result = model.generate(prompt=f"{HEAD_TEXT} {scene}", duration=35, steps=8, seed=seed,
                                        chunked_decode=True)
            sf.write(AUDIO / name, result[0].detach().float().cpu().T.numpy(), 44100, format="FLAC", subtype="PCM_16")
            records.append({"file": name, "scene": title, "prompt": f"{HEAD_TEXT} {scene}", "seed": seed,
                            "checkpoint": str(checkpoint), "seconds": round(time.monotonic() - started, 1)})
            print(json.dumps(records[-1], ensure_ascii=False), flush=True)
    (OUT / "样本记录.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
