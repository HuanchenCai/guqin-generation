"""Listening samples from the story-caption run (a chosen pass).

Part 1: 8 validation openings (unseen by both adapters; distinct performers
and pieces): new adapter prompted with story + tags + mode, vs adapter C from
the semitone test (329 pieces, tags + mode).
Part 2: text only, no opening: tag prompts and one free-written scene.
"""

import csv
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from stable_audio_3 import StableAudioModel

from build_guqin_captions import HEAD_TEXT, mode_text, normalize
from build_guqin_story_captions import load_pieces, story_text, tag_text
from generate_guqin_piece_comparison import export
from guqin_pitch_metric import best_pentatonic, cent_histogram, pitch_classes, tuning_offset

WORK = Path(__file__).resolve().parent
ROOT = WORK / "guqin_aligned_412"
OUT = WORK.parent / "outputs" / "古琴故事版试听"
AUDIO = OUT / "audio"
SEED = 2027
OLD_C = ROOT / "pilot3x_tags_mode" / "epoch=2-step=6624.ckpt"
TEXT_ONLY = [
    ("tags", ["孤寂", "夜", "江河"]),
    ("tags", ["激昂", "雄浑", "风"]),
    ("tags", ["平静", "流水", "隐逸"]),
    ("tags", ["苍茫", "塞外大漠", "思念"]),
    ("tags", ["欢快", "春", "飞鸟"]),
    ("free", "A lone traveller moored on an autumn river at night, the moon setting, "
             "remembering an old friend far away. 秋夜泊舟，月落江寒，怀念远方故人。"),
]


def openings(n=8) -> list[dict]:
    with (ROOT / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["split"] == "validation" and r["file"].endswith("_002.flac")]
    chosen, artists, pieces = [], set(), set()
    for r in sorted(rows, key=lambda r: r["file"]):
        piece = normalize(r["piece"])
        if r["artist"] not in artists and piece not in pieces:
            chosen.append(r)
            artists.add(r["artist"])
            pieces.add(piece)
        if len(chosen) == n:
            break
    return chosen


def main() -> None:
    new_ckpt = Path(sys.argv[1])
    AUDIO.mkdir(parents=True, exist_ok=True)
    pieces = load_pieces()
    items = []
    for i, row in enumerate(openings(), 1):
        audio, rate = sf.read(ROOT / "audio" / row["file"], dtype="float32", always_2d=True)
        window = audio[10 * rate:20 * rate]
        name = f"open_{i:02d}.flac"
        sf.write(AUDIO / name, window, rate, subtype="PCM_16")
        hist = cent_histogram(window.mean(axis=1))
        gong = int(best_pentatonic(pitch_classes(hist, tuning_offset(hist)))[1][0])
        info = pieces[normalize(row["piece"])]
        items.append({"file": name, "artist": row["artist"], "piece": row["piece"],
                      "prompts": {"new": f"{HEAD_TEXT} {story_text(info)} {tag_text(info['tags'])} {mode_text(gong)}",
                                  "C": f"{HEAD_TEXT} {tag_text(info['tags'])} {mode_text(gong)}"}})
    texts = []
    for i, (kind, value) in enumerate(TEXT_ONLY, 1):
        prompt = f"{HEAD_TEXT} {tag_text(value)}" if kind == "tags" else f"{HEAD_TEXT} {value}"
        label = "、".join(value) if kind == "tags" else value
        texts.append({"file": f"t{i:02d}_new.flac", "label": label, "prompt": prompt})
    records = []
    for code, ckpt in (("new", new_ckpt), ("C", OLD_C)):
        model = StableAudioModel.from_pretrained("medium", device="cuda")
        model.load_lora([str(export(ckpt))])
        model.set_lora_strength(1.0)
        jobs = [(f"o{i:02d}_{code}.flac", it["prompts"][code], it["file"]) for i, it in enumerate(items, 1)]
        if code == "new":
            jobs += [(t["file"], t["prompt"], None) for t in texts]
        for name, prompt, opening in jobs:
            started = time.monotonic()
            kwargs = {}
            if opening:
                context, rate = sf.read(AUDIO / opening, dtype="float32", always_2d=True)
                kwargs = {"inpaint_audio": (rate, torch.from_numpy(context.T.copy())),
                          "inpaint_mask_start_seconds": 10, "inpaint_mask_end_seconds": 35}
            with torch.inference_mode():
                result = model.generate(prompt=prompt, duration=35, steps=8, seed=SEED, chunked_decode=True, **kwargs)
            sf.write(AUDIO / name, result[0].detach().float().cpu().T.numpy(), 44100, format="FLAC", subtype="PCM_16")
            records.append({"file": name, "model": code, "opening": opening, "prompt": prompt,
                            "checkpoint": ckpt.name, "seconds": round(time.monotonic() - started, 1)})
            print(json.dumps(records[-1], ensure_ascii=False), flush=True)
        del model
        torch.cuda.empty_cache()
    (OUT / "样本记录.json").write_text(json.dumps({"openings": items, "texts": texts, "records": records},
                                                ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
