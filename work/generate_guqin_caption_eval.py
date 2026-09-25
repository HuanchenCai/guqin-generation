"""Listening test for the captioned (piece + mood tags) 412-piece adapter.

Part 1: continuation from 10 unseen openings, each a different performer and
piece (none in the 412 or old 222 training sets), one seed each. Candidates:
old empty-caption adapter, new adapter with no text, with matching tags, and
with deliberately opposite tags.
Part 2: text-only generation (no opening) from mood tags, old vs new adapter.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from stable_audio_3 import StableAudioModel

from build_guqin_captions import tag_text
from generate_guqin_piece_comparison import export

WORK = Path(__file__).resolve().parent
ROOT = WORK / "guqin_aligned_412"
OUT = WORK.parent / "outputs" / "古琴文字标签试听"
AUDIO = OUT / "audio"
LIBRARY = Path("Y:/Music/古琴曲")
OLD = ROOT / "full_412" / "epoch=1-step=2764.safetensors"
NEW = ROOT / "full_412_captioned" / "epoch=1-step=2764.safetensors"

# (path, seed, matching tags, opposite tags). Tags for pieces outside the tag
# table were drafted the same way as guqin_piece_tags.csv.
OPENINGS = [
    ("CD第1册/CD11/查阜西-大江东去.wav", 84917, ["雄浑", "激昂", "苍茫", "江河"], ["恬淡", "平静", "月夜"]),
    ("CD第3册/41/乐瑛-岳阳三醉-残.wav", 94017, ["醉", "洒脱", "仙境"], ["忧伤", "孤寂", "夜"]),
    ("CD第2册/28/张子谦-泛沧浪.wav", 20260925, ["洒脱", "恬淡", "江河"], ["悲愤", "激昂", "塞外大漠"]),
    ("CD第1册/CD06/詹澄秋-鹤舞洞天.wav", 31415, ["灵动", "仙境", "飞鸟"], ["忧伤", "孤寂", "夜"]),
    ("CD第3册/44/胥桐华-耕莘钓渭.wav", 27182, ["恬淡", "田园", "渔樵"], ["激昂", "雄浑", "风"]),
    ("CD第2册/38/喻绍泽-桃李园序.wav", 16180, ["欢快", "灵动", "春"], ["孤寂", "忧伤", "秋"]),
    ("CD第3册/46/吴振平-屈原问渡.wav", 57721, ["悲愤", "忧伤", "江河"], ["欢快", "灵动", "春"]),
    ("CD第3册/47/朱龙庵-鸥鹭忘机1.wav", 14142, ["恬淡", "洒脱", "飞鸟", "江河"], ["悲愤", "激昂", "塞外大漠"]),
    ("CD第4册/61/李传爱-潇湘水云.wav", 17320, ["苍茫", "云", "江河", "忧伤"], ["欢快", "灵动", "春"]),
    ("CD第1册/CD18/管平湖-秋鸿.wav", 22360, ["苍茫", "秋", "飞鸟"], ["欢快", "田园", "春"]),
]
TEXT_ONLY = [(["孤寂", "夜"], 11111), (["激昂", "雄浑"], 22222), (["平静", "流水"], 33333),
             (["欢快", "春"], 44444), (["苍茫", "塞外大漠"], 55555)]
HEAD = "Solo guqin, Chinese seven-string zither."


def prompt(tags: list[str]) -> str:
    return f"{HEAD} {tag_text(tags)}"


def prepare_openings() -> list[dict]:
    AUDIO.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (rel, seed, match, opposite) in enumerate(OPENINGS, 1):
        audio, rate = sf.read(LIBRARY / rel, dtype="float32", always_2d=True)
        start = 40
        while start + 10 < len(audio) / rate:
            window = audio[start * rate:(start + 10) * rate]
            if 20 * np.log10(np.sqrt(np.mean(window ** 2)) + 1e-9) > -32:
                break
            start += 10
        name = f"open_{index:02d}.flac"
        sf.write(AUDIO / name, window, rate, subtype="PCM_16")
        artist, piece = Path(rel).stem.split("-", 1)
        rows.append({"file": name, "artist": artist, "piece": piece, "source": rel,
                     "start_seconds": start, "seed": seed, "match": match, "opposite": opposite})
    return rows


def jobs(openings: list[dict]) -> list[dict]:
    out = []
    for i, o in enumerate(openings, 1):
        for code, adapter, tags in (("old", OLD, None), ("none", NEW, None),
                                    ("match", NEW, o["match"]), ("opposite", NEW, o["opposite"])):
            out.append({"part": "continue", "code": code, "adapter": adapter, "opening": o,
                        "prompt": prompt(tags) if tags else "", "tags": tags or [],
                        "seed": o["seed"], "file": f"o{i:02d}_{code}.flac"})
    for i, (tags, seed) in enumerate(TEXT_ONLY, 1):
        for code, adapter in (("old", OLD), ("new", NEW)):
            out.append({"part": "text", "code": code, "adapter": adapter, "opening": None,
                        "prompt": prompt(tags), "tags": tags, "seed": seed,
                        "file": f"t{i:02d}_{code}.flac"})
    return out


def main() -> None:
    export(NEW.with_suffix(".ckpt"))
    openings = prepare_openings()
    records_path = OUT / "样本记录.json"
    records = json.loads(records_path.read_text(encoding="utf-8")) if records_path.exists() else []
    done = {r["file"] for r in records}
    todo = [j for j in jobs(openings) if j["file"] not in done]
    model, loaded = None, None
    for adapter in (OLD, NEW):
        batch = [j for j in todo if j["adapter"] == adapter]
        if not batch:
            continue
        del model
        torch.cuda.empty_cache()
        model = StableAudioModel.from_pretrained("medium", device="cuda")
        model.load_lora([str(adapter)])
        model.set_lora_strength(1.0)
        for job in batch:
            started = time.monotonic()
            kwargs = {}
            if job["opening"]:
                context, rate = sf.read(AUDIO / job["opening"]["file"], dtype="float32", always_2d=True)
                kwargs = {"inpaint_audio": (rate, torch.from_numpy(context.T.copy())),
                          "inpaint_mask_start_seconds": 10, "inpaint_mask_end_seconds": 35}
            with torch.inference_mode():
                result = model.generate(prompt=job["prompt"], duration=35, steps=8,
                                        seed=job["seed"], chunked_decode=True, **kwargs)
            audio = result[0].detach().float().cpu().T.numpy()
            sf.write(AUDIO / job["file"], audio, 44100, format="FLAC", subtype="PCM_16")
            record = {k: v for k, v in job.items() if k not in ("adapter", "opening")}
            record.update(adapter=adapter.parent.name, opening=job["opening"]["file"] if job["opening"] else None,
                          seconds=round(time.monotonic() - started, 1))
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            records_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "openings.json").write_text(json.dumps(openings, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
