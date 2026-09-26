"""Does training longer, or naming the mode in text, reduce stray semitones?

Three adapters trained on the same 329-piece split, evaluated on test-split
openings (unseen compositions, each a different performer and piece):
  A  pilot_continuation: 1 pass, no text
  B  pilot3x_tags:       3 passes, mood tags
  C  pilot3x_tags_mode:  3 passes, mood tags + pentatonic mode
Every opening gets 3 seeds for the objective attack/pitch metrics; the
listening page uses the first seed only.
"""

import csv
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from stable_audio_3 import StableAudioModel

from build_guqin_captions import HEAD_TEXT, load_tags, mode_text, normalize, tag_text
from generate_guqin_piece_comparison import export
from guqin_onset_pitch import attack_report
from guqin_pitch_metric import best_pentatonic, cent_histogram, fit_report, pitch_classes, tuning_offset

WORK = Path(__file__).resolve().parent
ROOT = WORK / "guqin_aligned_412"
OUT = WORK.parent / "outputs" / "古琴半音对照"
AUDIO = OUT / "audio"
SEEDS = (2026, 7, 11)
MODELS = {"A": ROOT / "pilot_continuation", "B": ROOT / "pilot3x_tags", "C": ROOT / "pilot3x_tags_mode"}


def adapter(directory: Path) -> Path:
    ckpt = max(directory.glob("*.ckpt"), key=lambda p: int(p.stem.split("step=")[1]))
    return export(ckpt)


def openings(n=8) -> list[dict]:
    with (ROOT / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["split"] == "test" and r["file"].endswith("_002.flac")]
    chosen, artists, pieces = [], set(), set()
    for r in sorted(rows, key=lambda r: r["file"]):
        piece = normalize(r["piece"])
        if r["artist"] in artists or piece in pieces:
            continue
        chosen.append(r)
        artists.add(r["artist"])
        pieces.add(piece)
        if len(chosen) == n:
            break
    return chosen


def main() -> None:
    AUDIO.mkdir(parents=True, exist_ok=True)
    tags = load_tags()
    items = []
    for i, row in enumerate(openings(), 1):
        audio, rate = sf.read(ROOT / "audio" / row["file"], dtype="float32", always_2d=True)
        window = audio[10 * rate:20 * rate]
        name = f"open_{i:02d}.flac"
        sf.write(AUDIO / name, window, rate, subtype="PCM_16")
        hist = cent_histogram(window.mean(axis=1))
        _, mode = best_pentatonic(pitch_classes(hist, tuning_offset(hist)))
        piece = normalize(row["piece"])
        tag_prompt = f"{HEAD_TEXT} {tag_text(tags[piece]['tags'])}"
        items.append({"file": name, "artist": row["artist"], "piece": row["piece"], "source": row["source"],
                      "prompts": {"A": "", "B": tag_prompt, "C": f"{tag_prompt} {mode_text(int(mode[0]))}"}})
    records_path = OUT / "样本记录.json"
    records = json.loads(records_path.read_text(encoding="utf-8")) if records_path.exists() else []
    done = {r["file"] for r in records}
    for code, directory in MODELS.items():
        path = adapter(directory)
        model = StableAudioModel.from_pretrained("medium", device="cuda")
        model.load_lora([str(path)])
        model.set_lora_strength(1.0)
        for i, item in enumerate(items, 1):
            context, rate = sf.read(AUDIO / item["file"], dtype="float32", always_2d=True)
            for seed in SEEDS:
                name = f"o{i:02d}_s{seed}_{code}.flac"
                if name in done:
                    continue
                started = time.monotonic()
                with torch.inference_mode():
                    result = model.generate(prompt=item["prompts"][code], duration=35, steps=8, seed=seed,
                                            inpaint_audio=(rate, torch.from_numpy(context.T.copy())),
                                            inpaint_mask_start_seconds=10, inpaint_mask_end_seconds=35,
                                            chunked_decode=True)
                sf.write(AUDIO / name, result[0].detach().float().cpu().T.numpy(), rate,
                         format="FLAC", subtype="PCM_16")
                record = {"file": name, "model": code, "opening": item["file"], "seed": seed,
                          "prompt": item["prompts"][code], "adapter": path.parent.name,
                          "seconds": round(time.monotonic() - started, 1),
                          **fit_report(AUDIO / name, 10), **attack_report(AUDIO / name, 10)}
                records.append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
                records_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        del model
        torch.cuda.empty_cache()
    (OUT / "openings.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    real = [attack_report(AUDIO / it["file"], None)["attack_out_of_mode"] for it in items]
    summary = {"real_openings_attack_out_of_mode": round(float(np.mean(real)), 3)}
    for code in MODELS:
        rows = [r for r in records if r["model"] == code]
        summary[code] = {k: round(float(np.mean([r[k] for r in rows if r[k] is not None])), 3)
                         for k in ("attack_out_of_mode", "fit_to_opening_mode", "own_fit", "attacks_per_s")}
    (OUT / "客观指标.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
