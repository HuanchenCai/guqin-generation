"""Calibrate a CLAP recording-noise prompt against owner-flagged guqin recordings."""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from transformers import ClapModel, ClapProcessor


HERE = Path(__file__).resolve().parent
ROOT = Path(r"Y:\Music\古琴曲")
MODEL = "laion/clap-htsat-unfused"
EXCLUDED = {"张育瑾", "谢孝苹"}
TEXTS = {
    "clean_studio": "A clear recording of a solo guqin with a quiet background, delicate plucked strings and natural resonance.",
    "clean_music": "Clean high-fidelity instrumental music with no hiss, crackle or background noise.",
    "archival_hiss": "An old noisy archival recording of solo guqin with loud tape hiss and crackling in the background.",
    "constant_hiss": "A solo guqin performance with constant audible background hiss and static noise.",
    "scratchy_record": "A scratchy low fidelity gramophone music recording with loud surface noise.",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    with (HERE / "full_guqin_signal_scan.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not args.all:
        bad = [row for row in rows if row["artist"] in EXCLUDED]
        rest = [row for row in rows if row["artist"] not in EXCLUDED and
                not any(label in row["title"] for label in ("合奏", "琴箫", "琴簫", "琴歌", "唱弦", "伴奏"))]
        rows = bad + random.Random(42).sample(rest, 45)
    torch.set_num_threads(6)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = ClapProcessor.from_pretrained(MODEL)
    model = ClapModel.from_pretrained(MODEL).eval().to(device)
    prompts = processor(text=list(TEXTS.values()), return_tensors="pt", padding=True)
    with torch.inference_mode():
        text_features = model.get_text_features(
            **{key: val.to(device) for key, val in prompts.items()}).pooler_output
    output = []
    for number, row in enumerate(rows, 1):
        path = ROOT / Path(row["relative_path"].replace("\\", "/"))
        with sf.SoundFile(path) as stream:
            frames = min(10 * stream.samplerate, stream.frames)
            stream.seek(max(0, (stream.frames - frames) // 2))
            wave = stream.read(frames, dtype="float32", always_2d=True).mean(axis=1)
            rate = stream.samplerate
        if rate != 48000:
            wave = AF.resample(torch.from_numpy(wave), rate, 48000).numpy()
        inputs = processor(audio=[wave.astype(np.float32)], sampling_rate=48000, return_tensors="pt")
        with torch.inference_mode():
            features = model.get_audio_features(
                **{key: val.to(device) for key, val in inputs.items()}).pooler_output
            scores = (features @ text_features.T)[0].float().cpu().numpy()
        result = {"relative_path": row["relative_path"], "artist": row["artist"],
                  "user_noise_flag": row["artist"] in EXCLUDED,
                  **{key: round(float(score), 4) for key, score in zip(TEXTS, scores)}}
        result["noise_margin"] = round(max(result[key] for key in ("archival_hiss", "constant_hiss", "scratchy_record")) -
                                       max(result[key] for key in ("clean_studio", "clean_music")), 4)
        output.append(result)
        if number % 10 == 0 or number == len(rows):
            print(f"Scored {number}/{len(rows)}", flush=True)
    target = HERE / ("full_guqin_clap_noise.json" if args.all else "guqin_clap_noise_pilot.json")
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    flagged = [row["noise_margin"] for row in output if row["user_noise_flag"]]
    other = [row["noise_margin"] for row in output if not row["user_noise_flag"]]
    print(json.dumps({"count": len(output), "known_bad_median": float(np.median(flagged)),
                      "other_median": float(np.median(other)),
                      "known_bad_range": [min(flagged), max(flagged)],
                      "other_range": [min(other), max(other)]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
