"""Independently screen v1 clips for voice and accompaniment with AudioSet AST."""

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import ASTFeatureExtractor, ASTForAudioClassification


ROOT = Path(__file__).parent
CLIPS = ROOT / "guqin_v1" / "audio"
OUTPUT = ROOT / "guqin_v1_audioset_audit.csv"
MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
KEYWORDS = ("speech", "singing", "flute", "guitar", "drum", "harp", "zither", "musical instrument")


def load_audio(path: Path) -> np.ndarray:
    with sf.SoundFile(path) as stream:
        frames = 10 * stream.samplerate
        stream.seek(max(0, (stream.frames - frames) // 2))
        samples = stream.read(frames, dtype="float32", always_2d=True).mean(axis=1)
        samples = resample_poly(samples, 16000, stream.samplerate)
    return samples.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=CLIPS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    extractor = ASTFeatureExtractor.from_pretrained(MODEL)
    model = ASTForAudioClassification.from_pretrained(MODEL).eval()
    selected = {
        int(index): label
        for index, label in model.config.id2label.items()
        if any(word in label.lower() for word in KEYWORDS)
    }
    paths = sorted(args.input_dir.glob("*.flac"))
    rows = []
    for index, path in enumerate(paths, 1):
        features = extractor(load_audio(path), sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            scores = torch.sigmoid(model(**features).logits)[0].numpy()
        ranked = sorted(((label, float(scores[class_id])) for class_id, label in selected.items()), key=lambda x: x[1], reverse=True)
        row = {"clip": path.name, **{label: round(value, 4) for label, value in ranked}}
        rows.append(row)
        print(f"{index}/{len(paths)} {path.name}: {ranked[:5]}", flush=True)
    fieldnames = ["clip"] + sorted({key for row in rows for key in row if key != "clip"})
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
