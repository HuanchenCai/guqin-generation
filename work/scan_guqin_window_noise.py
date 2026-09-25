"""Measure friction-noise density of every aligned training window."""

import csv
import json
from pathlib import Path

import torch

from guqin_noise_features import features, load_mono

ROOT = Path(__file__).resolve().parent / "guqin_aligned_412"
TARGET = ROOT / "window_noise.csv"


def main() -> None:
    torch.set_num_threads(4)
    with (ROOT / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    skipped = set(json.loads((ROOT / "silent_windows_skipped.json").read_text(
        encoding="utf-8"))["pure_digital_silence_skipped"])
    rows = []
    for index, row in enumerate(manifest, 1):
        if row["file"] in skipped:
            continue
        rows.append({**row, **{k: round(v, 4) for k, v in
                               features(load_mono(ROOT / "audio" / row["file"])).items()}})
        if index % 200 == 0:
            print(f"{index}/{len(manifest)}", flush=True)
    with TARGET.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {TARGET}", flush=True)


if __name__ == "__main__":
    main()
