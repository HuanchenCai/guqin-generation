"""Merge separately encoded aligned batches without duplicating latent data."""

import json
import os
import csv
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parent / "guqin_aligned_412"
TARGET = ROOT / "latents"
EXPECTED = {"a": 2143, "b": 621}


def main() -> None:
    TARGET.mkdir(exist_ok=True)
    seen = set()
    duplicate = 0
    for prefix, count in EXPECTED.items():
        source = ROOT / f"latents_batch_{1 if prefix == 'a' else 2}"
        paths = sorted(source.glob("*.json"))
        if len(paths) != count:
            raise RuntimeError(f"{source}: {len(paths)} / {count} metadata files")
        for metadata in paths:
            audio = Path(json.loads(metadata.read_text(encoding="utf-8"))["path"]).name
            if audio in seen:
                duplicate += 1
                continue
            seen.add(audio)
            for extension in (".json", ".npy"):
                original = metadata.with_suffix(extension)
                if not original.exists():
                    raise RuntimeError(f"Missing {original}")
                destination = TARGET / f"{prefix}_{original.name}"
                if not destination.exists():
                    os.link(original, destination)
    with (ROOT / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        expected = {row["file"] for row in csv.DictReader(handle)}
    missing = sorted(expected - seen)
    unexpected = sorted(seen - expected)
    if unexpected:
        raise RuntimeError(f"Unexpected encoded sources: {unexpected[:10]}")
    for name in missing:
        audio, _ = sf.read(ROOT / "audio" / name, dtype="float32", always_2d=True)
        if np.any(audio):
            raise RuntimeError(f"Unencoded window has nonzero sound: {name}")
    (ROOT / "silent_windows_skipped.json").write_text(json.dumps({
        "all_manifest_windows": len(expected), "encoded_unique_windows": len(seen),
        "duplicate_replacements": duplicate, "pure_digital_silence_skipped": missing,
    }, indent=2), encoding="utf-8")
    silence = ROOT / "latents_batch_1" / "silence.npy"
    if not (TARGET / silence.name).exists():
        os.link(silence, TARGET / silence.name)
    print(f"Merged {len(seen)} unique aligned windows; skipped {len(missing)} pure-silence windows", flush=True)


if __name__ == "__main__":
    main()
