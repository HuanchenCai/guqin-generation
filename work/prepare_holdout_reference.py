"""Extract a held-out solo-guqin reference for audio-conditioned continuation."""

import csv
from pathlib import Path

import soundfile as sf


ROOT = Path(__file__).parent
MANIFEST = ROOT / "guqin_v0" / "source_manifest.csv"
OUTPUT = ROOT / "guqin_holdout_reference.flac"


def main() -> None:
    with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        held_out = [row for row in csv.DictReader(handle) if row["split"] == "holdout"]
    source = Path(next(row["path"] for row in held_out if "蔡德允" in row["path"]))
    with sf.SoundFile(source) as stream:
        frames = 30 * stream.samplerate
        stream.seek(int((stream.frames - frames) * 0.35))
        samples = stream.read(frames, dtype="float32", always_2d=True)
        sf.write(OUTPUT, samples, stream.samplerate, format="FLAC", subtype="PCM_16")
    print(f"{source} -> {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
