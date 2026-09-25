"""Make an unprocessed listening reel of every v0 training excerpt."""

import csv
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).parent
SOURCE = ROOT / "guqin_v0" / "audio"
MANIFEST = ROOT / "guqin_v0" / "source_manifest.csv"
OUTPUT = ROOT.parent / "outputs" / "古琴素材审听"
RATE = 44100
LENGTH = 6
GAP = 0.4


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        sources = [row for row in csv.DictReader(handle) if row["split"] == "train"]

    parts = []
    rows = []
    elapsed = 0.0
    for index, row in enumerate(sources, 1):
        volume = next(part for part in Path(row["path"]).parts if part.startswith("CD第"))
        for part_index in (1, 2):
            clip = SOURCE / f"{volume}_{index:02d}_{part_index}.flac"
            with sf.SoundFile(clip) as stream:
                if stream.samplerate != RATE or stream.channels != 2:
                    raise ValueError(f"Unexpected clip format: {clip}")
                count = LENGTH * RATE
                stream.seek(max(0, (stream.frames - count) // 2))
                audio = stream.read(count, dtype="float32", always_2d=True)
            if len(audio) < count:
                audio = np.pad(audio, ((0, count - len(audio)), (0, 0)))
            parts.append(audio)
            rows.append({
                "number": len(rows) + 1,
                "start_seconds": round(elapsed, 1),
                "end_seconds": round(elapsed + LENGTH, 1),
                "track": Path(row["path"]).stem,
                "clip": clip.name,
                "original_file": row["path"],
            })
            elapsed += LENGTH
            parts.append(np.zeros((int(GAP * RATE), 2), dtype="float32"))
            elapsed += GAP

    sf.write(OUTPUT / "训练素材逐段审听.flac", np.concatenate(parts), RATE, format="FLAC", subtype="PCM_16")
    with (OUTPUT / "时间对照表.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} samples, duration {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
