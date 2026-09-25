"""Build a small, lossless, track-disjoint guqin training set."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).parent
AUDIT = ROOT / "guqin_interior_audit.csv"
DATA = ROOT / "guqin_v0"
AUDIO = DATA / "audio"
CAPTION = (
    "Solo traditional Chinese guqin zither, sparse free rhythm, warm wooden "
    "plucked strings, resonant low notes, delicate sliding ornaments, long "
    "natural decays, intimate quiet room, unaccompanied instrumental"
)


def select_sources():
    by_volume = defaultdict(list)
    with AUDIT.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["error"]:
                continue
            path = Path(row["path"])
            volume = next(part for part in path.parts if part.startswith("CD第"))
            floor = float(row["interior_floor_dbfs"])
            dynamic_range = float(row["interior_range_db"])
            peak = float(row["sampled_peak_dbfs"])
            score = floor - 0.2 * dynamic_range + (4 if peak >= -0.5 else 0)
            by_volume[volume].append((score, path, row))

    train, holdout, used_artists = [], [], set()
    for volume, candidates in sorted(by_volume.items()):
        for _, path, row in sorted(candidates, key=lambda item: item[0]):
            artist = path.stem.split("-", 1)[0]
            if artist in used_artists:
                continue
            used_artists.add(artist)
            if sum(volume in p.parts for p, _ in train) < 4:
                train.append((path, row))
            elif sum(volume in p.parts for p, _ in holdout) < 1:
                holdout.append((path, row))
            else:
                break
    return train, holdout


def extract_clip(source: Path, target: Path, fraction: float, seconds: int = 30):
    with sf.SoundFile(source) as audio:
        clip_frames = int(seconds * audio.samplerate)
        if audio.frames < clip_frames:
            return False
        start = int((audio.frames - clip_frames) * fraction)
        audio.seek(start)
        samples = audio.read(clip_frames, dtype="float32", always_2d=True)
        if len(samples) != clip_frames:
            return False
        if np.max(np.abs(samples)) < 0.005:
            return False
        fade = min(int(0.02 * audio.samplerate), clip_frames // 10)
        samples[:fade] *= np.linspace(0, 1, fade, dtype="float32")[:, None]
        samples[-fade:] *= np.linspace(1, 0, fade, dtype="float32")[:, None]
        sf.write(target, samples, audio.samplerate, format="FLAC", subtype="PCM_16")
    return True


def main():
    train, holdout = select_sources()
    AUDIO.mkdir(parents=True, exist_ok=True)
    samples, manifest = [], []
    for split, sources in (("train", train), ("holdout", holdout)):
        for index, (path, row) in enumerate(sources, 1):
            manifest.append({"split": split, **row})
            print(f"{split}: {path}", flush=True)
            if split == "holdout":
                continue
            for part, fraction in enumerate((0.22, 0.65), 1):
                target = AUDIO / f"{path.parent.parent.name}_{index:02d}_{part}.flac"
                if extract_clip(path, target, fraction):
                    samples.append({
                        "audio_path": f"./{target.name}",
                        "caption": CAPTION,
                        "lyrics": "[Instrumental]",
                        "is_instrumental": True,
                        "genre": "Traditional Chinese guqin solo",
                    })
    with (AUDIO / "dataset.json").open("w", encoding="utf-8") as handle:
        json.dump(samples, handle, ensure_ascii=False, indent=2)
    with (DATA / "source_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Prepared {len(samples)} clips from {len(train)} tracks; {len(holdout)} held out", flush=True)


if __name__ == "__main__":
    main()
