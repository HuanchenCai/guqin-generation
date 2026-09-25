"""Create exact-length audio windows covering all 412 kept recordings."""

import csv
import hashlib
import json
import math
from pathlib import Path

import soundfile as sf


WORK = Path(__file__).resolve().parent
ROOT = Path(r"Y:\Music\古琴曲")
OUT = WORK / "guqin_aligned_412"
AUDIO = OUT / "audio"
CATALOG = WORK.parent / "outputs" / "古琴整曲筛选" / "候选整曲清单.json"
SPLIT = WORK.parent / "outputs" / "古琴整曲筛选" / "按曲目随机分组.csv"
SAMPLE_RATE = 44100
DOWNSAMPLE = 4096
WINDOW_FRAMES = (60 * SAMPLE_RATE // DOWNSAMPLE) * DOWNSAMPLE


def starts_for(frames: int) -> list[int]:
    if frames < WINDOW_FRAMES:
        raise ValueError(f"Recording shorter than one window: {frames} frames")
    count = math.ceil(frames / WINDOW_FRAMES)
    if count == 1:
        return [0]
    result = [round(i * (frames - WINDOW_FRAMES) / (count - 1)) for i in range(count)]
    if result[0] != 0 or result[-1] + WINDOW_FRAMES != frames:
        raise AssertionError("Source endpoints not covered")
    if any(right > left + WINDOW_FRAMES for left, right in zip(result, result[1:])):
        raise AssertionError("Uncovered source interval")
    return result


def main() -> None:
    AUDIO.mkdir(parents=True, exist_ok=True)
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    with SPLIT.open(encoding="utf-8-sig", newline="") as handle:
        assignments = {row["Y盘相对路径"]: row["分组"] for row in csv.DictReader(handle)}
    if len(assignments) != 412:
        raise RuntimeError("Split manifest does not contain exactly 412 source recordings")
    manifest = []
    for track_no, track in enumerate((row for row in catalog if row["id"] in assignments), 1):
        source = ROOT / Path(track["id"].replace("\\", "/"))
        with sf.SoundFile(source) as stream:
            if (stream.samplerate, stream.channels, stream.subtype) != (44100, 2, "PCM_16"):
                raise ValueError(f"Unexpected source format: {source}")
            positions = starts_for(stream.frames)
            prefix = hashlib.sha256(track["id"].encode("utf-8")).hexdigest()[:16]
            for index, start in enumerate(positions, 1):
                name = f"{prefix}_{index:03}.flac"
                target = AUDIO / name
                if not target.exists():
                    temp = AUDIO / f"{name}.part"
                    stream.seek(start)
                    data = stream.read(WINDOW_FRAMES, dtype="int16", always_2d=True)
                    if len(data) != WINDOW_FRAMES:
                        raise RuntimeError(f"Short read from {source}")
                    sf.write(temp, data, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
                    temp.replace(target)
                target.with_suffix(".txt").write_text("", encoding="utf-8")
                manifest.append({"file": name, "split": assignments[track["id"]],
                                 "piece": track["title"], "artist": track["artist"],
                                 "source": track["id"], "start_frame": start,
                                 "end_frame": start + WINDOW_FRAMES,
                                 "start_seconds": round(start / SAMPLE_RATE, 3),
                                 "end_seconds": round((start + WINDOW_FRAMES) / SAMPLE_RATE, 3)})
        if track_no % 25 == 0 or track_no == len(assignments):
            print(f"Ready {track_no}/412 recordings; {len(manifest)} aligned windows", flush=True)
    with (OUT / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    summary = {"source_recordings": len(assignments), "windows": len(manifest),
               "window_samples": WINDOW_FRAMES,
               "window_seconds": round(WINDOW_FRAMES / SAMPLE_RATE, 5),
               "split_windows": {split: sum(row["split"] == split for row in manifest)
                                 for split in ("train", "validation", "test")}}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
