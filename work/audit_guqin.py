"""Read-only audio quality inventory for the guqin collection."""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import soundfile as sf


def inspect_audio(path: Path) -> dict:
    info = sf.info(path)
    duration = info.frames / info.samplerate
    window = min(info.samplerate, info.frames)
    positions = np.array([0.2, 0.5, 0.8]) * max(0, info.frames - window)
    rms_db = []
    frame_rms_db = []
    peak = 0.0
    with sf.SoundFile(path) as audio:
        for position in positions:
            audio.seek(int(position))
            samples = audio.read(window, dtype="float32", always_2d=True)
            if samples.size == 0:
                continue
            mono = samples.mean(axis=1)
            rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
            rms_db.append(20 * np.log10(max(rms, 1e-8)))
            frame_length = max(1, info.samplerate // 20)
            for start in range(0, len(samples) - frame_length + 1, frame_length):
                frame = samples[start:start + frame_length]
                frame_rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
                frame_rms_db.append(20 * np.log10(max(frame_rms, 1e-8)))
            peak = max(peak, float(np.max(np.abs(samples))))
    return {
        "path": str(path),
        "duration_s": round(duration, 2),
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "subtype": info.subtype,
        "bytes": path.stat().st_size,
        "quietest_sampled_dbfs": round(min(rms_db), 1) if rms_db else "",
        "median_sampled_dbfs": round(float(np.median(rms_db)), 1) if rms_db else "",
        "interior_floor_dbfs": round(float(np.percentile(frame_rms_db, 10)), 1) if frame_rms_db else "",
        "interior_range_db": round(float(np.percentile(frame_rms_db, 90) - np.percentile(frame_rms_db, 10)), 1) if frame_rms_db else "",
        "sampled_peak_dbfs": round(20 * np.log10(max(peak, 1e-8)), 1),
        "error": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sample-per-volume", type=int, default=0)
    args = parser.parse_args()
    fields = [
        "path", "duration_s", "sample_rate", "channels", "subtype", "bytes",
        "quietest_sampled_dbfs", "median_sampled_dbfs", "interior_floor_dbfs",
        "interior_range_db", "sampled_peak_dbfs", "error",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        paths = sorted(args.root.rglob("*.wav"))
        if args.sample_per_volume:
            rng = random.Random(42)
            sampled = []
            for volume in sorted({p.relative_to(args.root).parts[0] for p in paths}):
                eligible = [
                    p for p in paths
                    if p.relative_to(args.root).parts[0] == volume
                    and not any(marker in p.stem for marker in ("合奏", "琴歌", "残"))
                    and 10_000_000 <= p.stat().st_size <= 140_000_000
                ]
                sampled.extend(rng.sample(eligible, min(args.sample_per_volume, len(eligible))))
            paths = sorted(sampled)
        print(f"Found {len(paths)} WAV files", flush=True)
        for index, path in enumerate(paths, 1):
            try:
                writer.writerow(inspect_audio(path))
            except Exception as exc:
                writer.writerow({"path": str(path), "error": str(exc)})
            handle.flush()
            if index % 25 == 0:
                print(f"Inspected {index}/{len(paths)}", flush=True)


if __name__ == "__main__":
    main()
