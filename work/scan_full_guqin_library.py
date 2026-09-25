"""Sample the full guqin WAV library for coarse recording-quality features."""

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf


HERE = Path(__file__).resolve().parent
ROOT = Path(r"Y:\Music\古琴曲")
CATALOG = HERE / "full_library_metadata.csv"
JSONL = HERE / "full_guqin_signal_scan.jsonl"
CSV = HERE / "full_guqin_signal_scan.csv"
EXCLUDED = {"张育瑾", "谢孝苹"}
FRACTIONS = (0.04, 0.22, 0.40, 0.58, 0.76, 0.94)
SECONDS_PER_SPOT = 4.0
FRAME_SECONDS = 0.25


def db(rms: float) -> float:
    return round(20 * np.log10(max(float(rms), 1e-9)), 2)


def scan(row: dict) -> dict:
    rel = row["relative_path"]
    path = ROOT / Path(rel.replace("\\", "/"))
    artist = row["name"].split("-", 1)[0]
    result = {"relative_path": rel, "artist": artist,
              "title": row["name"].split("-", 1)[-1].removesuffix(".wav"),
              "duration_sec": float(row["duration_sec"]),
              "artist_excluded": artist in EXCLUDED}
    try:
        with sf.SoundFile(path) as stream:
            rate = stream.samplerate
            if rate != 44100:
                raise ValueError(f"Unexpected sample rate {rate}")
            count = min(int(SECONDS_PER_SPOT * rate), stream.frames)
            max_start = max(stream.frames - count, 0)
            spots = []
            for fraction in FRACTIONS:
                stream.seek(int(max_start * fraction))
                samples = stream.read(count, dtype="float32", always_2d=True)
                spots.append(samples.mean(axis=1))
        audio = np.concatenate(spots)
        frame_size = int(FRAME_SECONDS * rate)
        audio = audio[: len(audio) // frame_size * frame_size]
        frames = audio.reshape(-1, frame_size)
        rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
        quiet_indices = np.argsort(rms)[:max(6, int(0.1 * len(rms)))]
        window = np.hanning(frame_size)
        spectrum = np.fft.rfft(frames[quiet_indices] * window, axis=1)
        highband = np.fft.rfftfreq(frame_size, d=1 / rate) >= 4000
        hf = np.fft.irfft(spectrum * highband, n=frame_size, axis=1)
        hf_rms = np.sqrt(np.mean(hf**2)) / np.sqrt(np.mean(window**2))
        quiet_rms = np.sqrt(np.mean(frames[quiet_indices].astype(np.float64) ** 2))
        full_rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
        result.update({
            "sampled_seconds": round(len(audio) / rate, 2),
            "sampled_rms_dbfs": db(full_rms),
            "quiet_10pct_rms_dbfs": db(quiet_rms),
            "quiet_4khz_plus_dbfs": db(hf_rms),
            "quiet_hf_below_music_db": round(db(hf_rms) - db(full_rms), 2),
            "quiet_hf_below_quiet_db": round(db(hf_rms) - db(quiet_rms), 2),
            "silent_frame_fraction": round(float(np.mean(rms < 10 ** (-55 / 20))), 4),
            "sampled_clip_fraction": round(float(np.mean(np.abs(audio) >= 0.999)), 6),
            "sampled_peak": round(float(np.max(np.abs(audio))), 5),
            "error": "",
        })
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    with CATALOG.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit:
        rows = rows[:args.limit]
    completed = {}
    if JSONL.exists():
        for line in JSONL.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            completed[item["relative_path"]] = item
    todo = [row for row in rows if row["relative_path"] not in completed]
    print(f"Catalog={len(rows)} done={len(completed)} todo={len(todo)}", flush=True)
    with JSONL.open("a", encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(scan, row): row for row in todo}
            for number, future in enumerate(as_completed(futures), 1):
                result = future.result()
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
                completed[result["relative_path"]] = result
                if number % 50 == 0 or number == len(todo):
                    print(f"Scanned {number}/{len(todo)} new, errors="
                          f"{sum(bool(item.get('error')) for item in completed.values())}", flush=True)
    ordered = [completed[row["relative_path"]] for row in rows]
    fields = list(ordered[0])
    for item in ordered:
        for field in item:
            if field not in fields:
                fields.append(field)
    with CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)
    print(f"Wrote {CSV} with {len(ordered)} tracks", flush=True)


if __name__ == "__main__":
    main()
