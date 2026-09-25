"""Measure persistent recording noise from short, distributed WAV excerpts.

The measurements rank files for listening. They do not decide whether a
historical recording is suitable for training.
"""

import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf


WORK = Path(__file__).resolve().parent
ROOT = Path(r"Y:\Music\古琴曲")
SOURCE = WORK / "full_guqin_signal_scan.csv"
OUTPUT = WORK / "guqin_noise_floor.csv"
FRACTIONS = (0.04, 0.22, 0.40, 0.58, 0.76, 0.94)
SAMPLE_SECONDS = 4
FRAME_SECONDS = 0.5


def power_db(value: float) -> float:
    return round(10 * np.log10(max(float(value), 1e-16)), 2)


def measure(row: dict) -> dict:
    result = {"relative_path": row["relative_path"], "artist": row["artist"],
              "title": row["title"], "error": ""}
    path = ROOT / Path(row["relative_path"].replace("\\", "/"))
    try:
        with sf.SoundFile(path) as stream:
            rate = stream.samplerate
            count = min(SAMPLE_SECONDS * rate, stream.frames)
            max_start = max(stream.frames - count, 0)
            spots = []
            for fraction in FRACTIONS:
                stream.seek(int(max_start * fraction))
                spots.append(stream.read(count, dtype="float32", always_2d=True).mean(axis=1))
        audio = np.concatenate(spots)
        frame_size = int(rate * FRAME_SECONDS)
        frames = audio[: len(audio) // frame_size * frame_size].reshape(-1, frame_size)
        if len(frames) < 8:
            raise ValueError("Too few audio frames")
        window = np.hanning(frame_size).astype("float32")
        bins = np.fft.rfftfreq(frame_size, 1 / rate)
        spectrum = np.fft.rfft(frames * window, axis=1)
        powers = (np.abs(spectrum) ** 2).astype("float64")
        steady = np.percentile(powers, 20, axis=0)
        full = float(np.mean(audio.astype("float64") ** 2))
        total_steady = float(steady[(bins >= 30) & (bins <= 12000)].sum())
        bands = {"low": (30, 180), "mid": (180, 3000), "high": (3000, 12000)}
        result["stationary_floor_vs_music_db"] = power_db(total_steady / max(float(powers.mean(axis=0).sum()), 1e-16))
        for label, (low, high) in bands.items():
            mask = (bins >= low) & (bins < high)
            result[f"{label}_stationary_vs_music_db"] = power_db(
                steady[mask].sum() / max(float(powers.mean(axis=0)[mask].sum()), 1e-16))
            result[f"{label}_stationary_absolute_db"] = power_db(
                float(steady[mask].sum()) / max(float(powers.mean(axis=0).sum()), 1e-16) * full)
        quiet_frames = powers[np.argsort(np.mean(frames.astype("float64") ** 2, axis=1))[:max(6, len(frames) // 5)]]
        hf = quiet_frames[:, (bins >= 3000) & (bins < 12000)]
        flatness = np.exp(np.mean(np.log(hf + 1e-20), axis=1)) / (np.mean(hf, axis=1) + 1e-20)
        result["quiet_high_spectral_flatness"] = round(float(np.median(flatness)), 5)
        for frequency in (50, 100):
            peak = steady[(bins >= frequency - 2) & (bins <= frequency + 2)].max(initial=0)
            surround = steady[(bins >= frequency - 20) & (bins <= frequency + 20)
                              & ((bins < frequency - 4) | (bins > frequency + 4))]
            result[f"hum_{frequency}_over_nearby_db"] = power_db(peak / max(float(np.median(surround)), 1e-16))
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def main() -> None:
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    completed = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(measure, row): row for row in rows}
        for number, future in enumerate(as_completed(futures), 1):
            result = future.result()
            completed[result["relative_path"]] = result
            if number % 50 == 0 or number == len(rows):
                print(f"Measured {number}/{len(rows)}", flush=True)
    ordered = [completed[row["relative_path"]] for row in rows]
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ordered[0]))
        writer.writeheader()
        writer.writerows(ordered)
    summary = {"total": len(ordered), "errors": sum(bool(row["error"]) for row in ordered)}
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
