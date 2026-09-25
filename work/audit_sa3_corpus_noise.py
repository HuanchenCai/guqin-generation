"""Measure a cautious high-frequency background-noise proxy for corpus triage."""

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parent / "sa3_overnight"


def db(value):
    return round(20 * math.log10(max(float(value), 1e-10)), 2)


def inspect(path):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    block_size = rate // 2
    blocks = mono[: len(mono) // block_size * block_size].reshape(-1, block_size)
    energies = np.sqrt(np.mean(blocks.astype(np.float64) ** 2, axis=1))
    quiet = blocks[np.argsort(energies)[: min(8, len(blocks))]].reshape(-1)
    spectrum = np.fft.rfft(quiet.astype(np.float64))
    freqs = np.fft.rfftfreq(len(quiet), 1 / rate)
    spectrum[freqs < 4000] = 0
    high = np.fft.irfft(spectrum, n=len(quiet))
    full_rms = np.sqrt(np.mean(mono.astype(np.float64) ** 2))
    high_rms = np.sqrt(np.mean(high.astype(np.float64) ** 2))
    quiet_rms = np.sqrt(np.mean(quiet.astype(np.float64) ** 2))
    return {"file": path.name, "full_rms_dbfs": db(full_rms),
            "quiet_rms_dbfs": db(quiet_rms), "quiet_highpass_dbfs": db(high_rms),
            "quiet_highpass_relative_db": db(high_rms / full_rms),
            "quiet_relative_db": db(quiet_rms / full_rms)}


def main():
    report = {}
    for group in ("small", "large"):
        rows = [inspect(path) for path in sorted((ROOT / f"{group}_audio").glob("*.flac"))]
        report[group] = rows
        values = [row["quiet_highpass_relative_db"] for row in rows]
        print(group, len(rows), "median", round(float(np.median(values)), 2),
              "range", min(values), max(values), flush=True)
    (ROOT / "noise_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
