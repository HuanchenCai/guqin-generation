"""Check whether noise measurements separate the two listener-rejected artists."""

import csv
from pathlib import Path

import numpy as np


WORK = Path(__file__).resolve().parent
with (WORK / "guqin_noise_floor.csv").open(encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
bad = [row for row in rows if row["artist"] in {"张育瑾", "谢孝苹"}]
other = [row for row in rows if row["artist"] not in {"张育瑾", "谢孝苹"}]
print(f"Known noisy={len(bad)}; other={len(other)}")
for key in ("stationary_floor_vs_music_db", "low_stationary_vs_music_db",
            "mid_stationary_vs_music_db", "high_stationary_vs_music_db",
            "low_stationary_absolute_db", "mid_stationary_absolute_db",
            "high_stationary_absolute_db", "quiet_high_spectral_flatness",
            "hum_50_over_nearby_db", "hum_100_over_nearby_db"):
    a = np.array([float(row[key]) for row in bad])
    b = np.array([float(row[key]) for row in other])
    auc = float(np.mean(a[:, None] > b[None, :]) + .5 * np.mean(a[:, None] == b[None, :]))
    print(f"{key:36s} bad {np.median(a):7.2f} other {np.median(b):7.2f} AUC {auc:.3f}")
