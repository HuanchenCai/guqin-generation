"""Summarize objective properties of the ten confirmed solo-guqin recordings."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parent
IN = ROOT / "guqin_v2" / "audio"
REPORT = ROOT / "guqin_v2" / "clip_report.csv"
OUT = ROOT.parent / "outputs" / "古琴交叉实验"


def db(x):
    return round(20 * np.log10(max(x, 1e-9)), 2)


def metrics(path):
    audio, rate = sf.read(path, always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)
    n = rate // 2
    chunks = mono[: len(mono) // n * n].reshape(-1, n)
    rms = np.sqrt(np.mean(chunks**2, axis=1))
    quiet_indices = np.argsort(rms)[: max(1, len(rms) // 10)]
    quiet = chunks[quiet_indices].reshape(-1)
    spec = np.abs(np.fft.rfft(quiet[: min(len(quiet), 131072)])) ** 2
    freq = np.fft.rfftfreq(min(len(quiet), 131072), 1 / rate)
    high_ratio = float(spec[freq >= 8000].sum() / max(spec.sum(), 1e-12))
    return {"sample_rate": rate, "channels": audio.shape[1], "duration": round(len(audio) / rate, 2),
            "rms_dbfs": db(float(np.sqrt(np.mean(mono**2)))),
            "quietest_10pct_frames_dbfs": db(float(np.sqrt(np.mean(rms[quiet_indices] ** 2)))),
            "dynamic_range_db": round(db(float(np.percentile(rms, 90))) - db(float(np.percentile(rms, 10))), 2),
            "high_frequency_fraction_in_quiet_frames": round(high_ratio, 5),
            "clipped_fraction": round(float(np.mean(np.abs(mono) >= 0.999)), 6)}


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    source_rows = defaultdict(list)
    with REPORT.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["accepted"].lower() == "true" and (IN / row["clip"]).exists():
                source_rows[row["source"]].append(row)
    summary = []
    for source, rows in source_rows.items():
        measurements = [metrics(IN / row["clip"]) for row in rows]
        first = measurements[0]
        summary.append({"recording": source, "date": rows[0]["recording_date"], "clip_count": len(rows),
                        "duration_minutes": round(sum(m["duration"] for m in measurements) / 60, 2),
                        "sample_rate": first["sample_rate"], "channels": first["channels"],
                        "median_rms_dbfs": round(float(np.median([m["rms_dbfs"] for m in measurements])), 2),
                        "median_quiet_frame_dbfs": round(float(np.median([m["quietest_10pct_frames_dbfs"] for m in measurements])), 2),
                        "median_dynamic_range_db": round(float(np.median([m["dynamic_range_db"] for m in measurements])), 2),
                        "median_high_freq_quiet_ratio": round(float(np.median([m["high_frequency_fraction_in_quiet_frames"] for m in measurements])), 5),
                        "max_clipped_fraction": max(m["clipped_fraction"] for m in measurements)})
    summary.sort(key=lambda x: (x["date"], x["recording"]))
    (OUT / "素材质量统计.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"recordings": len(summary), "clips": sum(x["clip_count"] for x in summary),
                      "total_minutes": round(sum(x["duration_minutes"] for x in summary), 2)}, ensure_ascii=False))
    for row in summary:
        print(row["date"], Path(row["recording"]).name, row["median_quiet_frame_dbfs"],
              row["median_high_freq_quiet_ratio"])


if __name__ == "__main__":
    main()
