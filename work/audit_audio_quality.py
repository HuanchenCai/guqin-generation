import csv
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(r"Y:\Music\古琴曲")
MANIFEST = Path(__file__).parent / "full_library_metadata.csv"
OUT = Path(__file__).parent / "full_library_quality.csv"


def db(value):
    return round(20 * np.log10(max(float(value), 1e-8)), 1)


with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))

for index, row in enumerate(rows, 1):
    path = ROOT / row["relative_path"]
    try:
        with sf.SoundFile(path) as audio:
            windows = []
            for frac in (0.12, 0.36, 0.60, 0.84):
                start = max(0, min(audio.frames - audio.samplerate * 4, int(audio.frames * frac)))
                audio.seek(start)
                windows.append(audio.read(audio.samplerate * 4, dtype="float32", always_2d=True))
        x = np.concatenate(windows)
        mono = x.mean(axis=1)
        frame_length = 4410
        mono = mono[: len(mono) // frame_length * frame_length]
        frames = mono.reshape(-1, frame_length)
        rms = np.sqrt(np.mean(frames.astype("float64") ** 2, axis=1))
        low_idx = np.argsort(rms)[: max(8, int(len(rms) * 0.1))]
        quiet = frames[low_idx].reshape(-1)
        spectrum = np.abs(np.fft.rfft(quiet * np.hanning(len(quiet)))) ** 2
        freqs = np.fft.rfftfreq(len(quiet), 1 / 44100)
        quiet_hiss = spectrum[(freqs >= 6500) & (freqs <= 16000)].sum()
        quiet_total = spectrum[(freqs >= 40) & (freqs <= 16000)].sum()
        row.update(
            sampled_peak_dbfs=db(np.max(np.abs(x))),
            sampled_clip_pct=round(100 * np.mean(np.abs(x) >= 0.999), 5),
            quiet_rms_dbfs=db(np.percentile(rms, 10)),
            median_rms_dbfs=db(np.median(rms)),
            loud_rms_dbfs=db(np.percentile(rms, 90)),
            dynamic_proxy_db=round(db(np.percentile(rms, 90)) - db(np.percentile(rms, 10)), 1),
            quiet_hiss_ratio=round(float(quiet_hiss / max(quiet_total, 1e-20)), 5),
            stereo_correlation=round(float(np.corrcoef(x.T)[0, 1]), 4),
        )
    except Exception as exc:
        row["quality_error"] = str(exc)
    if index % 50 == 0:
        print(f"scanned {index}/{len(rows)}", flush=True)

fields = list(rows[0])
with OUT.open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print("output", OUT, flush=True)
