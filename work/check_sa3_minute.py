"""Basic drift check for the one-pass SA3 minute sample."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft


OUT = Path(__file__).resolve().parent.parent / "outputs" / "古琴交叉实验"
x, rate = sf.read(OUT / "1995较新录音_SA3一分钟.wav", always_2d=True, dtype="float32")
x = x.mean(axis=1)
rows = []
for start in range(0, 60, 10):
    seg = x[start*rate : (start+10)*rate]
    _, _, z = stft(seg, fs=rate, nperseg=2048, noverlap=1024)
    mag = np.abs(z) + 1e-8
    rows.append({"start_seconds": start,
                 "rms": round(float(np.sqrt(np.mean(seg**2))), 4),
                 "flatness": round(float(np.mean(np.exp(np.mean(np.log(mag), axis=0)) / np.mean(mag, axis=0))), 4),
                 "clipped_fraction": round(float(np.mean(np.abs(seg) >= 0.999)), 5)})
(OUT / "一分钟信号检查.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(rows, ensure_ascii=False))
