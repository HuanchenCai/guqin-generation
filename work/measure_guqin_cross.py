"""Objective checks for the codec and continuation cross experiment."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, stft


OUT = Path(__file__).resolve().parent.parent / "outputs" / "古琴交叉实验"
LABELS = ["1995较新录音", "未参与训练录音", "1991中期录音", "1988较旧录音"]


def read(path):
    x, sr = sf.read(path, always_2d=True, dtype="float32")
    x = x.mean(axis=1)
    if sr != 32000:
        x = resample_poly(x, 32000, sr)
    return x


def spec(x):
    _, _, z = stft(x, fs=32000, nperseg=2048, noverlap=1536)
    return np.abs(z)


def measure():
    rows = []
    for label in LABELS:
        reference = read(OUT / f"{label}_原曲开头.wav")
        a = spec(reference)
        a = a / max(float(np.sqrt(np.mean(a**2))), 1e-9)
        for codec in ("EnCodec", "DAC", "SAME"):
            candidate = read(OUT / f"{label}_{codec}还原.wav")[:len(reference)]
            b = spec(candidate)
            n = min(a.shape[-1], b.shape[-1])
            b = b[:, :n] / max(float(np.sqrt(np.mean(b[:, :n]**2))), 1e-9)
            error = float(np.linalg.norm(a[:, :n] - b) / np.linalg.norm(a[:, :n]))
            rows.append({"recording": label, "codec": codec, "spectral_convergence": round(error, 4)})
    (OUT / "还原频谱比较.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in rows:
        print(row["recording"], row["codec"], row["spectral_convergence"])


if __name__ == "__main__":
    measure()
