"""Detect gross silence, clipping, or noise drift before user listening."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft


OUT = Path(__file__).resolve().parent.parent / "outputs" / "古琴交叉实验"
LABELS = ["1995较新录音", "未参与训练录音", "1991中期录音", "1988较旧录音"]
MODELS = ["原版", "古琴微调", "SA3纯音频续写"]


def metrics(x, rate):
    rms = float(np.sqrt(np.mean(x*x)))
    _, _, z = stft(x, fs=rate, nperseg=2048, noverlap=1024)
    mag = np.abs(z) + 1e-8
    flatness = float(np.mean(np.exp(np.mean(np.log(mag), axis=0)) / np.mean(mag, axis=0)))
    return {"rms": round(rms, 4), "flatness": round(flatness, 4),
            "clipping_fraction": round(float(np.mean(np.abs(x) >= 0.999)), 5)}


def main():
    rows = []
    for label in LABELS:
        for model in MODELS:
            x, rate = sf.read(OUT / f"{label}_{model}.wav", always_2d=True, dtype="float32")
            x = x.mean(axis=1)
            context = metrics(x[: 10*rate], rate)
            continuation = metrics(x[10*rate :], rate)
            row = {"recording": label, "model": model, "seconds": round(len(x)/rate, 2),
                   "context": context, "continuation": continuation,
                   "rms_ratio": round(continuation["rms"] / max(context["rms"], 1e-9), 2)}
            rows.append(row)
            print(label, model, row["rms_ratio"], continuation["flatness"], flush=True)
    (OUT / "生成信号检查.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
