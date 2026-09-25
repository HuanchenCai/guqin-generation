"""Coarse signal checks for the matched guqin continuation samples."""

from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parent.parent / "outputs" / "古琴微调对照"
for file in sorted(ROOT.glob("*.wav")):
    audio, rate = sf.read(file)
    rows = []
    for start in (0, 10, 14, 18):
        seg = audio[int(start * rate) : int(min(start + 4, len(audio) / rate) * rate)]
        rms = float(np.sqrt(np.mean(seg ** 2)))
        clips = float(np.mean(np.abs(seg) > 0.95))
        frames = np.lib.stride_tricks.sliding_window_view(seg, 2048)[::1024]
        power = np.abs(np.fft.rfft(frames * np.hanning(2048), axis=-1)) + 1e-8
        flatness = float(np.mean(np.exp(np.mean(np.log(power), axis=-1)) / np.mean(power, axis=-1)))
        rows.append((start, round(rms, 4), round(clips, 4), round(flatness, 4)))
    print(file.name, rows)
