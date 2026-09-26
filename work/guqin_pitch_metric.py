"""Pentatonic-fit metric for guqin audio.

Most guqin pieces are pentatonic; listeners flag generations with stray
semitones ("半音", "像在练琴", "中东感"). This measures how much tonal energy
falls on the best-fitting pentatonic set.

Spectral peaks between 60 Hz and 2 kHz are folded into a 1200-cent
pitch-class histogram. Historical recordings are often not tuned to A440, so
the tuning offset is estimated first and the histogram is then folded to 12
semitone classes.
"""

import numpy as np
import soundfile as sf
import torch

RATE = 44100
N_FFT = 8192
HOP = 2048
PENTATONIC = np.array([0, 2, 4, 7, 9])


def load_mono(path, start=None, end=None):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    assert rate == RATE, rate
    audio = audio.mean(axis=1)
    lo = 0 if start is None else int(start * RATE)
    hi = len(audio) if end is None else int(end * RATE)
    return audio[lo:hi]


def cent_histogram(mono: np.ndarray, peaks_per_frame: int = 8) -> np.ndarray:
    """Magnitude-weighted histogram of spectral-peak pitch classes, 10-cent bins."""
    spec = torch.stft(torch.from_numpy(mono), N_FFT, HOP, window=torch.hann_window(N_FFT),
                      return_complex=True).abs().numpy()
    freqs = np.fft.rfftfreq(N_FFT, 1 / RATE)
    band = (freqs >= 60) & (freqs <= 2000)
    spec, freqs = spec[band], freqs[band]
    hist = np.zeros(120)
    frame_max = spec.max(axis=0)
    loud = frame_max > frame_max.max() * 10 ** (-40 / 20)
    for f in np.nonzero(loud)[0]:
        col = spec[:, f]
        is_peak = (col[1:-1] > col[:-2]) & (col[1:-1] >= col[2:]) & (col[1:-1] > col.max() * 0.1)
        idx = np.nonzero(is_peak)[0] + 1
        idx = idx[np.argsort(col[idx])[::-1][:peaks_per_frame]]
        for i in idx:
            # Parabolic interpolation for sub-bin frequency.
            a, b, c = np.log(col[i - 1] + 1e-12), np.log(col[i] + 1e-12), np.log(col[i + 1] + 1e-12)
            shift = 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
            freq = freqs[i] + shift * (freqs[1] - freqs[0])
            cents = (1200 * np.log2(freq / 440.0)) % 1200
            hist[int(cents // 10) % 120] += col[i]
    return hist


def tuning_offset(hist: np.ndarray) -> int:
    """Offset in 10-cent bins (0-9) that best aligns energy with semitone centres."""
    scores = [sum(hist[(k * 10 + o + d) % 120] for k in range(12) for d in (-1, 0, 1)) for o in range(10)]
    return int(np.argmax(scores))


def pitch_classes(hist: np.ndarray, offset: int) -> np.ndarray:
    pcs = np.array([sum(hist[(k * 10 + offset + d) % 120] for d in range(-5, 5)) for k in range(12)])
    return pcs / max(pcs.sum(), 1e-12)


def best_pentatonic(pcs: np.ndarray) -> tuple[float, np.ndarray]:
    fits = [(pcs[(PENTATONIC + r) % 12].sum(), (PENTATONIC + r) % 12) for r in range(12)]
    return max(fits, key=lambda x: x[0])


def fit_report(path, context_seconds: float | None = 10.0) -> dict:
    """Own pentatonic fit of the generated part; with an opening, also fit to
    the opening's pentatonic set (same tuning offset)."""
    if context_seconds:
        ctx_hist = cent_histogram(load_mono(path, 0, context_seconds))
        gen_hist = cent_histogram(load_mono(path, context_seconds))
        offset = tuning_offset(ctx_hist)
        ctx_pcs, gen_pcs = pitch_classes(ctx_hist, offset), pitch_classes(gen_hist, offset)
        own_fit, _ = best_pentatonic(gen_pcs)
        ctx_fit, ctx_set = best_pentatonic(ctx_pcs)
        return {"own_fit": round(float(own_fit), 4), "context_fit": round(float(ctx_fit), 4),
                "fit_to_opening_mode": round(float(gen_pcs[ctx_set].sum()), 4)}
    hist = cent_histogram(load_mono(path))
    pcs = pitch_classes(hist, tuning_offset(hist))
    own_fit, _ = best_pentatonic(pcs)
    return {"own_fit": round(float(own_fit), 4)}
