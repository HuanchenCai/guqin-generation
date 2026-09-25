"""Signal features for guqin friction ("sawing") noise in generated audio.

Guqin left-hand slides produce short broadband friction bursts. They are part
of the instrument, but generations that overuse them sound like sawing. These
features measure how much of the clip is noisy (flat-spectrum) high-band
energy versus tonal energy, so clips can be compared with their own prefix.
"""

import numpy as np
import soundfile as sf
import torch

RATE = 44100
N_FFT = 2048
HOP = 512


def load_mono(path, start=None, end=None):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    assert rate == RATE, rate
    audio = audio.mean(axis=1)
    lo = 0 if start is None else int(start * RATE)
    hi = len(audio) if end is None else int(end * RATE)
    return audio[lo:hi]


def features(mono: np.ndarray) -> dict:
    x = torch.from_numpy(mono)
    spec = torch.stft(x, N_FFT, HOP, window=torch.hann_window(N_FFT),
                      return_complex=True).abs().pow(2).numpy() + 1e-12
    freqs = np.fft.rfftfreq(N_FFT, 1 / RATE)
    hf = (freqs >= 2500) & (freqs < 10000)
    body = (freqs >= 60) & (freqs < 2500)
    hf_power = spec[hf].mean(axis=0)
    body_power = spec[body].mean(axis=0)
    total = spec[(freqs >= 60) & (freqs < 16000)].sum(axis=0)
    flat = np.exp(np.log(spec[hf]).mean(axis=0)) / spec[hf].mean(axis=0)
    loud = 10 * np.log10(total)
    active = loud > loud.max() - 45
    ratio_db = 10 * np.log10(hf_power / body_power)
    noisy = active & (flat > 0.25) & (ratio_db > -30)
    # Burst = onset of a noisy run; rate is per second of audio.
    onsets = np.sum(noisy[1:] & ~noisy[:-1])
    seconds = len(mono) / RATE
    # HF flux: frame-to-frame rise of noisy high-band energy (scratchiness).
    hf_db = 10 * np.log10(hf_power)
    flux = np.clip(np.diff(hf_db), 0, None)
    return {
        "hf_flatness": float(np.median(flat[active])) if active.any() else 0.0,
        "hf_to_body_db": float(np.median(ratio_db[active])) if active.any() else -99.0,
        "noisy_fraction": float(noisy[active].mean()) if active.any() else 0.0,
        "noise_bursts_per_s": float(onsets / seconds),
        "hf_flux": float(np.mean(flux[active[1:]])) if active[1:].any() else 0.0,
        "rms_dbfs": float(20 * np.log10(np.sqrt(np.mean(mono.astype(np.float64) ** 2)) + 1e-9)),
    }


def clip_report(path, context_seconds=10.0) -> dict:
    """Features of the generated part, and their difference from the prefix."""
    gen = features(load_mono(path, context_seconds))
    ctx = features(load_mono(path, 0, context_seconds))
    out = {f"gen_{k}": round(v, 4) for k, v in gen.items()}
    for key in ("hf_flatness", "hf_to_body_db", "noisy_fraction", "noise_bursts_per_s", "hf_flux"):
        out[f"delta_{key}"] = round(gen[key] - ctx[key], 4)
    return out
