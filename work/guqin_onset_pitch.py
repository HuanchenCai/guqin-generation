"""Attack-pitch metric: are the notes the player *plucks* in the mode?

Guqin semitones normally appear inside slides and vibrato (吟猱绰注), not at
the moment a note is plucked. So instead of all tonal energy, this looks only
at the pitch 30-120 ms after each detected onset and counts how many attacks
fall outside the pentatonic set of the passage.
"""

import numpy as np
import torch

from guqin_pitch_metric import (RATE, best_pentatonic, cent_histogram, load_mono, pitch_classes,
                                tuning_offset)

HOP = 512
N_FFT = 4096


def spectrum(mono):
    spec = torch.stft(torch.from_numpy(mono), N_FFT, HOP, window=torch.hann_window(N_FFT),
                      return_complex=True).abs().numpy()
    return spec, np.fft.rfftfreq(N_FFT, 1 / RATE)


def onsets(spec, freqs, min_gap=0.08):
    """Frame indices of plucks from positive log-spectral flux."""
    band = (freqs >= 80) & (freqs <= 6000)
    logmag = np.log1p(100 * spec[band])
    flux = np.clip(np.diff(logmag, axis=1), 0, None).sum(axis=0)
    flux = np.concatenate([[0], flux])
    window = int(0.5 * RATE / HOP)
    local = np.array([np.median(flux[max(0, i - window):i + window]) for i in range(len(flux))])
    thresh = local + 1.5 * flux.std()
    loud = spec[band].sum(axis=0) > spec[band].sum(axis=0).max() * 10 ** (-35 / 20)
    peaks, last = [], -10 ** 9
    for i in range(1, len(flux) - 1):
        if flux[i] > thresh[i] and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1] and loud[i]:
            if (i - last) * HOP / RATE >= min_gap:
                peaks.append(i)
                last = i
    return peaks


def attack_pitch_class(spec, freqs, frame, offset_cents):
    """Harmonic-summation pitch over MIDI 36-84 on the recording's tuning grid."""
    lo, hi = frame + int(0.03 * RATE / HOP), frame + int(0.12 * RATE / HOP) + 1
    col = spec[:, lo:hi].mean(axis=1)
    if col.max() <= 0:
        return None
    step = freqs[1] - freqs[0]
    best, best_note = -1.0, None
    for note in range(36, 85):
        f0 = 440.0 * 2 ** ((note - 69 + offset_cents / 100) / 12)
        sal = 0.0
        for h in range(1, 6):
            b = f0 * h / step
            if b + 1 >= len(col):
                break
            i = int(b)
            sal += 0.8 ** (h - 1) * max(col[i], col[i + 1])
        if sal > best:
            best, best_note = sal, note
    # Pitch classes are indexed from A (as in guqin_pitch_metric), not C.
    return (best_note - 69) % 12 if best_note is not None else None


def attack_report(path, context_seconds=10.0) -> dict:
    """Out-of-mode attack rate of the generated part (after context_seconds),
    using the mode and tuning of the opening, or of the clip itself if no opening."""
    mono = load_mono(path)
    ref = mono[: int(context_seconds * RATE)] if context_seconds else mono
    hist = cent_histogram(ref)
    offset = tuning_offset(hist)
    _, mode = best_pentatonic(pitch_classes(hist, offset))
    gen = mono[int(context_seconds * RATE):] if context_seconds else mono
    spec, freqs = spectrum(gen)
    frames = onsets(spec, freqs)
    pcs = [attack_pitch_class(spec, freqs, f, offset * 10) for f in frames]
    pcs = [p for p in pcs if p is not None]
    outside = sum(p not in set(mode.tolist()) for p in pcs)
    seconds = len(gen) / RATE
    return {"attacks": len(pcs), "attacks_per_s": round(len(pcs) / seconds, 2),
            "attack_out_of_mode": round(outside / len(pcs), 4) if pcs else None}
