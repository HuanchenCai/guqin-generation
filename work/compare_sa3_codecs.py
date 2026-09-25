"""Compare reconstruction error and quiet-tail high-frequency energy."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft


OUT = Path(__file__).resolve().parent.parent / "outputs" / "古琴夜间训练"
SOURCES = ["较新录音", "未见录音"]
CODECS = [("DAC", "DAC还原.wav"), ("SAME-S", "SAME还原.wav"),
          ("SAME-L", "SAME-L还原.flac")]


def read(path):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != 44100:
        raise ValueError(path)
    return audio.mean(axis=1)


def log_spectral_distance(original, recovered, fft_size):
    _, _, base = stft(original, fs=44100, nperseg=fft_size, noverlap=fft_size * 3 // 4)
    _, _, test = stft(recovered, fs=44100, nperseg=fft_size, noverlap=fft_size * 3 // 4)
    shape = (min(base.shape[0], test.shape[0]), min(base.shape[1], test.shape[1]))
    base = base[: shape[0], : shape[1]]
    test = test[: shape[0], : shape[1]]
    # Suppress bins far below the audible source level so codec noise is not over-weighted.
    floor = max(float(np.max(np.abs(base))) * 0.0001, 1e-7)
    return round(float(np.mean(np.abs(20 * np.log10(np.maximum(np.abs(base), floor))
                                      - 20 * np.log10(np.maximum(np.abs(test), floor))))), 2)


def quiet_highpass_db(audio, selected, block_size):
    blocks = audio[: len(audio) // block_size * block_size].reshape(-1, block_size)
    selected = selected[selected < len(blocks)]
    quiet = blocks[selected].reshape(-1)
    spectrum = np.fft.rfft(quiet.astype(np.float64))
    spectrum[np.fft.rfftfreq(len(quiet), 1 / 44100) < 4000] = 0
    high = np.fft.irfft(spectrum, n=len(quiet))
    return 20 * np.log10(max(float(np.sqrt(np.mean(high * high))), 1e-10))


def main():
    rows = []
    block = 44100 // 4
    for source in SOURCES:
        original = read(OUT / f"{source}_原曲开头.wav")
        original_blocks = original[: len(original) // block * block].reshape(-1, block)
        rms = np.sqrt(np.mean(original_blocks.astype(np.float64) ** 2, axis=1))
        selected = np.argsort(rms)[: max(3, len(rms) // 10)]
        original_noise = quiet_highpass_db(original, selected, block)
        for codec, suffix in CODECS:
            reconstructed = read(OUT / f"{source}_{suffix}")
            length = min(len(original), len(reconstructed))
            original_part = original[:length]
            reconstructed = reconstructed[:length]
            row = {"source": source, "codec": codec,
                   "log_spectral_distance_2048_db": log_spectral_distance(original_part, reconstructed, 2048),
                   "log_spectral_distance_8192_db": log_spectral_distance(original_part, reconstructed, 8192),
                   "quiet_highpass_change_db": round(quiet_highpass_db(reconstructed, selected, block) - original_noise, 2)}
            rows.append(row)
            print(row, flush=True)
    (OUT / "音色还原客观检查.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
