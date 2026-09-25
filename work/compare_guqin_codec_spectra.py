"""Descriptive multi-resolution spectral error for the two codec roundtrips."""

import json
from pathlib import Path

import torch
import torchaudio

ROOT = Path(__file__).parent.parent / "outputs" / "古琴音色诊断"
FILES = {
    "MusicGen EnCodec 32k": ROOT / "02_MusicGen编码还原.wav",
    "DAC 44.1k": ROOT / "03_DAC44k编码还原.wav",
}


def load(path):
    audio, rate = torchaudio.load(str(path))
    audio = audio.mean(0)
    if rate != 44100:
        audio = torchaudio.functional.resample(audio, rate, 44100)
    return audio


def main():
    reference = load(ROOT / "01_原曲_1995古琴独奏.wav")
    rows = []
    for name, path in FILES.items():
        reconstructed = load(path)[:len(reference)]
        reference_crop = reference[:len(reconstructed)]
        reconstructed = reconstructed * (reference_crop.square().mean() / reconstructed.square().mean()).sqrt()
        errors = []
        for fft in (1024, 2048, 8192):
            window = torch.hann_window(fft)
            a = torch.stft(reference_crop, fft, hop_length=fft // 4, window=window, return_complex=True).abs()
            b = torch.stft(reconstructed, fft, hop_length=fft // 4, window=window, return_complex=True).abs()
            errors.append(round(float((torch.log1p(a) - torch.log1p(b)).abs().mean()), 4))
        rows.append({"codec": name, "mean_log_spectral_errors": errors})
    (ROOT / "频谱差异_仅供参考.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
