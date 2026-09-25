"""Basic signal sanity checks for the generated listening gallery."""

from pathlib import Path
import json
import torch
import torchaudio

RUN = Path(__file__).parent / "guqin_eval_v3"


def mono(path, target_rate=16000):
    audio, rate = torchaudio.load(str(path))
    audio = audio.mean(dim=0, keepdim=True)
    if rate != target_rate:
        audio = torchaudio.functional.resample(audio, rate, target_rate)
    return audio.squeeze(0)


def main():
    reference = mono(RUN / "reference_10s.wav")
    rows = []
    for path in sorted((RUN / "mp3").glob("*.mp3")):
        signal = mono(path)
        duration = len(signal) / 16000
        rms = float(torch.sqrt(torch.mean(signal.square())))
        peak = float(signal.abs().max())
        report = dict(file=path.name, duration_s=round(duration, 2), rms=round(rms, 4), peak=round(peak, 4))
        if "continuation" in path.name:
            n = min(len(reference), len(signal))
            x, y = reference[:n], signal[:n]
            x, y = x - x.mean(), y - y.mean()
            report["opening_correlation"] = round(float(torch.dot(x, y) / (torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y) + 1e-9)), 4)
            suffix = signal[10 * 16000:]
            report["generated_rms"] = round(float(torch.sqrt(torch.mean(suffix.square()))), 4) if len(suffix) else 0
        rows.append(report)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    (RUN / "signal_checks.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
