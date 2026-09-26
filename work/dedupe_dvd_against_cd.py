"""Find DVD passages that duplicate the CD recordings already in training.

Both sides are reduced to 12-bin chroma at 5 frames per second. Every 30 s of
each DVD, a 12 s query is slid across all 412 training recordings (their full
coverage windows) with normalised cross-correlation; the same recording
scores near 1, different recordings stay well below.
"""

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

WORK = Path(__file__).resolve().parent
ROOT = WORK / "guqin_aligned_412"
SR, FPS, N_FFT = 11025, 5, 4096
HOP = SR // FPS
QUERY, STEP = 12 * FPS, 30 * FPS


def chroma(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    x = AF.resample(torch.from_numpy(audio.mean(axis=1)), rate, SR)
    spec = torch.stft(x, N_FFT, HOP, window=torch.hann_window(N_FFT), return_complex=True).abs().numpy()
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    band = (freqs >= 80) & (freqs <= 2000)
    pc = np.round(12 * np.log2(freqs[band] / 440.0)).astype(int) % 12
    out = np.zeros((12, spec.shape[1]), dtype=np.float32)
    np.add.at(out, pc, np.log1p(100 * spec[band]))
    out /= np.linalg.norm(out, axis=0, keepdims=True) + 1e-6
    return out.T                                        # (frames, 12)


def main() -> None:
    with (ROOT / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    cache = ROOT / "cd_chroma.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        seq, owner, offset = data["seq"], data["owner"], data["offset"]
    else:
        parts, owners, offsets = [], [], []
        for i, row in enumerate(manifest):
            c = chroma(ROOT / "audio" / row["file"])
            parts.append(c)
            owners.append(np.full(len(c), i))
            offsets.append(float(row["start_seconds"]) + np.arange(len(c)) / FPS)
            if i % 200 == 0:
                print(f"cd chroma {i}/{len(manifest)}", flush=True)
        seq, owner, offset = np.concatenate(parts), np.concatenate(owners), np.concatenate(offsets)
        np.savez(cache, seq=seq, owner=owner, offset=offset)
    X = torch.from_numpy(seq).double()
    n = QUERY * 12
    # Sliding sums for the per-window standard deviation of the CD sequence.
    s1 = torch.cat([torch.zeros(1), X.sum(1).cumsum(0)])
    s2 = torch.cat([torch.zeros(1), (X ** 2).sum(1).cumsum(0)])
    win_sum = s1[QUERY:] - s1[:-QUERY]
    win_sq = s2[QUERY:] - s2[:-QUERY]
    win_std = torch.sqrt((win_sq - win_sum ** 2 / n).clamp_min(1e-9))
    size = 1 << int(np.ceil(np.log2(len(X) + QUERY)))
    FX = torch.fft.rfft(X.T, n=size)
    results = []
    for dvd in sorted((WORK / "guqin_dvd_audio").glob("DVD*.flac")):
        q_all = chroma(dvd)
        for start in range(0, len(q_all) - QUERY, STEP):
            q = torch.from_numpy(q_all[start:start + QUERY]).double()
            q = (q - q.mean()) / (q.std() + 1e-9)
            # irfft(FX * conj(FQ))[s] = sum_t X[s+t] q[t], the sliding dot product.
            raw = torch.fft.irfft(FX * torch.fft.rfft(q.T, n=size).conj(), n=size).sum(0)[:len(win_std)]
            # Pearson r with a z-scored query: raw / (n * std_X) = raw / (sqrt(n) * win_std).
            r = raw / (np.sqrt(n) * win_std)
            best = int(torch.argmax(r))
            row = manifest[int(owner[best])]
            results.append({"dvd": dvd.stem, "dvd_seconds": start / FPS, "r": round(float(r[best]), 3),
                            "cd_artist": row["artist"], "cd_piece": row["piece"], "cd_source": row["source"],
                            "cd_seconds": round(float(offset[best]), 1)})
        print(dvd.stem, "done", flush=True)
    (WORK / "guqin_dvd_dedupe.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
