"""Second dedupe pass: DVD audio against the 226 CD tracks outside training.

Same chroma matcher as dedupe_dvd_against_cd.py. A DVD passage counts as a
duplicate only when consecutive queries (30 s apart) hit the same CD track
about 30 s apart too; single high-scoring hits were shown to be spurious.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from dedupe_dvd_against_cd import FPS, QUERY, STEP, chroma

WORK = Path(__file__).resolve().parent
LIBRARY = Path("Y:/Music/古琴曲")


def matcher(seq):
    X = torch.from_numpy(seq).double()
    n = QUERY * 12
    s1 = torch.cat([torch.zeros(1), X.sum(1).cumsum(0)])
    s2 = torch.cat([torch.zeros(1), (X ** 2).sum(1).cumsum(0)])
    std = torch.sqrt(((s2[QUERY:] - s2[:-QUERY]) - (s1[QUERY:] - s1[:-QUERY]) ** 2 / n).clamp_min(1e-9))
    size = 1 << int(np.ceil(np.log2(len(X) + QUERY)))
    FX = torch.fft.rfft(X.T, n=size)

    def best(q):
        q = torch.from_numpy(q).double()
        q = (q - q.mean()) / (q.std() + 1e-9)
        r = torch.fft.irfft(FX * torch.fft.rfft(q.T, n=size).conj(), n=size).sum(0)[:len(std)] / (np.sqrt(n) * std)
        i = int(torch.argmax(r))
        return i, float(r[i])
    return best


def consistent(hits):
    """DVD seconds whose neighbouring query lands on the same track ~30 s on."""
    hits = sorted(hits, key=lambda h: h["dvd_seconds"])
    dup = set()
    for a, b in zip(hits, hits[1:]):
        if a["cd_source"] == b["cd_source"] and abs((b["cd_seconds"] - a["cd_seconds"]) - STEP / FPS) <= 3:
            dup |= {a["dvd_seconds"], b["dvd_seconds"]}
    return dup


def main() -> None:
    with (WORK / "guqin_aligned_412" / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        trained = {r["source"] for r in csv.DictReader(handle)}
    with (WORK / "guqin_noise_floor.csv").open(encoding="utf-8-sig", newline="") as handle:
        rest = [r["relative_path"] for r in csv.DictReader(handle) if r["relative_path"] not in trained]
    cache = WORK / "guqin_aligned_412" / "cd_rest_chroma.npz"
    if cache.exists():
        data = np.load(cache)
        seq, owner, offset = data["seq"], data["owner"], data["offset"]
    else:
        parts, owners, offsets = [], [], []
        for i, rel in enumerate(rest):
            c = chroma(LIBRARY / rel)
            parts.append(c)
            owners.append(np.full(len(c), i))
            offsets.append(np.arange(len(c)) / FPS)
            if i % 25 == 0:
                print(f"library chroma {i}/{len(rest)}", flush=True)
        seq, owner, offset = np.concatenate(parts), np.concatenate(owners), np.concatenate(offsets)
        np.savez(cache, seq=seq, owner=owner, offset=offset)
    best = matcher(seq)
    first = json.loads((WORK / "guqin_dvd_dedupe.json").read_text(encoding="utf-8"))
    hits = []
    for dvd in sorted((WORK / "guqin_dvd_audio").glob("DVD*.flac")):
        q_all = chroma(dvd)
        for start in range(0, len(q_all) - QUERY, STEP):
            i, r = best(q_all[start:start + QUERY])
            hits.append({"dvd": dvd.stem, "dvd_seconds": start / FPS, "r": round(r, 3),
                         "cd_source": rest[int(owner[i])], "cd_seconds": round(float(offset[i]), 1)})
        print(dvd.stem, "done", flush=True)
    summary = {}
    for dvd in sorted({h["dvd"] for h in hits}):
        train_dup = consistent([h for h in first if h["dvd"] == dvd])
        rest_dup = consistent([h for h in hits if h["dvd"] == dvd])
        total = sum(h["dvd"] == dvd for h in hits)
        summary[dvd] = {"queries": total, "dup_training": sorted(train_dup), "dup_other_cd": sorted(rest_dup),
                        "other_cd_sources": sorted({h["cd_source"] for h in hits
                                                    if h["dvd"] == dvd and h["dvd_seconds"] in rest_dup})}
        print(dvd, total, "dup training", len(train_dup), "dup other CD", len(rest_dup), flush=True)
    (WORK / "guqin_dvd_dedupe_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                                        encoding="utf-8")


if __name__ == "__main__":
    main()
