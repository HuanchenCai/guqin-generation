"""Correlate user ratings of the piece-level comparison with noise features."""

import json
import sys
from pathlib import Path

import numpy as np

from guqin_noise_features import clip_report

WORK = Path(__file__).resolve().parent
PAGE = WORK.parent / "outputs" / "古琴曲目分组对照"


def rank(values):
    order = np.argsort(np.argsort(values, kind="stable"), kind="stable").astype(float)
    return order


def spearman(a, b):
    return float(np.corrcoef(rank(np.array(a)), rank(np.array(b)))[0, 1])


def main() -> None:
    ratings = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["ratings"]
    rows = json.loads((PAGE / "样本记录.json").read_text(encoding="utf-8"))
    table = []
    for row in rows:
        key = f"{row['source']}|{row['seed']}|{row['variant']}"
        score = (ratings.get(key) or {}).get("score")
        if not score:
            continue
        table.append({"key": key, "variant": row["variant"], "score": score,
                      **clip_report(PAGE / "audio" / row["file"])})
    scores = [r["score"] for r in table]
    print("variant means:", {v: round(np.mean([r["score"] for r in table if r["variant"] == v]), 2)
                             for v in "ABC"})
    print("seed means:", {s: round(np.mean([r["score"] for r in table if f"|{s}|" in r["key"]]), 2)
                          for s in (84917, 94017)})
    print("\nSpearman(score, feature):")
    for feature in [k for k in table[0] if k.startswith(("gen_", "delta_"))]:
        print(f"  {feature:28s} {spearman(scores, [r[feature] for r in table]):+.2f}")
    print()
    for r in sorted(table, key=lambda r: r["score"]):
        print(r["score"], r["key"], "noisy", r["gen_noisy_fraction"], "d_noisy", r["delta_noisy_fraction"],
              "bursts", r["gen_noise_bursts_per_s"], "d_flat", r["delta_hf_flatness"], "rms", r["gen_rms_dbfs"])
    (WORK / "guqin_rating_noise_analysis.json").write_text(
        json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
