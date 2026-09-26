"""Check the pentatonic-fit metric against the user's ratings before using it.

Collects every rated SA3-family clip from four listening rounds, computes the
metric, and asks: does it track scores, does it flag clips the user called
out for stray semitones, and would picking the best-fitting candidate within
each group beat picking at random?
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from guqin_pitch_metric import fit_report

WORK = Path(__file__).resolve().parent
OUTPUTS = WORK.parent / "outputs"
DOWNLOADS = Path.home() / "Downloads"
SEMITONE = re.compile("半音|中东|平均律|按错|练琴|跑调|转音|鬼畜")


def rated_clips() -> list[dict]:
    clips = []
    # Round 1: piece-group comparison (A/B/C), openings validation_0N.
    state = json.loads((DOWNLOADS / "古琴曲目分组统一评分.json").read_text(encoding="utf-8"))["ratings"]
    for key, value in state.items():
        source, seed, variant = key.split("|")
        index = int(source.split("_")[1].split(".")[0])
        clips.append({"round": "曲目分组", "path": OUTPUTS / "古琴曲目分组对照" / "audio" / f"v{index:02d}_{seed}_{variant}.flac",
                      "group": f"r1|{source}|{seed}", "score": value["score"], "comment": "", "context": 10})
    # Round 2: inference sweep.
    rows = {r["file"]: r for r in json.loads((OUTPUTS / "古琴推理调参" / "样本记录.json").read_text(encoding="utf-8"))}
    state = json.loads((DOWNLOADS / "古琴推理调参统一评分.json").read_text(encoding="utf-8"))["state"]
    for name, value in state.items():
        r = rows[name]
        clips.append({"round": "推理调参", "path": OUTPUTS / "古琴推理调参" / "audio" / name,
                      "group": f"r2|{r['opening']}|{r['seed']}", "score": int(value["score"]),
                      "comment": value["comment"] + " ".join(value["issues"]), "context": 10})
    # Round 3: caption test (continuations and text-only).
    state = json.loads((DOWNLOADS / "古琴文字标签统一评分.json").read_text(encoding="utf-8"))["state"]
    for name, value in state.items():
        text_only = name.startswith("t")
        clips.append({"round": "文字标签", "path": OUTPUTS / "古琴文字标签试听" / "audio" / name,
                      "group": f"r3|{name.split('_')[0]}", "score": int(value["score"]),
                      "comment": value["comment"] + " ".join(value["issues"]),
                      "context": None if text_only else 10})
    # Round 4: SA3 reference clips from the latent-LM page.
    state = json.loads((DOWNLOADS / "古琴潜空间续写统一评分.json").read_text(encoding="utf-8"))["state"]
    for name, value in state.items():
        if name.endswith("_sa3.flac"):
            clips.append({"round": "潜空间页SA3", "path": OUTPUTS / "古琴潜空间续写" / "audio" / name,
                          "group": f"r4|{name}", "score": int(value["score"]),
                          "comment": value["comment"] + " ".join(value["issues"]), "context": 10})
    return clips


def rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def spearman(a, b):
    return float(np.corrcoef(rank(np.array(a)), rank(np.array(b)))[0, 1])


def auc(pos, neg):
    """Probability a flagged clip has a LOWER fit than an unflagged one."""
    pairs = [(p < n) + 0.5 * (p == n) for p in pos for n in neg]
    return float(np.mean(pairs)) if pairs else float("nan")


def main() -> None:
    clips = rated_clips()
    for clip in clips:
        clip.update(fit_report(clip["path"], clip["context"]))
        clip["metric"] = clip.get("fit_to_opening_mode", clip["own_fit"])
        clip["semitone_flag"] = bool(SEMITONE.search(clip["comment"]))
    scores = [c["score"] for c in clips]
    print(f"{len(clips)} rated clips; {sum(c['semitone_flag'] for c in clips)} flagged for semitones")
    for key in ("metric", "own_fit"):
        print(f"Spearman(score, {key}) = {spearman(scores, [c[key] for c in clips]):+.3f}")
    for rnd in sorted({c["round"] for c in clips}):
        sub = [c for c in clips if c["round"] == rnd]
        print(f"  {rnd}: n={len(sub)} Spearman={spearman([c['score'] for c in sub], [c['metric'] for c in sub]):+.3f}")
    flagged = [c["metric"] for c in clips if c["semitone_flag"]]
    others = [c["metric"] for c in clips if not c["semitone_flag"]]
    print(f"semitone-flagged mean fit {np.mean(flagged):.3f} vs others {np.mean(others):.3f}; AUC {auc(flagged, others):.3f}")
    openings = [c["context_fit"] for c in clips if "context_fit" in c]
    print(f"real openings pentatonic fit: mean {np.mean(openings):.3f}, min {np.min(openings):.3f}")
    # Selection test within groups of >= 3 rated candidates.
    groups = defaultdict(list)
    for c in clips:
        groups[c["group"]].append(c)
    picked, avg, top = [], [], 0
    for g in groups.values():
        if len(g) < 3:
            continue
        best = max(g, key=lambda c: c["metric"])
        picked.append(best["score"])
        avg.append(np.mean([c["score"] for c in g]))
        top += best["score"] == max(c["score"] for c in g)
    print(f"selection over {len(picked)} groups: picked mean {np.mean(picked):.2f} vs random {np.mean(avg):.2f}; "
          f"picked a top-scored clip in {top}/{len(picked)}")
    out = [{k: (str(v) if isinstance(v, Path) else v) for k, v in c.items()} for c in clips]
    (WORK / "guqin_pitch_metric_validation.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
