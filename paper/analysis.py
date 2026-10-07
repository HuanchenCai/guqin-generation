"""Recompute every number, table and figure in the preprint from frozen data.

Inputs: paper/data (listening ratings, validation logs) and the sample
records under outputs/. Outputs: paper/generated/{numbers.json,*.tex,*.pdf}.
Run with the project venv: python paper/analysis.py
"""

import itertools
import json
import re
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "paper" / "data"
OUTS = ROOT / "outputs"
GEN = ROOT / "paper" / "generated"
GEN.mkdir(exist_ok=True)


def ratings(name: str) -> dict:
    blob = json.loads((DATA / "ratings" / f"{name}.json").read_text(encoding="utf-8"))
    state = blob.get("state") or blob.get("ratings")
    out = {}
    for key, value in state.items():
        if value.get("score") not in (None, ""):
            out[key] = {"score": int(value["score"]), "issues": value.get("issues", []),
                        "comment": value.get("comment", "")}
    return out


def mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


def paired_signflip(a, b):
    """Exact two-sided sign-flip permutation p-value for mean(a - b)."""
    d = np.array(a, float) - np.array(b, float)
    obs = abs(d.mean())
    flips = [abs((d * np.array(s)).mean()) for s in itertools.product((1, -1), repeat=len(d))]
    return float(np.mean([f >= obs - 1e-12 for f in flips]))


def bootstrap_ci(xs, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    xs = np.array(xs, float)
    means = rng.choice(xs, size=(n, len(xs)), replace=True).mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def flagged(entry, word):
    return any(word in i for i in entry["issues"]) or word in entry["comment"]


N = {}

# R1: piece-split comparison of masking strategies (validation openings).
r = ratings("古琴曲目分组统一评分")
by = {v: [x["score"] for k, x in r.items() if k.endswith("|" + v)] for v in "ABC"}
seed = {s: [x["score"] for k, x in r.items() if f"|{s}|" in k] for s in ("84917", "94017")}
N["R1"] = {"means": {k: mean(v) for k, v in by.items()}, "n": {k: len(v) for k, v in by.items()},
           "seed_means": {k: mean(v) for k, v in seed.items()}}

# R2: inference settings (openings from a performer seen by the 412 adapter).
r = ratings("古琴推理调参统一评分")
rec = {x["file"]: x for x in json.loads((OUTS / "古琴推理调参" / "样本记录.json").read_text(encoding="utf-8"))}
v2 = {}
groups = {}
for f, x in r.items():
    v2.setdefault(rec[f]["variant"], []).append(x["score"])
    groups.setdefault((rec[f]["opening"], rec[f]["seed"]), []).append(x["score"])
gm = [mean(g) for g in groups.values()]
N["R2"] = {"means": {k: mean(v) for k, v in v2.items()}, "n": {k: len(v) for k, v in v2.items()},
           "ones": {k: sum(s == 1 for s in v) for k, v in v2.items()},
           "group_mean_range": [min(gm), max(gm)], "groups": len(groups)}

# R3: text captions (unseen openings) and text-only generation.
r = ratings("古琴文字标签统一评分")
cont = {c: [x["score"] for k, x in r.items() if k.startswith("o") and k.endswith(f"_{c}.flac")]
        for c in ("old", "none", "match", "opposite")}
text = {c: [x["score"] for k, x in r.items() if k.startswith("t") and k.endswith(f"_{c}.flac")] for c in ("old", "new")}
N["R3"] = {"continuation": {k: mean(v) for k, v in cont.items()}, "continuation_n": {k: len(v) for k, v in cont.items()},
           "text_only": {k: mean(v) for k, v in text.items()}, "text_only_n": {k: len(v) for k, v in text.items()},
           "text_only_p": paired_signflip(text["new"], text["old"])}

# R4: from-scratch latent language model vs SA3 reference.
r = ratings("古琴潜空间续写统一评分")
lm = [x["score"] for k, x in r.items() if "sa3" not in k]
sa3 = [x["score"] for k, x in r.items() if "sa3" in k]
N["R4"] = {"latent_lm_mean": mean(lm), "latent_lm_n": len(lm), "sa3_mean": mean(sa3), "sa3_n": len(sa3),
           "latent_lm_all_ones": all(s == 1 for s in lm)}

# R5: longer training and mode captions on the 329-piece split (test openings).
r = ratings("古琴半音对照统一评分")
r5 = {v: [r[f"o{i:02d}_s7_{v}.flac"]["score"] for i in range(1, 9)] for v in "ABC"}
N["R5"] = {"means": {k: mean(v) for k, v in r5.items()},
           "semitone_flags": {v: sum(flagged(r[f"o{i:02d}_s7_{v}.flac"], "半音") for i in range(1, 9)) for v in "ABC"},
           "p_C_vs_A": paired_signflip(r5["C"], r5["A"])}

# R6 and R7: story-caption model, pass 2 then pass 14, against C (validation openings).
r6 = ratings("古琴故事版统一评分")
r7 = ratings("古琴第14遍对比统一评分")
p14 = [r7[f"o{i:02d}_pass14.flac"]["score"] for i in range(1, 9)]
p2 = [r7[f"o{i:02d}_pass2.flac"]["score"] for i in range(1, 9)]
c7 = [r7[f"o{i:02d}_C.flac"]["score"] for i in range(1, 9)]
s14 = [r7[f"s{k}_pass14.flac"]["score"] for k in range(1, 6)]
s2 = [r7[f"s{k}_pass2.flac"]["score"] for k in range(1, 6)]
N["R7"] = {"continuation": {"pass14": mean(p14), "pass2": mean(p2), "C": mean(c7)},
           "continuation_ci": {"pass14": bootstrap_ci(p14), "pass2": bootstrap_ci(p2), "C": bootstrap_ci(c7)},
           "p_pass14_vs_C": paired_signflip(p14, c7), "p_pass14_vs_pass2": paired_signflip(p14, p2),
           "scenes": {"pass14": mean(s14), "pass2": mean(s2)}, "p_scenes": paired_signflip(s14, s2),
           "flags": {v: {w: sum(flagged(r7[f"o{i:02d}_{v}.flac"], w) for i in range(1, 9)) for w in ("半音", "拉锯")}
                     for v in ("pass14", "pass2", "C")}}
N["R6"] = {"continuation": {"pass2": mean([r6[k]["score"] for k in r6 if k.endswith("_new.flac") and k.startswith("o")]),
                            "C": mean([r6[k]["score"] for k in r6 if k.endswith("_C.flac")])},
           "text_only_free_scene": r6.get("t06_new.flac", {}).get("score")}

# Test-retest: the same pass-2 and C files were rated in R6 and again in R7.
pairs = [(r6[f"o{i:02d}_{a}.flac"]["score"], r7[f"o{i:02d}_{b}.flac"]["score"])
         for i in range(1, 9) for a, b in (("new", "pass2"), ("C", "C")) if f"o{i:02d}_{a}.flac" in r6]
diffs = [abs(x - y) for x, y in pairs]
N["retest"] = {"pairs": len(pairs), "exact": sum(d == 0 for d in diffs), "within_one": sum(d <= 1 for d in diffs),
               "mean_abs_diff": mean(diffs)}

# Validation loss of the story-caption run, and of the from-scratch latent LM.
hist = json.loads((DATA / "story_run_history.json").read_text(encoding="utf-8"))
N["story_val"] = {str(h["pass"]): h["val"] for h in hist}
lm_log = [json.loads(l) for l in (DATA / "latent_lm_log.jsonl").read_text(encoding="utf-8").splitlines()]
lm_val = [(x["step"], x["val"]) for x in lm_log if "val" in x]
best = min(lm_val, key=lambda t: t[1])
N["latent_lm_val"] = {"best_step": best[0], "best": best[1], "final": lm_val[-1][1]}

# Overfitting check on the 329-piece split (B/C, three passes).
ov = {}
for line in (DATA / "overfit_validation.log").read_text(encoding="utf-8").splitlines():
    d = json.loads(line.split("VALIDATION ", 1)[1])
    ov[d["checkpoint"] or "base"] = d["val/avg_loss"]
N["overfit_329"] = {re.sub(r".*guqin_aligned_412/", "", k): v for k, v in ov.items()}

# Pitch metric validation against ratings.
pm = json.loads((DATA / "guqin_pitch_metric_validation.json").read_text(encoding="utf-8"))


def spearman(a, b):
    ra, rb = (np.argsort(np.argsort(x)).astype(float) for x in (a, b))
    return float(np.corrcoef(ra, rb)[0, 1])


def within_group(clips, key, sign=1):
    groups = {}
    for c in clips:
        if c.get(key) is not None:
            groups.setdefault(c["group"], []).append(c)
    ds, dm, pick, drop, rand = [], [], [], [], []
    for g in groups.values():
        if len(g) < 3:
            continue
        s = np.array([c["score"] for c in g], float)
        m = sign * np.array([c[key] for c in g], float)
        ds += list(s - s.mean())
        dm += list(m - m.mean())
        pick.append(s[np.argmax(m)])
        drop.append(np.delete(s, np.argmin(m)).mean())
        rand.append(s.mean())
    return {"within_group_spearman": spearman(ds, dm), "pick_best": mean(pick), "drop_worst": mean(drop),
            "random": mean(rand), "groups": len(pick)}


real_attack = json.loads((DATA / "real_windows_attack.json").read_text(encoding="utf-8"))
gen_attack = [c["attack_out_of_mode"] for c in pm if c.get("attack_out_of_mode") is not None]
N["pitch_metric"] = {
    "clips": len(pm),
    "spearman_own_fit": spearman([c["score"] for c in pm], [c["own_fit"] for c in pm]),
    "own_fit": within_group(pm, "own_fit"),
    "attack": within_group(pm, "attack_out_of_mode", sign=-1),
    "attack_out_of_mode": {"real_median": float(np.median([x["attack_out_of_mode"] for x in real_attack])),
                           "generated_mean": mean(gen_attack),
                           "flagged_mean": mean([c["attack_out_of_mode"] for c in pm
                                                 if c["semitone_flag"] and c.get("attack_out_of_mode") is not None])},
    "semitone_flagged": sum(c["semitone_flag"] for c in pm)}

# Signal-level "friction noise" metric vs the user's sawing tags (R2 clips).
r2 = ratings("古琴推理调参统一评分")
saw = [rec[f]["gen_noisy_fraction"] for f, x in r2.items() if flagged(x, "拉锯")]
no_saw = [rec[f]["gen_noisy_fraction"] for f, x in r2.items() if not flagged(x, "拉锯")]
N["noise_metric"] = {"tagged_n": len(saw), "tagged_mean": mean(saw), "untagged_mean": mean(no_saw),
                     "spearman_score": spearman([x["score"] for x in r2.values()],
                                                [rec[f]["gen_noisy_fraction"] for f in r2])}

(GEN / "numbers.json").write_text(json.dumps(N, ensure_ascii=False, indent=1), encoding="utf-8")

# Figure 1: validation loss of the story-caption run.
passes = [h["pass"] for h in hist]
vals = [h["val"] for h in hist]
fig, ax = plt.subplots(figsize=(4.6, 2.6))
ax.plot(passes, vals, marker="o", ms=3, lw=1.2, color="#2f5d50")
bp = min(hist, key=lambda h: h["val"])
ax.scatter([bp["pass"]], [bp["val"]], s=40, color="#c4572a", zorder=3, label=f"best (pass {bp['pass']})")
ax.set_xlabel("training pass (2500 steps each)")
ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
ax.set_ylabel("validation loss")
ax.legend(frameon=False, fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(GEN / "val_loss.pdf")
fig.savefig(GEN / "val_loss.png", dpi=200)  # README copy

# Figure 2: from-scratch latent LM validation loss (overfitting after 10k steps).
fig, ax = plt.subplots(figsize=(4.6, 2.6))
ax.plot([s / 1000 for s, _ in lm_val], [v for _, v in lm_val], lw=1.2, color="#2f5d50")
ax.axvline(best[0] / 1000, ls="--", lw=1, color="#c4572a")
ax.set_xlabel("training step (thousands)")
ax.set_ylabel("validation loss")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(GEN / "latent_lm_val.pdf")
fig.savefig(GEN / "latent_lm_val.png", dpi=200)  # README copy
print(json.dumps(N, ensure_ascii=False, indent=1))
