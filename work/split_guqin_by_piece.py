"""Reproducible random train/validation/test split grouped by musical piece."""

import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


WORK = Path(__file__).resolve().parent
OUT = WORK.parent / "outputs" / "古琴整曲筛选"
SEED = "guqin-piece-split-20260925-v1"


def piece_key(title: str) -> str:
    value = unicodedata.normalize("NFKC", title)
    value = re.sub(r"[\s《》〈〉()（）\[\]【】·•._]", "", value)
    while True:
        updated = re.sub(r"(?:[-－]?(?:\d+|残(?:本)?|节选|上|下))$", "", value)
        if updated == value:
            return value
        value = updated


def main() -> None:
    tracks = json.loads((OUT / "候选整曲清单.json").read_text(encoding="utf-8"))
    state = json.loads((WORK / "guqin_full_feedback_applied.json").read_text(encoding="utf-8"))["state"]
    selected = [row for row in tracks if state.get(row["id"], {}).get("decision") == "keep"]
    groups = defaultdict(list)
    for row in selected:
        groups[piece_key(row["title"])].append(row)
    ordered = sorted(groups, key=lambda name: hashlib.sha256((SEED + "/" + name).encode("utf-8")).hexdigest())
    total_seconds = sum(row["seconds"] for row in selected)
    split_names = ("train", "validation", "test")
    holdout_groups = round(len(groups) * .10)
    group_seconds = {name: sum(row["seconds"] for row in rows) for name, rows in groups.items()}
    group_tracks = {name: len(rows) for name, rows in groups.items()}
    rng = random.Random(SEED)
    best = None
    for _ in range(10000):
        draw = ordered.copy()
        rng.shuffle(draw)
        validation = set(draw[:holdout_groups])
        test = set(draw[holdout_groups:2 * holdout_groups])
        stats = [(sum(group_seconds[name] for name in pool),
                  sum(group_tracks[name] for name in pool)) for pool in (validation, test)]
        score = sum(((duration / total_seconds - .10) / .10) ** 2 +
                    ((count / len(selected) - .10) / .10) ** 2
                    for duration, count in stats)
        if best is None or score < best[0]:
            best = (score, validation, test)
    _, validation, test = best
    assigned = {name: "validation" if name in validation else "test" if name in test else "train"
                for name in groups}
    current = {split: sum(group_seconds[name] for name, value in assigned.items() if value == split)
               for split in split_names}
    rows = []
    for row in selected:
        rows.append({"分组": assigned[piece_key(row["title"])], "归一曲目": piece_key(row["title"]),
                     "演奏者": row["artist"], "曲名": row["title"],
                     "Y盘相对路径": row["id"], "时长秒": row["seconds"],
                     "原反馈来源": state[row["id"]]["origin"]})
    with (OUT / "按曲目随机分组.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"seed": SEED, "tracks": len(selected), "piece_groups": len(groups),
               "splits": {split: {"tracks": sum(row["分组"] == split for row in rows),
                                    "pieces": sum(value == split for value in assigned.values()),
                                    "hours": round(current[split] / 3600, 2)} for split in split_names},
               "piece_leakage": len({name for name, values in groups.items()
                                     if len({assigned[piece_key(row["title"])] for row in values}) > 1})}
    (OUT / "按曲目随机分组摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
