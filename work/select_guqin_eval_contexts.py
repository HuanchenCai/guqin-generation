"""Choose reproducible, audible source openings from piece-held-out recordings."""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


WORK = Path(__file__).resolve().parent
OUT = WORK.parent / "outputs" / "古琴整曲筛选"
SEED = "guqin-eval-contexts-20260925-v1"


def scans() -> dict:
    result = {}
    for path in (WORK / "sa3_feedback_bulk" / "whole_track_scan.jsonl",
                 WORK / "guqin_full_scan.jsonl"):
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                result[row["relative_path"]] = row
    return result


def pick_start(info: dict) -> int | None:
    rms = np.asarray(info["one_second_rms"], dtype=float)
    clipped = np.asarray(info["one_second_clip"], dtype=float)
    if len(rms) < 35:
        return None
    start_options = range(3, max(4, min(len(rms) - 11, 180)))
    options = []
    for start in start_options:
        active = rms[start:start+10]
        if len(active) != 10:
            continue
        level = 20 * math.log10(max(float(np.sqrt(np.mean(active ** 2))), 1e-8))
        clip = float(np.mean(clipped[start:start+10]))
        if -39 < level < -12 and clip < .0002:
            options.append((abs(level + 25), start))
    return min(options)[1] if options else None


def main() -> None:
    with (OUT / "按曲目随机分组.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    analysis = scans()
    selected = []
    for split in ("validation", "test"):
        for origin, count in (("明确评分", 2), ("用户指定：1–400 待判断均可用", 2)):
            pool = [row for row in rows if row["分组"] == split and row["原反馈来源"] == origin]
            pool.sort(key=lambda row: hashlib.sha256((SEED + row["Y盘相对路径"]).encode()).hexdigest())
            chosen_pieces = set()
            for row in pool:
                piece = row["归一曲目"]
                if piece in chosen_pieces:
                    continue
                info = analysis[row["Y盘相对路径"]]
                start = pick_start(info)
                if start is None:
                    continue
                selected.append({"分组": split, "反馈来源": origin, "归一曲目": piece,
                                 "演奏者": row["演奏者"], "曲名": row["曲名"],
                                 "Y盘相对路径": row["Y盘相对路径"],
                                 "前奏起点秒": start, "前奏长度秒": 10})
                chosen_pieces.add(piece)
                if len(chosen_pieces) == count:
                    break
            if len(chosen_pieces) != count:
                raise RuntimeError(f"Not enough audible contexts for {split}/{origin}")
    with (OUT / "评估开头清单.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    print(json.dumps({"contexts": len(selected), "validation": 4, "test": 4}, ensure_ascii=False))


if __name__ == "__main__":
    main()
