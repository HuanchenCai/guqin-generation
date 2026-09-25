"""Merge explicit full-recording votes with the user's 1–400 default-use rule."""

import csv
import json
from collections import Counter
from pathlib import Path


WORK = Path(__file__).resolve().parent
OUT = WORK.parent / "outputs" / "古琴整曲筛选"
CATALOG = OUT / "候选整曲清单.json"
FEEDBACK = WORK / "guqin_full_feedback_original.json"
APPLIED = WORK / "guqin_full_feedback_applied.json"
LIST = OUT / "整曲筛选合并结果.csv"


def main() -> None:
    tracks = json.loads(CATALOG.read_text(encoding="utf-8"))
    original = json.loads(FEEDBACK.read_text(encoding="utf-8"))
    explicit = original["state"]
    if len(tracks) != 571 or sum(t["priority"] != "其他候选" for t in tracks[:400]) != 400:
        raise RuntimeError("Candidate ordering changed; review default 1–400 rule")
    merged = {}
    rows = []
    for index, track in enumerate(tracks, 1):
        answer = explicit.get(track["id"], {})
        decision = answer.get("decision", "")
        origin = "明确评分" if decision else "未判断"
        if index <= 400 and not decision:
            decision = "keep"
            origin = "用户指定：1–400 待判断均可用"
        if decision:
            merged[track["id"]] = {**answer, "decision": decision, "origin": origin}
        rows.append({"序号": index, "演奏者": track["artist"], "曲名": track["title"],
                     "Y盘相对路径": track["id"], "来源组": track["priority"],
                     "原始判断": answer.get("decision", ""), "最终判断": decision,
                     "判断依据": origin, "备注": answer.get("comment", ""),
                     "可用时间段": answer.get("ranges", "")})
    with LIST.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    APPLIED.write_text(json.dumps({"experiment": "guqin_full_recordings_v1",
                                   "merge_rule": "1–400 pending => keep",
                                   "state": merged}, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = Counter(row["最终判断"] for row in rows)
    heldout = sum(row["演奏者"] == "蔡德允" and row["最终判断"] == "keep" for row in rows)
    summary = {"total": len(rows), "explicit": sum(row["判断依据"] == "明确评分" for row in rows),
               "assumed_usable": sum(row["判断依据"] == "用户指定：1–400 待判断均可用" for row in rows),
               "final": dict(counts), "kept_holdout_artist": heldout,
               "kept_for_training_after_holdout": counts["keep"] - heldout}
    (OUT / "整曲筛选合并摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
