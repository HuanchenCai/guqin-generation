"""Merge listener feedback with the library and prepare a second source check."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from build_guqin_recording_screen import (CLIPS, OUTPUT, build_page,
                                          listen_priority, make_excerpt, read_csv)


WORK = Path(__file__).resolve().parent
FEEDBACK = WORK / "古琴录音筛选反馈_20260924.json"
HELDOUT_ARTIST = "蔡德允"


def second_track(tracks: list[dict], first_path: str, first_decision: str) -> dict | None:
    remaining = [row for row in tracks if row["Y盘相对路径"] != first_path]
    if not remaining:
        return None
    preferred = [row for row in remaining if row["首轮层级"] == "技术优先，待耳听"]
    if first_decision != "可用于训练":
        return min(preferred or remaining, key=listen_priority)
    secondary = [row for row in remaining if row["首轮层级"] != "技术优先，待耳听"]
    pool = secondary or preferred or remaining
    ranked = sorted(pool, key=listen_priority)
    return ranked[len(ranked) // 2]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    payload = json.loads(FEEDBACK.read_text(encoding="utf-8"))
    if payload.get("experiment") != "guqin_recording_screen_v1":
        raise ValueError("Unexpected feedback format")
    state = payload["state"]
    first_rows = read_csv(OUTPUT / "审听队列.csv")
    catalog = read_csv(OUTPUT / "逐首筛选清单.csv")
    by_path = {row["Y盘相对路径"]: row for row in catalog}
    first_by_artist = {}
    grouped = defaultdict(list)
    for row in catalog:
        if row["首轮层级"] != "暂不训练":
            grouped[row["演奏者"]].append(row)
    for row in first_rows:
        if row["已知噪声参照"] == "False":
            first_by_artist[row["演奏者"]] = row
    reviewed = []
    for path, answer in state.items():
        if path not in by_path:
            raise ValueError(f"Unknown feedback path: {path}")
        source = by_path[path]
        reviewed.append({"演奏者": source["演奏者"], "曲名": source["曲名"],
                         "Y盘相对路径": path, "录音判断": answer.get("decision", ""),
                         "问题标记": "；".join(answer.get("issues", [])),
                         "听感备注": answer.get("comment", ""),
                         "首轮层级": source["首轮层级"]})
    write_csv(OUTPUT / "首轮听感反馈.csv", reviewed)

    decisions = {row["演奏者"]: state[row["Y盘相对路径"]]["decision"]
                 for row in first_rows if row["已知噪声参照"] == "False"}
    possible = []
    for artist, first in first_by_artist.items():
        if artist == HELDOUT_ARTIST:
            continue
        decision = decisions[artist]
        next_row = second_track(grouped[artist], first["Y盘相对路径"], decision)
        if next_row is not None:
            order = {"不适合训练": 0, "有氛围，待处理": 1, "可用于训练": 2}.get(decision, 3)
            possible.append((order, -len(grouped[artist]), artist, first, next_row))
    possible.sort(key=lambda item: item[:3])
    second_rows = []
    for number, (_, _, artist, first, chosen) in enumerate(possible, len(first_rows) + 1):
        filename = f"{number:03d}.flac"
        target = CLIPS / filename
        if not target.exists():
            make_excerpt(chosen["Y盘相对路径"], target)
        answer = state[first["Y盘相对路径"]]
        previous = "；".join(filter(None, (answer.get("decision", ""),
                                          "、".join(answer.get("issues", [])),
                                          answer.get("comment", ""))))
        tracks = grouped[artist]
        second_rows.append({"演奏者": artist, "曲名": chosen["曲名"],
                            "Y盘相对路径": chosen["Y盘相对路径"],
                            "试听文件": f"试听片段/{filename}",
                            "该演奏者候选首数": len(tracks),
                            "该演奏者技术优先首数": sum(row["首轮层级"] == "技术优先，待耳听" for row in tracks),
                            "本曲技术优先": chosen["首轮层级"] == "技术优先，待耳听",
                            "已知噪声参照": False, "前一段评价": previous})
        if len(second_rows) % 20 == 0 or len(second_rows) == len(possible):
            print(f"Prepared {len(second_rows)}/{len(possible)} second excerpts", flush=True)
    write_csv(OUTPUT / "第二轮审听队列.csv", second_rows)

    second_paths = {row["Y盘相对路径"] for row in second_rows}
    first_paths = {row["Y盘相对路径"] for row in first_rows}
    updated = []
    for row in catalog:
        item = dict(row)
        path = row["Y盘相对路径"]
        artist = row["演奏者"]
        answer = state.get(path, {})
        decision = answer.get("decision", "")
        if artist == HELDOUT_ARTIST:
            status = "评估留出：整位演奏者不进训练"
        elif row["首轮层级"] == "暂不训练":
            status = "暂不训练：既定排除"
        elif decision == "可用于训练":
            status = "代表片段已评可用：待全曲抽查"
        elif decision == "不适合训练":
            status = "已试听不适合：曲目级排除"
        elif decision == "有氛围，待处理":
            status = "已试听有氛围：暂不进干净训练集"
        elif decisions[artist] == "可用于训练":
            status = "同来源代表曲可用：本曲待复听"
        elif decisions[artist] == "不适合训练":
            status = "同来源代表曲不适合：换曲复核"
        else:
            status = "同来源有氛围：本曲待复听"
        item.update({"听感后状态": status, "首轮本曲判断": decision,
                     "首轮问题标记": "；".join(answer.get("issues", [])),
                     "首轮听感备注": answer.get("comment", ""),
                     "第二轮是否排队": "是" if path in second_paths else "否",
                     "首轮是否代表曲": "是" if path in first_paths else "否"})
        updated.append(item)
    write_csv(OUTPUT / "逐首筛选清单_听感更新.csv", updated)
    confirmed = [row for row in updated if row["听感后状态"] == "代表片段已评可用：待全曲抽查"]
    write_csv(OUTPUT / "首段听感可用录音.csv", confirmed)
    build_page(first_rows, second_rows, state)
    summary = {"candidate_artists_reviewed": len(decisions),
               "first_decisions": dict(Counter(decisions.values())),
               "second_excerpts": len(second_rows),
               "second_first_rejected": sum(decisions[row["演奏者"]] == "不适合训练" for row in second_rows),
               "second_first_atmospheric": sum(decisions[row["演奏者"]] == "有氛围，待处理" for row in second_rows),
               "second_first_accepted": sum(decisions[row["演奏者"]] == "可用于训练" for row in second_rows),
               "confirmed_track_count": len(confirmed),
               "confirmed_hours": round(sum(float(row["时长分钟"]) for row in confirmed) / 60, 2),
               "updated_status_counts": dict(Counter(row["听感后状态"] for row in updated))}
    (WORK / "guqin_feedback_analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
