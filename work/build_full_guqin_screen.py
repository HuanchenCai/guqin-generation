"""Publish a conservative first-pass inventory of the full guqin WAV library."""

import csv
import json
from collections import Counter
from pathlib import Path


WORK = Path(__file__).resolve().parent
OUT = WORK.parent / "outputs" / "古琴夜间训练"
EXCLUDED_ARTISTS = {"张育瑾", "谢孝苹"}
NONSOLO_MARKERS = ("合奏", "琴箫", "琴簫", "琴歌", "唱弦", "伴奏")


def main() -> None:
    with (WORK / "full_guqin_signal_scan.csv").open(encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle))
    rows = []
    for item in source:
        duration = float(item["duration_sec"])
        clip = float(item["sampled_clip_fraction"])
        hf = float(item["quiet_hf_below_music_db"])
        gap = float(item["quiet_10pct_rms_dbfs"]) - float(item["sampled_rms_dbfs"])
        checks = []
        if clip > 0.001:
            checks.append("抽样削波>0.1%")
        if hf > -40:
            checks.append("安静处高频偏强")
        if gap > -7:
            checks.append("安静处底声相对偏高")
        if duration < 60:
            checks.append("不足1分钟")
        if item["artist"] in EXCLUDED_ARTISTS:
            status = "暂不训练：你指出该演奏者录音底噪大"
        elif any(marker in item["title"] for marker in NONSOLO_MARKERS):
            status = "暂不训练：标题标明合奏或伴奏"
        elif item["error"]:
            status = "暂不训练：文件读取失败"
        elif checks:
            status = "候选：建议优先复听"
        else:
            status = "候选：待抽听"
        rows.append({
            "筛选状态": status,
            "演奏者": item["artist"],
            "曲名": item["title"],
            "Y盘相对路径": item["relative_path"],
            "时长分钟": round(duration / 60, 2),
            "采样率Hz": 44100,
            "声道": 2,
            "PCM位深": 16,
            "提示复听的指标": "；".join(checks),
            "抽样削波比例": item["sampled_clip_fraction"],
            "安静处高频相对全曲dB": item["quiet_hf_below_music_db"],
            "安静处总声相对全曲dB": round(gap, 2),
        })
    rows.sort(key=lambda row: (
        {"候选：待抽听": 0, "候选：建议优先复听": 1}.get(row["筛选状态"], 2),
        row["演奏者"], row["曲名"],
    ))
    with (OUT / "全曲库粗筛清单.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row["筛选状态"] for row in rows)
    durations = {status: round(sum(row["时长分钟"] for row in rows if row["筛选状态"] == status) / 60, 2)
                 for status in counts}
    summary = {"total": len(rows), "counts": dict(counts), "hours": durations,
               "candidate_hours": round(sum(row["时长分钟"] for row in rows
                                            if row["筛选状态"].startswith("候选")) / 60, 2)}
    (WORK / "full_guqin_screen_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
