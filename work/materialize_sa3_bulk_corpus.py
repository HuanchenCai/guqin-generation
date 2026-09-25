"""Cut one signal-checked training minute from each approved-source recording."""

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf


WORK = Path(__file__).resolve().parent
ROOT = Path(r"Y:\Music\古琴曲")
SCAN = WORK / "sa3_feedback_bulk" / "whole_track_scan.jsonl"
OUT = WORK / "sa3_feedback_bulk" / "audio"
PUBLISHED = WORK.parent / "outputs" / "古琴录音筛选" / "批量训练片段清单.csv"
SECONDS = 60


def choose_window(record: dict) -> dict | None:
    rms = np.asarray(record["one_second_rms"], dtype="float64")
    clips = np.asarray(record["one_second_clip"], dtype="float64")
    if len(rms) < SECONDS:
        return None
    kernel = np.ones(SECONDS, dtype="float64") / SECONDS
    rolling_rms = np.sqrt(np.convolve(rms ** 2, kernel, mode="valid"))
    rolling_clip = np.convolve(clips, kernel, mode="valid")
    rolling_quiet = np.convolve((rms < 10 ** (-55 / 20)).astype("float64"), kernel, mode="valid")
    db = 20 * np.log10(np.maximum(rolling_rms, 1e-9))
    score = rolling_clip * 10000 + rolling_quiet * 10 + np.abs(db + 22) / 12
    score[:min(3, len(score))] += .3
    safe = (rolling_clip < 0.0002) & (rolling_quiet <= .2) & (db > -36) & (db < -8)
    if not safe.any():
        return None
    start = int(np.argmin(np.where(safe, score, np.inf)))
    return {"start_seconds": start, "duration_seconds": SECONDS,
            "rms_dbfs": round(float(db[start]), 2),
            "clipped_fraction": round(float(rolling_clip[start]), 7),
            "quiet_second_fraction": round(float(rolling_quiet[start]), 3)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in SCAN.read_text(encoding="utf-8").splitlines()]
    rows.sort(key=lambda row: (row["artist"], row["title"], row["relative_path"]))
    results = []
    for number, record in enumerate(rows, 1):
        row = {"演奏者": record["artist"], "曲名": record["title"],
               "Y盘相对路径": record["relative_path"],
               "原先技术层级": record["old_tier"],
               "原先复听提示": record["previous_flag"]}
        if record["error"]:
            row.update({"处理": "暂缓：文件错误", "原因": record["error"]})
        else:
            window = choose_window(record)
            if window is None:
                row.update({"处理": "暂缓：未找到合适一分钟", "原因": "每个一分钟窗口均有削波、长静音或异常电平"})
            else:
                source = ROOT / Path(record["relative_path"].replace("\\", "/"))
                with sf.SoundFile(source) as stream:
                    stream.seek(window["start_seconds"] * stream.samplerate)
                    audio = stream.read(SECONDS * stream.samplerate, dtype="float32", always_2d=True)
                    if len(audio) != SECONDS * stream.samplerate:
                        raise ValueError(f"Short read from {source}")
                    filename = f"guqin_{number:03d}.flac"
                    target = OUT / filename
                    if not target.exists():
                        sf.write(target, audio, stream.samplerate, format="FLAC", subtype="PCM_16")
                    target.with_suffix(".txt").write_text("", encoding="utf-8")
                row.update({"处理": "本轮训练", "原因": "", "片段文件": filename, **window})
        results.append(row)
        if number % 25 == 0 or number == len(rows):
            print(f"Prepared {number}/{len(rows)}; selected={sum(item['处理'] == '本轮训练' for item in results)}", flush=True)
    columns = ["演奏者", "曲名", "Y盘相对路径", "原先技术层级", "原先复听提示", "处理", "原因",
               "片段文件", "start_seconds", "duration_seconds", "rms_dbfs", "clipped_fraction", "quiet_second_fraction"]
    with PUBLISHED.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    summary = {"source_tracks": len(results), "selected_tracks": sum(row["处理"] == "本轮训练" for row in results),
               "selected_artists": len({row["演奏者"] for row in results if row["处理"] == "本轮训练"}),
               "minutes": sum(row["处理"] == "本轮训练" for row in results),
               "held_for_signal_quality": sum(row["处理"] != "本轮训练" for row in results)}
    (WORK / "sa3_feedback_bulk" / "corpus_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
