"""Check whole recordings from listener-approved source folders in one batch."""

import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf


WORK = Path(__file__).resolve().parent
OUTPUT = WORK.parent / "outputs" / "古琴录音筛选"
ROOT = Path(r"Y:\Music\古琴曲")
SCAN_DIR = WORK / "sa3_feedback_bulk"
JSONL = SCAN_DIR / "whole_track_scan.jsonl"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def source_folder(path: str) -> str:
    return str(Path(path.replace("\\", "/")).parent).replace("/", "\\")


def scan(row: dict) -> dict:
    result = {"relative_path": row["Y盘相对路径"], "artist": row["演奏者"],
              "title": row["曲名"], "old_tier": row["首轮层级"],
              "previous_flag": row["提示复听的指标"], "error": ""}
    path = ROOT / Path(row["Y盘相对路径"].replace("\\", "/"))
    rms_values = []
    clip_values = []
    peak_values = []
    dc_values = []
    try:
        with sf.SoundFile(path) as stream:
            result.update({"rate": stream.samplerate, "channels": stream.channels,
                           "subtype": stream.subtype, "duration_seconds": round(stream.frames / stream.samplerate, 2)})
            step = stream.samplerate * 5
            for block in stream.blocks(blocksize=step, dtype="float32", always_2d=True):
                whole = (len(block) // stream.samplerate) * stream.samplerate
                if whole == 0:
                    continue
                seconds = block[:whole].reshape(-1, stream.samplerate, stream.channels)
                rms_values.extend(np.sqrt(np.mean(seconds.astype("float64") ** 2, axis=(1, 2))).tolist())
                clip_values.extend(np.mean(np.abs(seconds) >= 0.999, axis=(1, 2)).tolist())
                peak_values.extend(np.max(np.abs(seconds), axis=(1, 2)).tolist())
                dc_values.extend(np.abs(np.mean(seconds.astype("float64"), axis=(1, 2))).tolist())
        rms = np.asarray(rms_values, dtype="float64")
        clips = np.asarray(clip_values, dtype="float64")
        peaks = np.asarray(peak_values, dtype="float64")
        dc = np.asarray(dc_values, dtype="float64")
        result.update({"rms_dbfs": round(float(20 * np.log10(max(np.sqrt(np.mean(rms ** 2)), 1e-9))), 2),
                       "peak": round(float(peaks.max()), 5),
                       "clip_fraction": round(float(clips.mean()), 7),
                       "seconds_with_clip_over_0_1pct": int(np.sum(clips > 0.001)),
                       "seconds_under_minus55db": int(np.sum(rms < 10 ** (-55 / 20))),
                       "max_dc": round(float(dc.max()), 5),
                       "one_second_rms": np.round(rms, 6).tolist(),
                       "one_second_clip": np.round(clips, 7).tolist()})
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def main() -> None:
    SCAN_DIR.mkdir(parents=True, exist_ok=True)
    approved = read_csv(OUTPUT / "首段听感可用录音.csv")
    catalog = read_csv(OUTPUT / "逐首筛选清单_听感更新.csv")
    folders = {row["演奏者"]: source_folder(row["Y盘相对路径"]) for row in approved}
    candidates = [row for row in catalog
                  if row["演奏者"] in folders
                  and row["听感后状态"] in {"代表片段已评可用：待全曲抽查", "同来源代表曲可用：本曲待复听"}]
    same_folder = [row for row in candidates if source_folder(row["Y盘相对路径"]) == folders[row["演奏者"]]]
    other_folder = [row for row in candidates if source_folder(row["Y盘相对路径"]) != folders[row["演奏者"]]]
    for label, rows in (("同来源目录候选", same_folder), ("跨目录候选", other_folder)):
        with (OUTPUT / f"{label}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    completed = {}
    if JSONL.exists():
        for line in JSONL.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            completed[item["relative_path"]] = item
    pending = [row for row in same_folder if row["Y盘相对路径"] not in completed]
    print(f"Approved-source pool={len(candidates)}, same-folder={len(same_folder)}, other-folder={len(other_folder)}, remaining={len(pending)}", flush=True)
    with JSONL.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(scan, row): row for row in pending}
            for number, future in enumerate(as_completed(futures), 1):
                result = future.result()
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                completed[result["relative_path"]] = result
                if number % 20 == 0 or number == len(pending):
                    print(f"Whole-track checked {number}/{len(pending)} new; errors={sum(bool(x['error']) for x in completed.values())}", flush=True)
    results = [completed[row["Y盘相对路径"]] for row in same_folder]
    summary_rows = [{key: value for key, value in row.items() if not key.startswith("one_second_")} for row in results]
    with (OUTPUT / "同来源整曲检查.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(json.dumps({"same_folder_checked": len(results),
                      "errors": sum(bool(row["error"]) for row in results),
                      "with_clipping_seconds": sum(row.get("seconds_with_clip_over_0_1pct", 0) > 0 for row in results)},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
