"""Inspect every solo candidate in full, reusing completed track scans."""

import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scan_sa3_bulk_sources import scan


WORK = Path(__file__).resolve().parent
OUT = WORK.parent / "outputs" / "古琴录音筛选"
CATALOG = OUT / "逐首筛选清单_听感更新.csv"
OLD = WORK / "sa3_feedback_bulk" / "whole_track_scan.jsonl"
NEW = WORK / "guqin_full_scan.jsonl"


def main() -> None:
    with CATALOG.open(encoding="utf-8-sig", newline="") as handle:
        candidates = [row for row in csv.DictReader(handle)
                      if row["筛选状态"].startswith("候选")]
    completed = {}
    for source in (OLD, NEW):
        if source.exists():
            for line in source.read_text(encoding="utf-8").splitlines():
                result = json.loads(line)
                completed[result["relative_path"]] = result
    pending = [row for row in candidates if row["Y盘相对路径"] not in completed]
    print(f"Candidate recordings: {len(candidates)}; cached: {len(candidates)-len(pending)}; to scan: {len(pending)}", flush=True)
    with NEW.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(scan, row) for row in pending]
            for number, future in enumerate(as_completed(futures), 1):
                result = future.result()
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                completed[result["relative_path"]] = result
                if number % 25 == 0 or number == len(pending):
                    print(f"Checked {number}/{len(pending)} new recordings", flush=True)
    rows = [{key: value for key, value in completed[row["Y盘相对路径"]].items()
             if not key.startswith("one_second_")} for row in candidates]
    with (OUT / "全候选整曲检查.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"candidate_tracks": len(rows),
                      "errors": sum(bool(row["error"]) for row in rows),
                      "with_clipping_seconds": sum(row.get("seconds_with_clip_over_0_1pct", 0) > 0 for row in rows)},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
