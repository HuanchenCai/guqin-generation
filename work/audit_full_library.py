import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

import soundfile as sf


ROOT = Path(r"Y:\Music\古琴曲")
OUT = Path(__file__).parent / "full_library_metadata.csv"

rows = []
for path in sorted(p for folder in ROOT.glob("CD第*") for p in folder.rglob("*.wav")):
    stat = path.stat()
    row = {
        "relative_path": str(path.relative_to(ROOT)),
        "name": path.name,
        "bytes": stat.st_size,
        "file_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "sample_rate_hz": "",
        "channels": "",
        "subtype": "",
        "duration_sec": "",
        "pcm_bitrate_kbps": "",
        "error": "",
    }
    try:
        info = sf.info(path)
        row.update(
            sample_rate_hz=info.samplerate,
            channels=info.channels,
            subtype=info.subtype,
            duration_sec=round(info.duration, 2),
        )
        if info.subtype.startswith("PCM_"):
            bits = int(info.subtype.split("_")[1])
            row["pcm_bitrate_kbps"] = int(info.samplerate * info.channels * bits / 1000)
    except Exception as exc:
        row["error"] = str(exc)
    rows.append(row)

with OUT.open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("files", len(rows), "errors", sum(bool(r["error"]) for r in rows))
for field in ("sample_rate_hz", "channels", "subtype", "pcm_bitrate_kbps", "file_modified"):
    print(field, Counter(r[field] for r in rows).most_common(20))
print("output", OUT)
