"""Turn full-recording listener decisions into lossless, fully covered training windows."""

import argparse
import csv
import json
import re
from pathlib import Path

import soundfile as sf


WORK = Path(__file__).resolve().parent
SOURCE = Path(r"Y:\Music\古琴曲")
CATALOG = WORK.parent / "outputs" / "古琴整曲筛选" / "候选整曲清单.json"
WINDOW = 60


def seconds(value: str) -> int:
    parts = value.strip().split(":")
    if not all(part.isdigit() for part in parts) or not 1 <= len(parts) <= 3:
        raise ValueError(f"Invalid time: {value}")
    result = 0
    for part in parts:
        result = result * 60 + int(part)
    return result


def parse_ranges(raw: str, duration: float) -> list[tuple[float, float]]:
    ranges = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"\s*([0-9:]+)\s*[-–—]\s*([0-9:]+)\s*", line)
        if not match:
            raise ValueError(f"Invalid range: {line}")
        start, end = seconds(match[1]), seconds(match[2])
        if start >= end or end > duration + 1:
            raise ValueError(f"Range outside recording: {line}")
        ranges.append((start, min(end, duration)))
    if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
        raise ValueError("Overlapping or unsorted ranges")
    return ranges


def windows(start: float, end: float) -> list[tuple[float, float]]:
    length = end - start
    if length <= WINDOW:
        return [(start, end)]
    result = []
    cursor = start
    while cursor + WINDOW < end:
        result.append((cursor, cursor + WINDOW))
        cursor += WINDOW
    if cursor < end:
        tail = (min(cursor, end - WINDOW), end)
        if not result or tail != result[-1]:
            result.append(tail)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=WORK / "guqin_full_corpus")
    parser.add_argument("--materialize", action="store_true",
                        help="Read every selected source interval and write lossless FLAC windows")
    args = parser.parse_args()
    feedback = json.loads(args.feedback.read_text(encoding="utf-8"))
    state = feedback["state"]
    tracks = json.loads(CATALOG.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    if args.materialize:
        (args.output / "audio").mkdir(exist_ok=True)
    rows = []
    problems = []
    selected_duration = 0.0
    for track in tracks:
        answer = state.get(track["id"], {})
        decision = answer.get("decision")
        if decision not in {"keep", "partial"}:
            continue
        source = SOURCE / Path(track["id"].replace("\\", "/"))
        with sf.SoundFile(source) as stream:
            duration = stream.frames / stream.samplerate
            if decision == "keep":
                spans = [(0, duration)]
            else:
                try:
                    spans = parse_ranges(answer.get("ranges", ""), duration)
                    if not spans:
                        raise ValueError("Partial recording has no usable time range")
                except ValueError as error:
                    problems.append({"recording": track["id"], "error": str(error)})
                    continue
            selected_duration += sum(end - start for start, end in spans)
            for span_index, (start, end) in enumerate(spans, 1):
                for chunk_index, (chunk_start, chunk_end) in enumerate(windows(start, end), 1):
                    filename = f"guqin_{len(rows)+1:05d}.flac"
                    rows.append({"file": filename, "source": track["id"], "artist": track["artist"],
                                 "title": track["title"], "span": span_index, "window": chunk_index,
                                 "start_seconds": round(chunk_start, 3),
                                 "end_seconds": round(chunk_end, 3),
                                 "duration_seconds": round(chunk_end - chunk_start, 3)})
                    if args.materialize:
                        stream.seek(round(chunk_start * stream.samplerate))
                        count = round(chunk_end * stream.samplerate) - stream.tell()
                        audio = stream.read(count, dtype="float32", always_2d=True)
                        if len(audio) != count:
                            raise RuntimeError(f"Short read: {source}")
                        target = args.output / "audio" / filename
                        if not target.exists():
                            temp = target.with_suffix(".part.flac")
                            sf.write(temp, audio, stream.samplerate, format="FLAC", subtype="PCM_16")
                            temp.replace(target)
                        target.with_suffix(".txt").write_text("", encoding="utf-8")
                if args.materialize and len(rows) % 50 == 0:
                    print(f"Prepared {len(rows)} full-coverage windows", flush=True)
    if rows:
        with (args.output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {"selected_recordings": len({row["source"] for row in rows}),
               "source_hours": round(selected_duration / 3600, 2),
               "training_windows": len(rows), "problems": problems,
               "materialized": args.materialize}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
