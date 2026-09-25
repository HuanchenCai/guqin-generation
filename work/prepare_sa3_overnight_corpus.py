"""Build matched small/large solo-guqin corpora for an overnight SA3 study."""

import json
import math
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parent
SOURCE = Path(r"Y:\Music\古琴曲\CD第4册")
OUT = ROOT / "sa3_overnight"
SMALL = OUT / "small_audio"
LARGE = OUT / "large_audio"
LARGE_SECONDS = 60
BAD_NAMES = ("合奏", "琴歌", "残", "历史录音")


def db(rms):
    return round(20 * math.log10(max(float(rms), 1e-9)), 2)


def inspect(audio, rate):
    mono = audio.mean(axis=1)
    frame = max(1, rate // 4)
    blocks = mono[: len(mono) // frame * frame].reshape(-1, frame)
    rms = np.sqrt(np.mean(blocks.astype(np.float64) ** 2, axis=1))
    return {"rms_dbfs": db(np.sqrt(np.mean(mono.astype(np.float64) ** 2))),
            "quiet_frame_dbfs": db(np.percentile(rms, 10)),
            "dynamic_range_db": round(db(np.percentile(rms, 90)) - db(np.percentile(rms, 10)), 2),
            "near_silence_fraction": round(float(np.mean(rms < 0.001)), 4),
            "clipped_fraction": round(float(np.mean(np.abs(audio) >= 0.999)), 6),
            "peak": round(float(np.max(np.abs(audio))), 5)}


def main():
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    SMALL.mkdir(parents=True, exist_ok=True)
    LARGE.mkdir(parents=True, exist_ok=True)
    small_rows = []
    for path in sorted((ROOT / "guqin_v2" / "audio").glob("*.flac")):
        dest = SMALL / path.name
        if not dest.exists():
            shutil.copy2(path, dest)
        dest.with_suffix(".txt").write_text("", encoding="utf-8")
        small_rows.append({"file": dest.name, "source": str(path), "seconds": round(sf.info(dest).duration, 2)})

    candidates = [p for p in sorted(SOURCE.rglob("*.wav"))
                  if p.parent.name.isdigit() and 57 <= int(p.parent.name) <= 69
                  and not any(marker in p.stem for marker in BAD_NAMES)]
    large_rows = []
    for index, path in enumerate(candidates, 1):
        row = {"source": str(path), "folder": path.parent.name,
               "performer": path.stem.split("-")[0], "title": path.stem.split("-", 1)[-1]}
        try:
            info = sf.info(path)
            row.update({"source_seconds": round(info.duration, 2), "source_rate": info.samplerate,
                        "source_channels": info.channels, "source_subtype": info.subtype})
            if info.duration < LARGE_SECONDS + 10 or info.samplerate != 44100 or info.channels != 2:
                row["decision"] = "exclude: short or unexpected format"
            else:
                start = max(5, (info.duration - LARGE_SECONDS) / 2)
                with sf.SoundFile(path) as src:
                    src.seek(int(start * info.samplerate))
                    audio = src.read(int(LARGE_SECONDS * info.samplerate), dtype="float32", always_2d=True)
                row.update(inspect(audio, info.samplerate))
                row["start_seconds"] = round(start, 2)
                if row["near_silence_fraction"] > 0.25:
                    row["decision"] = "exclude: too much near silence"
                elif row["clipped_fraction"] > 0.005:
                    row["decision"] = "exclude: clipping"
                elif row["rms_dbfs"] < -37 or row["rms_dbfs"] > -8:
                    row["decision"] = "exclude: extreme level"
                else:
                    row["decision"] = "include"
                    name = f"solo_{index:03d}.flac"
                    target = LARGE / name
                    if not target.exists():
                        sf.write(target, audio, info.samplerate, format="FLAC", subtype="PCM_16")
                    target.with_suffix(".txt").write_text("", encoding="utf-8")
                    row["file"] = name
        except Exception as exc:
            row["decision"] = f"exclude: {type(exc).__name__}: {exc}"
        large_rows.append(row)
        print(f"{index}/{len(candidates)} {row['decision']} {path.name}", flush=True)
    report = {"small_recordings": 10, "small_clips": len(small_rows),
              "small_minutes": round(sum(x["seconds"] for x in small_rows) / 60, 2),
              "large_candidates": len(candidates),
              "large_included": sum(x["decision"] == "include" for x in large_rows),
              "large_minutes": sum(x["decision"] == "include" for x in large_rows),
              "large_performers": len({x["performer"] for x in large_rows if x["decision"] == "include"}),
              "small_files": small_rows, "large_files": large_rows}
    (OUT / "corpus_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("files")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
