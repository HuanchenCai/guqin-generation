import csv
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(r"Y:\Music\古琴曲")
WORK = Path(__file__).parent
OUTPUT = WORK.parent / "outputs" / "古琴新素材筛选"
QUALITY = WORK / "full_library_quality.csv"
SAMPLES = [
    ("谢孝苹-忆故人.wav", "1995-09-19", 192),
    ("谢孝苹-良宵引1.wav", "1995-07-20/21", 192),
    ("谢孝苹-良宵引2.wav", "1995-09-19", 192),
    ("谢孝苹-长门怨.wav", "1995-09-19", 192),
    ("谢孝苹-洞庭秋思.wav", "1995-09-19", 192),
    ("李璠-平沙落雁1.wav", "1991-1992", 170),
    ("李璠-高山1.wav", "1991-1992", 170),
    ("吴兆基-潇湘水云2.wav", "1988", 152),
    ("吴兆基-秋塞吟3.wav", "1988", 152),
    ("吴兆基-石上流泉2.wav", "1988", 152),
]
RATE = 44100
CLIP_SECONDS = 15
GAP_SECONDS = 0.7


def db(value):
    return round(20 * np.log10(max(float(value), 1e-8)), 1)


def best_midpoint(audio):
    count = CLIP_SECONDS * RATE
    candidates = []
    for fraction in (0.24, 0.38, 0.52, 0.66, 0.80):
        start = max(0, min(audio.frames - count, round(audio.frames * fraction)))
        audio.seek(start)
        x = audio.read(count, dtype="float32", always_2d=True)
        if len(x) != count:
            continue
        rms = np.sqrt(np.mean(x.astype("float64") ** 2))
        # Prefer the central candidate that is audible but avoids unusually loud events.
        candidates.append((rms, -abs(fraction - 0.52), start, x))
    if not candidates:
        raise ValueError(f"Too short: {audio.name}")
    strong = [v for v in candidates if v[0] >= np.median([x[0] for x in candidates])]
    return max(strong, key=lambda v: v[1])


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with QUALITY.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    by_name = {row["name"]: row for row in source_rows}
    timeline = 0.0
    parts = []
    rows = []
    for index, (name, recorded, page) in enumerate(SAMPLES, 1):
        source = by_name[name]
        with sf.SoundFile(ROOT / source["relative_path"]) as audio:
            rms, _, start_frame, x = best_midpoint(audio)
        original_peak = float(np.max(np.abs(x)))
        gain = min(10 ** ((-23 - db(rms)) / 20), 10 ** (12 / 20), 0.98 / max(original_peak, 1e-6))
        x *= gain
        fade = np.linspace(0, 1, int(0.1 * RATE), dtype="float32")[:, None]
        x[: len(fade)] *= fade
        x[-len(fade) :] *= fade[::-1]
        parts.append(x)
        rows.append({
            "序号": index,
            "合辑开始秒": round(timeline, 1),
            "合辑结束秒": round(timeline + CLIP_SECONDS, 1),
            "文件名": name,
            "录制年代": recorded,
            "年代依据_本地书PDF页": page,
            "源文件": str(ROOT / source["relative_path"]),
            "原曲时长秒": source["duration_sec"],
            "采样率Hz": source["sample_rate_hz"],
            "位深": source["subtype"],
            "声道": source["channels"],
            "PCM码率kbps": source["pcm_bitrate_kbps"],
            "片段起点秒": round(start_frame / RATE, 1),
            "安静段RMS_dBFS_估计": source["quiet_rms_dbfs"],
            "中位RMS_dBFS_估计": source["median_rms_dbfs"],
            "抽样削波百分比": source["sampled_clip_pct"],
            "审听状态": "待人工确认古琴独奏、底噪、音色",
        })
        timeline += CLIP_SECONDS
        parts.append(np.zeros((round(GAP_SECONDS * RATE), 2), dtype="float32"))
        timeline += GAP_SECONDS
        print(f"{index}/{len(SAMPLES)} {name}", flush=True)

    reel = np.concatenate(parts)
    flac = OUTPUT / "较晚录音候选审听.flac"
    sf.write(flac, reel, RATE, format="FLAC", subtype="PCM_16")
    with (OUTPUT / "候选曲目与年代.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    shutil.copyfile(WORK / "full_library_metadata.csv", OUTPUT / "全库音频规格.csv")
    print(f"reel={flac}, duration={timeline:.1f}s", flush=True)


if __name__ == "__main__":
    main()
