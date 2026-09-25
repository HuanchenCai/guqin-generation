"""Prepare a source-audited guqin set for SFT adapter training."""

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).parent
SOURCE_CSV = ROOT.parent / "outputs" / "古琴新素材筛选" / "候选曲目与年代.csv"
DATA = ROOT / "guqin_v2"
AUDIO = DATA / "audio"
CLIP_SECONDS = 45
MAX_CLIPS_PER_SOURCE = 6
MAX_CLIPPED_FRACTION = 0.0001
CAPTION = (
    "guqin_solo, one unaccompanied Chinese seven-string guqin (qin) "
    "playing traditional guqin music in free rhythm, natural acoustic resonance"
)


def main() -> None:
    AUDIO.mkdir(parents=True, exist_ok=True)
    with SOURCE_CSV.open(encoding="utf-8-sig", newline="") as handle:
        sources = list(csv.DictReader(handle))

    samples = []
    report = []
    for source_index, row in enumerate(sources, 1):
        source = Path(row["源文件"])
        if any(marker in source.stem for marker in ("合奏", "琴歌", "残")):
            raise ValueError(f"Excluded filename: {source}")
        with sf.SoundFile(source) as stream:
            frame_count = CLIP_SECONDS * stream.samplerate
            available = stream.frames / stream.samplerate - 2
            clip_count = min(MAX_CLIPS_PER_SOURCE, int(available // CLIP_SECONDS))
            if clip_count < 1:
                continue
            last_start = stream.frames / stream.samplerate - CLIP_SECONDS - 1
            starts = np.linspace(1, last_start, clip_count)
            for clip_index, start_sec in enumerate(starts, 1):
                stream.seek(round(float(start_sec) * stream.samplerate))
                audio = stream.read(frame_count, dtype="float32", always_2d=True)
                if len(audio) != frame_count:
                    raise RuntimeError(f"Short read: {source}")
                peak = float(np.max(np.abs(audio)))
                clipped = float(np.mean(np.abs(audio) >= 0.999))
                rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
                rms_dbfs = float(20 * np.log10(max(rms, 1e-10)))
                accepted = peak >= 0.005 and rms_dbfs >= -48 and clipped <= MAX_CLIPPED_FRACTION
                name = f"source_{source_index:02d}_part_{clip_index:02d}.flac"
                if accepted:
                    fade = min(round(0.02 * stream.samplerate), frame_count // 10)
                    audio[:fade] *= np.linspace(0, 1, fade, dtype="float32")[:, None]
                    audio[-fade:] *= np.linspace(1, 0, fade, dtype="float32")[:, None]
                    sf.write(AUDIO / name, audio, stream.samplerate, format="FLAC", subtype="PCM_16")
                    samples.append({
                        "audio_path": f"./{name}",
                        "caption": CAPTION,
                        "lyrics": "[Instrumental]",
                        "is_instrumental": True,
                        "genre": "Traditional Chinese guqin solo",
                        "custom_tag": "guqin_solo",
                    })
                report.append({
                    "source": str(source),
                    "recording_date": row["录制年代"],
                    "clip": name,
                    "start_sec": round(float(start_sec), 2),
                    "duration_sec": CLIP_SECONDS,
                    "accepted": accepted,
                    "rms_dbfs": round(rms_dbfs, 2),
                    "clipped_fraction": round(clipped, 7),
                    "manually_confirmed_guqin_solo": True,
                })
        print(f"Prepared {source_index}/{len(sources)}: {source.name}", flush=True)

    with (AUDIO / "dataset.json").open("w", encoding="utf-8") as handle:
        json.dump(samples, handle, ensure_ascii=False, indent=2)
    with (DATA / "clip_report.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report[0].keys()))
        writer.writeheader()
        writer.writerows(report)
    print(f"Prepared {len(samples)} clips, {len(samples) * CLIP_SECONDS / 60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
