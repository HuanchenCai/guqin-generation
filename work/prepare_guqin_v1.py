"""Build a controlled solo-guqin v1 dataset without bass-biased captions."""

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).parent
MANIFEST = ROOT / "guqin_v0" / "source_manifest.csv"
DATA = ROOT / "guqin_v1"
AUDIO = DATA / "audio"
CAPTION = (
    "guqin_solo, one unaccompanied Chinese seven-string guqin (qin) "
    "playing traditional guqin music in free rhythm, natural acoustic resonance"
)
SECONDS = 35
FRACTIONS = (0.10, 0.43, 0.76)
MAX_CLIPPED_FRACTION = 0.0001


def extract_clip(source: Path, target: Path, fraction: float) -> tuple[bool, float]:
    with sf.SoundFile(source) as stream:
        frames = SECONDS * stream.samplerate
        if stream.frames < frames:
            return False, 0.0
        start = int((stream.frames - frames) * fraction)
        stream.seek(start)
        audio = stream.read(frames, dtype="float32", always_2d=True)
        if len(audio) != frames or np.max(np.abs(audio)) < 0.005:
            return False, 0.0
        clipped_fraction = float(np.mean(np.abs(audio) >= 0.999))
        fade = min(int(0.02 * stream.samplerate), frames // 10)
        audio[:fade] *= np.linspace(0, 1, fade, dtype="float32")[:, None]
        audio[-fade:] *= np.linspace(1, 0, fade, dtype="float32")[:, None]
        sf.write(target, audio, stream.samplerate, format="FLAC", subtype="PCM_16")
        return True, clipped_fraction


def main() -> None:
    AUDIO.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        sources = [row for row in csv.DictReader(handle) if row["split"] == "train"]

    samples = []
    report = []
    for index, row in enumerate(sources, 1):
        source = Path(row["path"])
        if any(marker in source.stem for marker in ("合奏", "琴歌", "残")):
            raise ValueError(f"Excluded marker found: {source}")
        volume = next(part for part in source.parts if part.startswith("CD第"))
        for part, fraction in enumerate(FRACTIONS, 1):
            target = AUDIO / f"{volume}_{index:02d}_{part}.flac"
            extracted, clipped_fraction = extract_clip(source, target, fraction)
            success = extracted and clipped_fraction <= MAX_CLIPPED_FRACTION
            if extracted and not success:
                target.unlink()
            report.append({
                "source": str(source),
                "clip": target.name,
                "accepted": success,
                "clipped_fraction": round(clipped_fraction, 6),
                "filename_solo_screen": True,
                "manual_solo_confirmed": False,
            })
            if success:
                samples.append({
                    "audio_path": f"./{target.name}",
                    "caption": CAPTION,
                    "lyrics": "[Instrumental]",
                    "is_instrumental": True,
                    "genre": "Traditional Chinese guqin solo",
                    "custom_tag": "guqin_solo",
                })
        print(f"Prepared {index}/{len(sources)}: {source.name}", flush=True)

    with (AUDIO / "dataset.json").open("w", encoding="utf-8") as handle:
        json.dump(samples, handle, ensure_ascii=False, indent=2)
    with (DATA / "clip_report.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report[0].keys()))
        writer.writeheader()
        writer.writerows(report)
    print(f"Prepared {len(samples)} clips, {len(samples) * SECONDS / 60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
