"""Create lossless local FLAC copies so file-based listening works without Y: URLs."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import soundfile as sf


WORK = Path(__file__).resolve().parent
SOURCE = Path(r"Y:\Music\古琴曲")
OUT = WORK.parent / "outputs" / "古琴整曲筛选" / "整曲音频"
CATALOG = WORK.parent / "outputs" / "古琴整曲筛选" / "候选整曲清单.json"


def copy_one(item: tuple[int, dict]) -> tuple[int, str, int]:
    number, track = item
    source = SOURCE / Path(track["id"].replace("\\", "/"))
    target = OUT / f"{number:04d}.flac"
    with sf.SoundFile(source) as original:
        if target.exists():
            try:
                with sf.SoundFile(target) as existing:
                    if (existing.frames == original.frames and
                        existing.samplerate == original.samplerate and
                        existing.channels == original.channels):
                        return number, "cached", target.stat().st_size
            except Exception:
                pass
        temp = target.with_suffix(".part.flac")
        with sf.SoundFile(temp, mode="w", samplerate=original.samplerate,
                          channels=original.channels, format="FLAC", subtype="PCM_16") as output:
            for block in original.blocks(blocksize=original.samplerate * 10,
                                         dtype="int16", always_2d=True):
                output.write(block)
        with sf.SoundFile(temp) as result:
            if (result.frames != original.frames or result.samplerate != original.samplerate
                    or result.channels != original.channels):
                raise RuntimeError(f"Copy properties differ: {source}")
        temp.replace(target)
    return number, "copied", target.stat().st_size


def main() -> None:
    tracks = json.loads(CATALOG.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    completed = 0
    copied = 0
    bytes_total = 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(copy_one, item) for item in enumerate(tracks, 1)]
        for future in as_completed(futures):
            number, state, size = future.result()
            completed += 1
            copied += state == "copied"
            bytes_total += size
            if completed % 25 == 0 or completed == len(tracks):
                print(f"Ready {completed}/{len(tracks)} full recordings; new={copied}; local GiB={bytes_total/2**30:.2f}", flush=True)
    print(json.dumps({"ready": completed, "new_copies": copied,
                      "local_gib": round(bytes_total/2**30, 2)}), flush=True)


if __name__ == "__main__":
    main()
