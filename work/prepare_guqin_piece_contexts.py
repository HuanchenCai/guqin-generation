"""Cache the fixed song-held-out continuation openings as short lossless files."""

import csv
import json
from pathlib import Path

import soundfile as sf
import torch
import torchaudio


WORK = Path(__file__).resolve().parent
SOURCE = Path("Y:/Music/古琴曲")
ROOT = WORK / "guqin_aligned_412" / "eval_contexts"
INDEX = WORK.parent / "outputs" / "古琴整曲筛选" / "评估开头清单.csv"


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(INDEX.open(encoding="utf-8-sig", newline="")))
    records = []
    for index, row in enumerate(rows, 1):
        source = SOURCE / row["Y盘相对路径"]
        start = int(row["前奏起点秒"])
        with sf.SoundFile(source) as stream:
            stream.seek(start * stream.samplerate)
            audio = stream.read(10 * stream.samplerate, dtype="float32", always_2d=True)
            rate = stream.samplerate
        if len(audio) < 10 * rate:
            raise RuntimeError(f"Source too short: {source}")
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        else:
            audio = audio[:, :2]
        if rate != 44100:
            audio = torchaudio.functional.resample(torch.from_numpy(audio.T), rate, 44100).T.numpy()
        audio = audio[:441000]
        file = f"{row['分组']}_{index:02d}.flac"
        sf.write(ROOT / file, audio, 44100, format="FLAC", subtype="PCM_24")
        records.append({"file": file, **row})
    (ROOT / "index.json").write_text(json.dumps(records, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print(f"Prepared {len(records)} fixed openings")


if __name__ == "__main__":
    main()
