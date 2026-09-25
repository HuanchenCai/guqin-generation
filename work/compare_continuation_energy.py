"""Measure preserved intro and obvious silence in old/new continuation tests."""

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


ROOT = Path(__file__).parent
FILES = {
    "reference": ROOT / "guqin_holdout_reference.flac",
    "old_sft": ROOT.parent / "outputs" / "古琴第二版试听" / "SFT续写_前10秒原曲.flac",
    "new_sft": ROOT.parent / "outputs" / "古琴新SFT试听" / "新版SFT_续写对照_前10秒原曲.flac",
}


def load(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != 48000:
        audio = resample_poly(audio, 48000, rate, axis=0)
    return audio.mean(axis=1)


def rms_db(audio: np.ndarray) -> float:
    return float(20 * np.log10(max(np.sqrt(np.mean(audio.astype(np.float64) ** 2)), 1e-10)))


def main() -> None:
    samples = {name: load(path) for name, path in FILES.items()}
    reference = samples["reference"]
    for name, audio in samples.items():
        chunks = [rms_db(audio[start * 48000:end * 48000]) for start, end in ((1, 8), (10, 12), (12, 20), (20, 30))]
        intro = audio[48000:8 * 48000]
        corr = float(np.corrcoef(reference[48000:8 * 48000], intro)[0, 1])
        print(f"{name}: intro_corr={corr:.4f}, rms_dbfs_1to8_10to12_12to20_20to30={chunks}")


if __name__ == "__main__":
    main()
