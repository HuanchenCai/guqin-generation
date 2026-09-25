"""Check that the intro is preserved and the repaint region differs from source."""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("turbo", "sft"), default="turbo")
    args = parser.parse_args()
    root = Path(__file__).parent
    name = "音频续写_前10秒原曲.flac" if args.variant == "turbo" else "SFT续写_前10秒原曲.flac"
    source, source_rate = sf.read(root / "guqin_holdout_reference.flac", dtype="float32", always_2d=True)
    generated, generated_rate = sf.read(
        root.parent / "outputs" / "古琴第二版试听" / name,
        dtype="float32",
        always_2d=True,
    )
    if source_rate != generated_rate:
        source = resample_poly(source, generated_rate, source_rate, axis=0)
    for start, end in ((1, 8), (12, 28)):
        original = source[int(start * generated_rate):int(end * generated_rate)].mean(axis=1)
        output = generated[int(start * generated_rate):int(end * generated_rate)].mean(axis=1)
        count = min(len(original), len(output))
        correlation = np.corrcoef(original[:count], output[:count])[0, 1]
        print(f"{start}-{end}s correlation: {correlation:.4f}")
    print(f"duration: {len(generated)/generated_rate:.2f}s, sample_rate: {generated_rate}")


if __name__ == "__main__":
    main()
