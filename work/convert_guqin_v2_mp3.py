"""Create compact audition copies and verify generated audio is nonempty."""

from pathlib import Path

import numpy as np
import soundfile as sf
import torchaudio


OUTPUT = Path(__file__).parent.parent / "outputs" / "古琴新SFT试听"


def main() -> None:
    for source in sorted(OUTPUT.glob("*.flac")):
        waveform, sample_rate = torchaudio.load(str(source))
        duration = waveform.shape[1] / sample_rate
        peak = float(waveform.abs().max())
        rms = float(np.sqrt(np.mean(waveform.numpy().astype(np.float64) ** 2)))
        if not 29 <= duration <= 31 or peak < 0.01 or rms < 0.001:
            raise ValueError(f"Bad generated audio: {source.name}, {duration=}, {peak=}, {rms=}")
        target = source.with_suffix(".mp3")
        torchaudio.save(str(target), waveform, sample_rate, format="mp3")
        info = sf.info(source)
        print(f"{source.name}: {duration:.1f}s, {info.samplerate}Hz, peak={peak:.3f}, rms={rms:.4f} -> {target.name}")


if __name__ == "__main__":
    main()
