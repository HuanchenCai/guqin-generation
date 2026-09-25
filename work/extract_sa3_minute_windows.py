"""Save three short jump points from the minute-long continuation."""

from pathlib import Path

import soundfile as sf


out = Path(__file__).resolve().parent.parent / "outputs" / "古琴交叉实验"
audio, rate = sf.read(out / "1995较新录音_SA3一分钟.wav", always_2d=True)
for start in (10, 30, 50):
    sf.write(out / f"1995较新录音_SA3一分钟_{start}秒.wav",
             audio[start * rate : (start + 10) * rate], rate)
