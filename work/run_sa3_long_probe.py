"""One-pass minute-long no-text guqin continuation, without feedback chaining."""

import json
import time
from pathlib import Path

import soundfile as sf
import torchaudio
from stable_audio_3 import StableAudioModel


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "guqin_v2" / "audio" / "source_01_part_04.flac"
OUT = ROOT.parent / "outputs" / "古琴交叉实验"


def main():
    wave, sr = torchaudio.load(str(SOURCE))
    wave = wave[:, 4*sr:14*sr].mean(0, keepdim=True).repeat(2, 1)
    model = StableAudioModel.from_pretrained("small-music", device="cuda")
    started = time.monotonic()
    result = model.generate(prompt="", duration=60, steps=8, seed=94017,
                            inpaint_audio=(sr, wave), inpaint_mask_start_seconds=10,
                            inpaint_mask_end_seconds=60, chunked_decode=True)
    elapsed = round(time.monotonic() - started, 2)
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "1995较新录音_SA3一分钟.wav"
    sf.write(target, result[0].detach().float().cpu().T.numpy(), 44100)
    report = {"source": str(SOURCE), "prompt": "empty", "context_seconds": 10,
              "total_seconds": 60, "model": "Stable Audio 3 Small Music", "sampling_steps": 8,
              "generation_plus_decode_seconds_after_loading": elapsed, "file": target.name}
    (OUT / "一分钟试验记录.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
