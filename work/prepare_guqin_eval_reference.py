"""Prepare the shared solo-guqin opening for continuation comparisons."""

from pathlib import Path
import json

import torch
import torchaudio

ROOT = Path(__file__).parent / "guqin_eval_v3"
SOURCE = Path(__file__).parent / "guqin_holdout_reference.flac"
START = 0.0


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    info = torchaudio.info(str(SOURCE))
    audio, sr = torchaudio.load(str(SOURCE), frame_offset=int(START * info.sample_rate), num_frames=30 * info.sample_rate)
    audio = audio.mean(0, keepdim=True)
    audio = torchaudio.functional.resample(audio, sr, 44100) if sr != 44100 else audio
    prompt = audio[:, :10 * 44100]
    torchaudio.save(str(ROOT / "reference_30s.wav"), audio, 44100)
    torchaudio.save(str(ROOT / "reference_10s.wav"), prompt, 44100)
    torchaudio.save(str(ROOT / "reference_10s.mp3"), prompt, 44100, format="mp3")
    print(json.dumps({"source": str(SOURCE), "start_s": START, "sample_rate": 44100,
                      "peak": float(audio.abs().max()), "rms": float(torch.sqrt(audio.square().mean()))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
