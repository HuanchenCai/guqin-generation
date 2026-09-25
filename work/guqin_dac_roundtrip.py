"""Reconstruct the same clean guqin excerpt through pretrained 44.1 kHz DAC."""

import json
from pathlib import Path

import torch
import torchaudio
from transformers import DacModel

ROOT = Path(__file__).parent
MODEL = ROOT / "models" / "dac_44khz"
OUTPUT = ROOT.parent / "outputs" / "古琴音色诊断"


def main():
    audio, rate = torchaudio.load(str(OUTPUT / "01_原曲_1995古琴独奏.wav"))
    model = DacModel.from_pretrained(str(MODEL), local_files_only=True).to("cuda").eval()
    hop = model.config.hop_length
    padded = torch.nn.functional.pad(audio, (0, (-audio.shape[-1]) % hop))
    with torch.inference_mode():
        encoded = model.encode(padded.unsqueeze(0).to("cuda"))
        result = model.decode(audio_codes=encoded.audio_codes).audio_values
    reconstructed = result[0].detach().cpu().float().reshape(1, -1)[:, :audio.shape[-1]]
    path = OUTPUT / "03_DAC44k编码还原.wav"
    torchaudio.save(str(path), reconstructed, rate)
    report = {"codec": "DAC 44.1kHz", "code_shape": list(encoded.audio_codes.shape),
              "original_rms": float(torch.sqrt(audio.square().mean())),
              "reconstruction_rms": float(torch.sqrt(reconstructed.square().mean()))}
    (OUTPUT / "DAC检测结果.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
