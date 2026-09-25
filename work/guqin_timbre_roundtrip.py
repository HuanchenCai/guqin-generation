"""Check whether MusicGen's frozen codec preserves guqin string decay."""

import json
from pathlib import Path

import torch
import torchaudio
from transformers import MusicgenForConditionalGeneration

ROOT = Path(__file__).parent
SOURCE = ROOT / "guqin_v2" / "audio" / "source_01_part_04.flac"
MODEL = ROOT / "models" / "musicgen-small"
OUTPUT = ROOT.parent / "outputs" / "古琴音色诊断"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    audio, sr = torchaudio.load(str(SOURCE))
    audio = audio.mean(0, keepdim=True)[:, :30 * sr]
    torchaudio.save(str(OUTPUT / "01_原曲_1995古琴独奏.wav"), audio, sr)
    print("Loading MusicGen codec", flush=True)
    model = MusicgenForConditionalGeneration.from_pretrained(
        str(MODEL), local_files_only=True, torch_dtype=torch.float32).to("cuda").eval()
    codec = model.audio_encoder
    rate = codec.config.sampling_rate
    prepared = torchaudio.functional.resample(audio, sr, rate) if sr != rate else audio
    with torch.inference_mode():
        encoded = codec.encode(prepared.unsqueeze(0).to("cuda"))
        decoded = codec.decode(encoded.audio_codes, encoded.audio_scales).audio_values
    reconstructed = decoded[0].detach().cpu().float()
    if reconstructed.ndim == 1:
        reconstructed = reconstructed.unsqueeze(0)
    reconstructed = reconstructed[:, :prepared.shape[-1]]
    torchaudio.save(str(OUTPUT / "02_MusicGen编码还原.wav"), reconstructed, rate)
    report = {
        "source": str(SOURCE), "original_rate": sr, "codec_rate": rate,
        "original_rms": float(torch.sqrt(audio.square().mean())),
        "reconstruction_rms": float(torch.sqrt(reconstructed.square().mean())),
        "code_shape": list(encoded.audio_codes.shape),
    }
    (OUTPUT / "检测结果.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
