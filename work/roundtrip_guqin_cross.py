"""Same-source EnCodec and DAC reconstruction for the model/data cross-check."""

import json
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from transformers import DacModel, MusicgenForConditionalGeneration


ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "outputs" / "古琴交叉实验"
SEEDS = [
    ("1995较新录音", ROOT / "guqin_v2" / "audio" / "source_01_part_04.flac", 4),
    ("未参与训练录音", ROOT / "guqin_holdout_reference.flac", 5),
    ("1991中期录音", ROOT / "guqin_v2" / "audio" / "source_06_part_01.flac", 4),
    ("1988较旧录音", ROOT / "guqin_v2" / "audio" / "source_08_part_01.flac", 4),
]


def wave(path, start, target_rate):
    x, rate = torchaudio.load(str(path))
    x = x[:, int(start * rate) : int((start + 10) * rate)].mean(0, keepdim=True)
    return torchaudio.functional.resample(x, rate, target_rate) if rate != target_rate else x


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    details = []
    musicgen = MusicgenForConditionalGeneration.from_pretrained(
        str(ROOT / "models" / "musicgen-small"), local_files_only=True, torch_dtype=torch.float32
    ).to("cuda").eval()
    with torch.inference_mode():
        for label, source, start in SEEDS:
            x = wave(source, start, 32000)
            encoded = musicgen.audio_encoder.encode(x[None].to("cuda"))
            decoded = musicgen.audio_encoder.decode(encoded.audio_codes, encoded.audio_scales).audio_values
            y = decoded[0].detach().float().cpu().reshape(-1)[: x.shape[-1]]
            sf.write(OUT / f"{label}_EnCodec还原.wav", y.numpy(), 32000)
            details.append({"recording": label, "representation": "EnCodec 32kHz, MusicGen audio tokens",
                            "source": str(source), "code_shape": list(encoded.audio_codes.shape)})
            print(f"EnCodec {label}", flush=True)
    del musicgen
    torch.cuda.empty_cache()
    dac = DacModel.from_pretrained(str(ROOT / "models" / "dac_44khz"), local_files_only=True).to("cuda").eval()
    with torch.inference_mode():
        for label, source, start in SEEDS:
            x = wave(source, start, 44100)
            pad = (-x.shape[-1]) % dac.config.hop_length
            encoded = dac.encode(torch.nn.functional.pad(x, (0, pad))[None].to("cuda"))
            decoded = dac.decode(audio_codes=encoded.audio_codes).audio_values
            y = decoded[0].detach().float().cpu().reshape(-1)[: x.shape[-1]]
            sf.write(OUT / f"{label}_DAC还原.wav", y.numpy(), 44100)
            details.append({"recording": label, "representation": "DAC 44.1kHz codes",
                            "source": str(source), "code_shape": list(encoded.audio_codes.shape)})
            print(f"DAC {label}", flush=True)
    (OUT / "还原记录.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
