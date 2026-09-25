"""Cross-check SAME reconstruction and no-text SA3 continuation on guqin."""

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from stable_audio_3 import AutoencoderModel, StableAudioModel


ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "outputs" / "古琴交叉实验"
SEEDS = [
    ("1995较新录音", ROOT / "guqin_v2" / "audio" / "source_01_part_04.flac", 4),
    ("未参与训练录音", ROOT / "guqin_holdout_reference.flac", 5),
    ("1991中期录音", ROOT / "guqin_v2" / "audio" / "source_06_part_01.flac", 4),
    ("1988较旧录音", ROOT / "guqin_v2" / "audio" / "source_08_part_01.flac", 4),
]


def seed_audio(path, start):
    wave, sr = torchaudio.load(str(path))
    wave = wave[:, int(start * sr) : int((start + 10) * sr)]
    # Use the same mono source in both models; duplicate only at the stereo interface.
    wave = wave.mean(0, keepdim=True).repeat(2, 1)
    return wave, sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ae = AutoencoderModel.from_pretrained("same-s", device="cuda")
    model = StableAudioModel.from_pretrained("small-music", device="cuda")
    rows = []
    for index, (label, path, start) in enumerate(SEEDS[: args.count]):
        wave, sr = seed_audio(path, start)
        original = torchaudio.functional.resample(wave, sr, 44100) if sr != 44100 else wave
        sf.write(OUT / f"{label}_原曲开头.wav", original.T.numpy(), 44100)
        with torch.inference_mode():
            latent = ae.encode(wave, sr)
            restored = ae.decode(latent)[0].detach().float().cpu()
        sf.write(OUT / f"{label}_SAME还原.wav", restored.T.numpy()[: len(original.T)], 44100)
        print(f"Reconstructed {label}, latent {tuple(latent.shape)}", flush=True)
        torch.manual_seed(84917 + index)
        result = model.generate(prompt="", duration=22, steps=8, seed=84917 + index,
                                inpaint_audio=(sr, wave), inpaint_mask_start_seconds=10,
                                inpaint_mask_end_seconds=22, chunked_decode=True)
        audio = result[0].detach().float().cpu()
        sf.write(OUT / f"{label}_SA3纯音频续写.wav", audio.T.numpy(), 44100)
        rows.append({"label": label, "source": str(path), "source_start": start, "seed": 84917 + index,
                     "prompt": "empty", "context_seconds": 10, "continuation_seconds": 12,
                     "model": "Stable Audio 3 Small Music", "audio_representation": "SAME-S 44.1kHz stereo latent"})
        print(f"Generated {label}", flush=True)
    (OUT / "SA3生成记录.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
