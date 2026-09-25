"""Check whether the larger SA3 audio representation preserves guqin cleanly."""

import json
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from stable_audio_3 import AutoencoderModel


ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "outputs" / "古琴夜间训练"
SOURCES = {
    "较新录音": (ROOT / "guqin_v2" / "audio" / "source_01_part_04.flac", 4),
    "未见录音": (ROOT / "guqin_holdout_reference.flac", 5),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = AutoencoderModel.from_pretrained("same-l", device="cuda")
    rows = []
    for label, (path, start) in SOURCES.items():
        audio, rate = torchaudio.load(str(path))
        audio = audio[:, start * rate : (start + 10) * rate].mean(0, keepdim=True).repeat(2, 1)
        if rate != 44100:
            audio = torchaudio.functional.resample(audio, rate, 44100)
            rate = 44100
        with torch.inference_mode():
            latent = model.encode(audio, rate)
            restored = model.decode(latent)[0].detach().float().cpu()
        restored = restored[:, : audio.shape[-1]]
        file = f"{label}_SAME-L还原.flac"
        sf.write(OUT / file, restored.T.numpy(), rate, format="FLAC", subtype="PCM_16")
        rows.append({"source": label, "file": file, "representation": "SAME-L",
                     "latent_shape": list(latent.shape)})
        print(rows[-1], flush=True)
    (OUT / "SAME-L还原_记录.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
