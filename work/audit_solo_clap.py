"""Screen selected guqin clips for accompaniment with a pretrained audio-text model.

CLAP similarities are triage signals, not definitive instrument labels.
"""

import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import ClapModel, ClapProcessor


ROOT = Path(__file__).parent
CLIPS = ROOT / "guqin_v0" / "audio"
OUTPUT = ROOT / "guqin_solo_clap_audit.csv"
MODEL_ID = "laion/clap-htsat-unfused"
LABELS = {
    "solo_guqin": "A solo guqin performance by one person, one Chinese seven-string qin only.",
    "solo_zither": "One person playing a solo plucked zither without any accompaniment.",
    "guqin_flute": "A duet with Chinese guqin and bamboo flute playing together.",
    "guqin_voice": "A person singing while playing a Chinese guqin.",
    "ensemble": "An ensemble of several musical instruments playing together.",
    "bass_drums": "Music with bass guitar and a drum beat.",
    "speech": "A person speaking over music.",
}


def load_ten_seconds(path: Path) -> np.ndarray:
    with sf.SoundFile(path) as source:
        frames = 10 * source.samplerate
        source.seek(max(0, (source.frames - frames) // 2))
        audio = source.read(frames, dtype="float32", always_2d=True).mean(axis=1)
        if source.samplerate != 48000:
            audio = resample_poly(audio, 48000, source.samplerate)
    return audio.astype(np.float32)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = ClapProcessor.from_pretrained(MODEL_ID)
    model = ClapModel.from_pretrained(MODEL_ID).to(device).eval()
    text_inputs = processor(text=list(LABELS.values()), return_tensors="pt", padding=True)
    with torch.inference_mode():
        text_features = model.get_text_features(**{k: v.to(device) for k, v in text_inputs.items()})
        text_features = torch.nn.functional.normalize(text_features, dim=-1)

    paths = sorted(CLIPS.glob("*_1.flac"))
    rows = []
    for path in paths:
        audio = load_ten_seconds(path)
        audio_inputs = processor(audios=[audio], sampling_rate=48000, return_tensors="pt")
        with torch.inference_mode():
            audio_features = model.get_audio_features(**{k: v.to(device) for k, v in audio_inputs.items()})
            audio_features = torch.nn.functional.normalize(audio_features, dim=-1)
            similarities = (audio_features @ text_features.T).squeeze(0).cpu().numpy()
        row = {"clip": path.name, **dict(zip(LABELS.keys(), np.round(similarities, 4)))}
        rows.append(row)
        print(row, flush=True)

    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(OUTPUT, flush=True)


if __name__ == "__main__":
    main()
