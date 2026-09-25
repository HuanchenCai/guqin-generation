"""Flag possible accompaniment in the newly added guqin clips (CPU only)."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import ClapModel, ClapProcessor


ROOT = Path(__file__).resolve().parent / "sa3_overnight"
MODEL = "laion/clap-htsat-unfused"
LABELS = {
    "solo_guqin": "A solo guqin performance by one person, one Chinese seven-string qin only.",
    "solo_zither": "One person playing a solo plucked zither without any accompaniment.",
    "guqin_flute": "A duet with Chinese guqin and bamboo flute playing together.",
    "guqin_voice": "A person singing while playing a Chinese guqin.",
    "ensemble": "An ensemble of several musical instruments playing together.",
    "bass_drums": "Music with bass guitar and a drum beat.",
    "speech": "A person speaking over music.",
}


def sample(path):
    with sf.SoundFile(path) as file:
        file.seek(max(0, (file.frames - 10 * file.samplerate) // 2))
        wave = file.read(10 * file.samplerate, dtype="float32", always_2d=True).mean(axis=1)
        if file.samplerate != 48000:
            wave = resample_poly(wave, 48000, file.samplerate)
    return wave.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["small", "large"], default="large")
    args = parser.parse_args()
    torch.set_num_threads(6)
    processor = ClapProcessor.from_pretrained(MODEL)
    model = ClapModel.from_pretrained(MODEL).eval().to("cpu")
    text = processor(text=list(LABELS.values()), return_tensors="pt", padding=True)
    with torch.inference_mode():
        text_features = torch.nn.functional.normalize(model.get_text_features(**text), dim=-1)
    rows = []
    for index, path in enumerate(sorted((ROOT / f"{args.group}_audio").glob("*.flac")), 1):
        audio = sample(path)
        inputs = processor(audios=[audio], sampling_rate=48000, return_tensors="pt")
        with torch.inference_mode():
            features = torch.nn.functional.normalize(model.get_audio_features(**inputs), dim=-1)
            scores = (features @ text_features.T)[0].numpy()
        row = {"file": path.name, **{label: round(float(score), 4) for label, score in zip(LABELS, scores)}}
        row["suspected_other_instruments"] = max(row[key] for key in ("guqin_flute", "guqin_voice", "ensemble", "bass_drums", "speech")) > max(row["solo_guqin"], row["solo_zither"]) + 0.02
        rows.append(row)
        if index % 10 == 0 or row["suspected_other_instruments"]:
            print(index, path.name, row["suspected_other_instruments"], flush=True)
    (ROOT / f"solo_clap_audit_{args.group}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("suspected", sum(row["suspected_other_instruments"] for row in rows), "of", len(rows), flush=True)


if __name__ == "__main__":
    main()
