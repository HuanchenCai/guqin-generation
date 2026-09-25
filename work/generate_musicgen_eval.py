"""Evaluate the official MusicGen small codec-token model on guqin prompts."""

import json
from pathlib import Path

import torch
import torchaudio
from transformers import AutoProcessor, MusicgenForConditionalGeneration

from guqin_eval_prompts import PROMPTS

ROOT = Path(__file__).parent
RUN = ROOT / "guqin_eval_v3"
MP3 = RUN / "mp3"
MODEL = ROOT / "models" / "musicgen-small"


def save(values, rate, path):
    audio = torch.as_tensor(values).detach().cpu().float()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    elif audio.ndim == 3:
        audio = audio[0]
    torchaudio.save(str(path), audio, rate, format="mp3")
    return round(audio.shape[-1] / rate, 2)


def main():
    MP3.mkdir(parents=True, exist_ok=True)
    print("Loading MusicGen small", flush=True)
    processor = AutoProcessor.from_pretrained(str(MODEL), local_files_only=True)
    model = MusicgenForConditionalGeneration.from_pretrained(
        str(MODEL), local_files_only=True, torch_dtype=torch.float16,
    ).to("cuda")
    model.eval()
    rate = model.config.audio_encoder.sampling_rate
    records = []
    for prompt in PROMPTS:
        path = MP3 / f"musicgen_small_{prompt['id']}.mp3"
        if not path.exists():
            torch.manual_seed(prompt["seed"])
            inputs = processor(text=[prompt["text"]], padding=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                values = model.generate(**inputs, do_sample=True, guidance_scale=3.0, max_new_tokens=750)
            duration = save(values, rate, path)
        else:
            info = torchaudio.info(str(path))
            duration = round(info.num_frames / info.sample_rate, 2)
        records.append(dict(id=f"musicgen_small_{prompt['id']}", model="musicgen_small",
                            representation="codec_tokens", task="free", brief=prompt["title"],
                            prompt=prompt["text"], seed=prompt["seed"], path=str(path),
                            duration_s=duration))
        print(f"Saved {path} {duration}s", flush=True)
    (RUN / "records_musicgen_small.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    path = MP3 / "musicgen_small_continuation.mp3"
    if not path.exists():
        reference, sr = torchaudio.load(str(RUN / "reference_10s.wav"))
        reference = reference.mean(0, keepdim=True)
        reference = torchaudio.functional.resample(reference, sr, rate).squeeze(0).numpy()
        torch.manual_seed(7101)
        inputs = processor(audio=reference, sampling_rate=rate,
                           text=["One solo Chinese guqin continues its wistful traditional melody with slow notes, natural slides and pauses; no other instruments."],
                           padding=True, return_tensors="pt").to("cuda")
        inputs["input_values"] = inputs["input_values"].half()
        with torch.inference_mode():
            values = model.generate(**inputs, do_sample=True, guidance_scale=3.0,
                                    max_new_tokens=950)
        duration = save(values, rate, path)
    else:
        info = torchaudio.info(str(path))
        duration = round(info.num_frames / info.sample_rate, 2)
    record = dict(id="musicgen_small_continuation", model="musicgen_small",
                  representation="codec_tokens", task="continuation", path=str(path),
                  duration_s=duration, reference_start_s=10, seed=7101)
    (RUN / "record_musicgen_small_continuation.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {path} {duration}s", flush=True)


if __name__ == "__main__":
    main()
