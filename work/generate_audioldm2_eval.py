"""Evaluate the official AudioLDM2 music spectrogram-latent model."""

import json
from pathlib import Path
from types import MethodType

import torch
import torchaudio
from diffusers import AudioLDM2Pipeline

from guqin_eval_prompts import PROMPTS

ROOT = Path(__file__).parent
RUN = ROOT / "guqin_eval_v3"
MP3 = RUN / "mp3"
MODEL = ROOT / "models" / "audioldm2-music"


def update_generation_kwargs(self, outputs, model_kwargs):
    """Restore the small GPT2 cache helper removed in recent Transformers."""
    model_kwargs["past_key_values"] = outputs.past_key_values
    if "attention_mask" in model_kwargs:
        mask = model_kwargs["attention_mask"]
        model_kwargs["attention_mask"] = torch.cat((mask, mask.new_ones((mask.shape[0], 1))), dim=-1)
    return model_kwargs


def main():
    MP3.mkdir(parents=True, exist_ok=True)
    print("Loading AudioLDM2 music", flush=True)
    pipe = AudioLDM2Pipeline.from_pretrained(str(MODEL), torch_dtype=torch.float16,
                                             use_safetensors=True, local_files_only=True)
    if not hasattr(pipe.language_model, "_update_model_kwargs_for_generation"):
        pipe.language_model._update_model_kwargs_for_generation = MethodType(
            update_generation_kwargs, pipe.language_model)
    pipe.to("cuda")
    rate = pipe.vocoder.config.sampling_rate
    records = []
    for prompt in PROMPTS:
        path = MP3 / f"audioldm2_music_{prompt['id']}.mp3"
        if not path.exists():
            generator = torch.Generator(device="cuda").manual_seed(prompt["seed"])
            audio = pipe(prompt=prompt["text"], negative_prompt="speech, voice, singing, drums, bass, piano, orchestra",
                         audio_length_in_s=15.0, num_inference_steps=50,
                         guidance_scale=3.5, generator=generator).audios[0]
            signal = torch.as_tensor(audio).float().reshape(1, -1)
            torchaudio.save(str(path), signal, rate, format="mp3")
            duration = round(signal.shape[-1] / rate, 2)
        else:
            info = torchaudio.info(str(path))
            duration = round(info.num_frames / info.sample_rate, 2)
        records.append(dict(id=f"audioldm2_music_{prompt['id']}", model="audioldm2_music",
                            representation="mel_spectrogram_latent", task="free", brief=prompt["title"],
                            prompt=prompt["text"], seed=prompt["seed"], path=str(path),
                            duration_s=duration))
        print(f"Saved {path} {duration}s", flush=True)
    (RUN / "records_audioldm2_music.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
