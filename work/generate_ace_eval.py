"""Generate distinct guqin excerpts with ACE waveform-latent model variants."""

import argparse
import json
import shutil
from pathlib import Path

import torchaudio

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music

from guqin_eval_prompts import PROMPTS


ROOT = Path(__file__).parent
PROJECT = ROOT / "ACE-Step-1.5"
RUN = ROOT / "guqin_eval_v3"
RAW = RUN / "raw"
MP3 = RUN / "mp3"

MODELS = {
    "ace_sft_old": {
        "checkpoint": "acestep-v15-sft",
        "adapter": ROOT / "guqin_sft" / "train" / "final",
        "label": "ACE-Step SFT · 旧版古琴微调",
    },
    "ace_base": {
        "checkpoint": "acestep-v15-base",
        "adapter": None,
        "label": "ACE-Step Base · 原模型",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    args = parser.parse_args()
    config = MODELS[args.model]
    RAW.mkdir(parents=True, exist_ok=True)
    MP3.mkdir(parents=True, exist_ok=True)
    handler = AceStepHandler()
    _, ok = handler.initialize_service(
        project_root=str(PROJECT),
        config_path=config["checkpoint"],
        device="cuda",
        use_flash_attention=True,
    )
    if not ok:
        raise RuntimeError("ACE-Step model initialization failed")
    if config["adapter"] is not None:
        handler.load_lora(str(config["adapter"]))
        if not handler.lora_loaded:
            raise RuntimeError("LoRA load failed")
        handler.use_lora = True

    records = []
    for prompt in PROMPTS:
        stem = f"{args.model}_{prompt['id']}"
        flac = RAW / f"{stem}.flac"
        mp3 = MP3 / f"{stem}.mp3"
        if not flac.exists():
            params = GenerationParams(
                task_type="text2music",
                caption=prompt["text"],
                lyrics="[Instrumental]",
                instrumental=True,
                duration=15,
                seed=prompt["seed"],
                shift=1.0,
                inference_steps=50,
                guidance_scale=7.0,
                thinking=False,
                use_cot_caption=False,
                use_cot_language=False,
                use_cot_metas=False,
            )
            generation = generate_music(
                handler,
                None,
                params,
                GenerationConfig(batch_size=1, audio_format="flac", use_random_seed=False),
                save_dir=str(RAW),
            )
            if not generation.success:
                raise RuntimeError(generation.error or "Generation failed")
            shutil.move(generation.audios[0]["path"], flac)
        if not mp3.exists():
            audio, rate = torchaudio.load(str(flac))
            torchaudio.save(str(mp3), audio, rate, format="mp3")
        records.append({
            "id": stem,
            "model": args.model,
            "model_label": config["label"],
            "representation": "waveform_latent",
            "brief": prompt["title"],
            "prompt": prompt["text"],
            "seed": prompt["seed"],
            "path": str(mp3),
            "duration_s": 15,
        })
        print(f"Saved {mp3}", flush=True)
    (RUN / f"records_{args.model}.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
