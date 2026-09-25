"""Continue one identical 10-second guqin opening with ACE model variants."""

import argparse
import json
import shutil
from pathlib import Path

import torchaudio
from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music

ROOT = Path(__file__).parent
RUN = ROOT / "guqin_eval_v3"
RAW = RUN / "raw"
MP3 = RUN / "mp3"
PROMPT = (
    "A single Chinese seven-string guqin continues the same wistful traditional qin melody, "
    "with patient melodic development, sustained resonance, graceful left-hand slides, "
    "natural pauses and occasional harmonics. Only the guqin, no voice or accompaniment."
)
MODELS = {
    "ace_sft_old": ("acestep-v15-sft", ROOT / "guqin_sft" / "train" / "final", 50, 1.0, 7.0),
    "ace_sft_new": ("acestep-v15-sft", ROOT / "guqin_v2" / "train" / "final", 50, 1.0, 7.0),
    "ace_base": ("acestep-v15-base", None, 50, 1.0, 7.0),
    "ace_turbo": ("acestep-v15-turbo", None, 8, 3.0, 1.0),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    args = parser.parse_args()
    checkpoint, adapter, steps, shift, guidance = MODELS[args.model]
    RAW.mkdir(parents=True, exist_ok=True)
    MP3.mkdir(parents=True, exist_ok=True)
    flac = RAW / f"{args.model}_continuation.flac"
    mp3 = MP3 / f"{args.model}_continuation.mp3"
    if not flac.exists():
        handler = AceStepHandler()
        _, ok = handler.initialize_service(
            project_root=str(ROOT / "ACE-Step-1.5"), config_path=checkpoint,
            device="cuda", use_flash_attention=True,
        )
        if not ok:
            raise RuntimeError("ACE model initialization failed")
        if adapter:
            handler.load_lora(str(adapter))
            if not handler.lora_loaded:
                raise RuntimeError("LoRA load failed")
            handler.use_lora = True
        params = GenerationParams(
            task_type="repaint", src_audio=str(RUN / "reference_30s.wav"),
            caption=PROMPT, lyrics="[Instrumental]", instrumental=True,
            duration=30, seed=7101, shift=shift, inference_steps=steps,
            guidance_scale=guidance, repainting_start=10.0, repainting_end=30.0,
            chunk_mask_mode="explicit", thinking=False, use_cot_caption=False,
            use_cot_language=False, use_cot_metas=False,
        )
        result = generate_music(handler, None, params,
            GenerationConfig(batch_size=1, audio_format="flac", use_random_seed=False),
            save_dir=str(RAW))
        if not result.success:
            raise RuntimeError(result.error or "Generation failed")
        shutil.move(result.audios[0]["path"], flac)
    if not mp3.exists():
        audio, sr = torchaudio.load(str(flac))
        torchaudio.save(str(mp3), audio, sr, format="mp3")
    record = dict(id=f"{args.model}_continuation", model=args.model,
                  representation="waveform_latent", task="continuation", path=str(mp3),
                  prompt=PROMPT, seed=7101, duration_s=30, reference_start_s=10)
    (RUN / f"record_{args.model}_continuation.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(mp3, flush=True)


if __name__ == "__main__":
    main()
