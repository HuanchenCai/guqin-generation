"""Repaint the last 20 seconds of held-out guqin audio with the v1 adapter."""

import argparse
import shutil
from pathlib import Path

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music

from generate_guqin_strict import PROMPT


ROOT = Path(__file__).parent
PROJECT = ROOT / "ACE-Step-1.5"
SOURCE = ROOT / "guqin_holdout_reference.flac"
OUTPUT = ROOT.parent / "outputs" / "古琴第二版试听"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("turbo", "sft"), default="turbo")
    args = parser.parse_args()
    adapter = ROOT / ("guqin_v1" if args.variant == "turbo" else "guqin_sft") / "train" / "final"
    name = "音频续写_前10秒原曲.flac" if args.variant == "turbo" else "SFT续写_前10秒原曲.flac"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / name).exists():
        print(OUTPUT / name)
        return
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    handler = AceStepHandler()
    _, ok = handler.initialize_service(
        project_root=str(PROJECT),
        config_path="acestep-v15-" + args.variant,
        device="cuda",
        use_flash_attention=True,
    )
    if not ok:
        raise RuntimeError("Model initialization failed")
    handler.load_lora(str(adapter))
    if not handler.lora_loaded:
        raise RuntimeError("LoRA loading failed")
    handler.use_lora = True
    params = GenerationParams(
        task_type="repaint",
        src_audio=str(SOURCE),
        caption="guqin_solo, " + PROMPT,
        lyrics="[Instrumental]",
        instrumental=True,
        duration=30,
        seed=5074,
        shift=3.0 if args.variant == "turbo" else 1.0,
        inference_steps=8 if args.variant == "turbo" else 50,
        guidance_scale=7.0,
        repainting_start=10.0,
        repainting_end=30.0,
        chunk_mask_mode="explicit",
        thinking=False,
        use_cot_caption=False,
        use_cot_language=False,
        use_cot_metas=False,
    )
    config = GenerationConfig(batch_size=1, audio_format="flac", use_random_seed=False)
    result = generate_music(handler, None, params, config, save_dir=str(OUTPUT))
    if not result.success:
        raise RuntimeError(result.error or "Generation failed")
    shutil.move(result.audios[0]["path"], str(OUTPUT / name))
    print(OUTPUT / name, flush=True)


if __name__ == "__main__":
    main()
