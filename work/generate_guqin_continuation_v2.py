"""Diagnostic continuation using the same held-out reference as the old SFT test."""

import shutil
from pathlib import Path

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music

from generate_guqin_strict import PROMPT


ROOT = Path(__file__).parent
PROJECT = ROOT / "ACE-Step-1.5"
SOURCE = ROOT / "guqin_holdout_reference.flac"
ADAPTER = ROOT / "guqin_v2" / "train" / "final"
OUTPUT = ROOT.parent / "outputs" / "古琴新SFT试听"
TARGET = OUTPUT / "新版SFT_续写对照_前10秒原曲.flac"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        print(TARGET, flush=True)
        return
    handler = AceStepHandler()
    _, ok = handler.initialize_service(
        project_root=str(PROJECT),
        config_path="acestep-v15-sft",
        device="cuda",
        use_flash_attention=True,
    )
    if not ok:
        raise RuntimeError("SFT model initialization failed")
    handler.load_lora(str(ADAPTER))
    if not handler.lora_loaded:
        raise RuntimeError("SFT LoRA loading failed")
    handler.use_lora = True
    params = GenerationParams(
        task_type="repaint",
        src_audio=str(SOURCE),
        caption="guqin_solo, " + PROMPT,
        lyrics="[Instrumental]",
        instrumental=True,
        duration=30,
        seed=5074,
        shift=1.0,
        inference_steps=50,
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
    shutil.move(result.audios[0]["path"], str(TARGET))
    print(TARGET, flush=True)


if __name__ == "__main__":
    main()
