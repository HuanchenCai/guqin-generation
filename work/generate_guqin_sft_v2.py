"""Generate controlled A/B clips from the source-audited SFT adapter."""

import shutil
from pathlib import Path

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music

from generate_guqin_strict import PROMPT


ROOT = Path(__file__).parent
PROJECT = ROOT / "ACE-Step-1.5"
ADAPTER = ROOT / "guqin_v2" / "train" / "final"
OUTPUT = ROOT.parent / "outputs" / "古琴新SFT试听"


def make_clip(handler: AceStepHandler, seed: int, prompt: str, name: str) -> None:
    target = OUTPUT / name
    if target.exists():
        print(target, flush=True)
        return
    params = GenerationParams(
        task_type="text2music",
        caption=prompt,
        lyrics="[Instrumental]",
        instrumental=True,
        duration=30,
        seed=seed,
        shift=1.0,
        inference_steps=50,
        guidance_scale=7.0,
        thinking=False,
        use_cot_caption=False,
        use_cot_language=False,
        use_cot_metas=False,
    )
    config = GenerationConfig(batch_size=1, audio_format="flac", use_random_seed=False)
    result = generate_music(handler, None, params, config, save_dir=str(OUTPUT))
    if not result.success:
        raise RuntimeError(result.error or "Generation failed")
    shutil.move(result.audios[0]["path"], str(target))
    print(target, flush=True)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
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
    make_clip(handler, 5070, PROMPT, "新版SFT_严格同提示_5070.flac")
    make_clip(handler, 5070, "guqin_solo, " + PROMPT, "新版SFT_带标签_5070.flac")
    make_clip(handler, 5071, "guqin_solo, " + PROMPT, "新版SFT_带标签_5071.flac")


if __name__ == "__main__":
    main()
