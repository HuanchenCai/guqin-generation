"""Generate several candidate clips from the revised solo-guqin adapter."""

import shutil
from pathlib import Path

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music

from generate_guqin_strict import PROMPT as DETAIL_PROMPT
from prepare_guqin_v1 import CAPTION as SIMPLE_PROMPT


ROOT = Path(__file__).parent
PROJECT = ROOT / "ACE-Step-1.5"
ADAPTER = ROOT / "guqin_v1" / "train" / "final"
OUTPUT = ROOT.parent / "outputs" / "古琴第二版试听"


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
        shift=3.0,
        inference_steps=8,
        thinking=False,
        use_cot_caption=False,
        use_cot_language=False,
        use_cot_metas=False,
    )
    config = GenerationConfig(batch_size=1, audio_format="flac", use_random_seed=False)
    result = generate_music(handler, None, params, config, save_dir=str(OUTPUT))
    if not result.success:
        raise RuntimeError(result.error or "Generation failed")
    shutil.move(result.audios[0]["path"], target)
    print(target, flush=True)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    handler = AceStepHandler()
    _, ok = handler.initialize_service(
        project_root=str(PROJECT),
        config_path="acestep-v15-turbo",
        device="cuda",
        use_flash_attention=True,
    )
    if not ok:
        raise RuntimeError("Model initialization failed")
    handler.load_lora(str(ADAPTER))
    if not handler.lora_loaded:
        raise RuntimeError("LoRA loading failed")
    handler.use_lora = True
    for seed in (5070, 5071, 5072, 5073):
        make_clip(handler, seed, SIMPLE_PROMPT, f"单古琴_{seed}.flac")
    make_clip(handler, 5070, "guqin_solo, " + DETAIL_PROMPT, "单古琴_细节提示_5070.flac")
    make_clip(handler, 5070, DETAIL_PROMPT, "单古琴_严格同提示_5070.flac")


if __name__ == "__main__":
    main()
