"""Generate a strict guqin baseline from ACE-Step 1.5 Base with CFG."""

import argparse
import shutil
from pathlib import Path

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music

from generate_guqin_strict import PROMPT


ROOT = Path(__file__).parent
PROJECT = ROOT / "ACE-Step-1.5"
OUTPUT = ROOT.parent / "outputs" / "古琴单乐器试验"
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("base", "sft"), default="base")
    args = parser.parse_args()
    model_name = "acestep-v15-" + args.variant
    name = "04_ACE基础版_高约束.flac" if args.variant == "base" else "05_ACE指令版_高约束.flac"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / name).exists():
        print(OUTPUT / name)
        return
    handler = AceStepHandler()
    _, ok = handler.initialize_service(
        project_root=str(PROJECT),
        config_path=model_name,
        device="cuda",
        use_flash_attention=True,
    )
    if not ok:
        raise RuntimeError("Base model initialization failed")
    params = GenerationParams(
        task_type="text2music",
        caption=PROMPT,
        lyrics="[Instrumental]",
        instrumental=True,
        duration=30,
        seed=5070,
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
    shutil.move(result.audios[0]["path"], str(OUTPUT / name))
    print(OUTPUT / name, flush=True)


if __name__ == "__main__":
    main()
