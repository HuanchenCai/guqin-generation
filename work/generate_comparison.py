"""Generate matching base and fine-tuned guqin clips for listening review."""

import shutil
from pathlib import Path

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music


ROOT = Path(__file__).parent
PROJECT = ROOT / "ACE-Step-1.5"
ADAPTER = ROOT / "guqin_v0" / "train" / "final"
OUTPUT = ROOT.parent / "outputs" / "古琴试听对比"
PROMPT = (
    "Solo traditional Chinese guqin zither, sparse free rhythm, warm wooden "
    "plucked strings, resonant low notes, delicate sliding ornaments, long "
    "natural decays, intimate quiet room, unaccompanied instrumental, no singing"
)


def make_clip(handler: AceStepHandler, seed: int, name: str) -> None:
    params = GenerationParams(
        task_type="text2music",
        caption=PROMPT,
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
    source = Path(result.audios[0]["path"])
    target = OUTPUT / name
    shutil.move(str(source), str(target))
    print(target, flush=True)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    handler = AceStepHandler()
    status, ok = handler.initialize_service(
        project_root=str(PROJECT),
        config_path="acestep-v15-turbo",
        device="cuda",
        use_flash_attention=True,
    )
    print(status, flush=True)
    if not ok:
        raise RuntimeError("Model initialization failed")
    if not (OUTPUT / "01_基础模型.flac").exists():
        make_clip(handler, 5070, "01_基础模型.flac")
    handler.load_lora(str(ADAPTER))
    if not handler.lora_loaded:
        raise RuntimeError("LoRA loading failed")
    handler.use_lora = True
    make_clip(handler, 5070, "02_古琴微调_同种子.flac")
    make_clip(handler, 5071, "03_古琴微调_另一种子.flac")


if __name__ == "__main__":
    main()
