"""Test whether corrected solo-instrument conditioning fixes ACE-Step v0 output."""

import shutil
from pathlib import Path

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music


ROOT = Path(__file__).parent
PROJECT = ROOT / "ACE-Step-1.5"
ADAPTER = ROOT / "guqin_v0" / "train" / "final"
OUTPUT = ROOT.parent / "outputs" / "古琴单乐器试验"
PROMPT = (
    "One performer playing one seven-string Chinese guqin (qin) alone. "
    "Sparse unmetered traditional qin melody with predominantly pentatonic phrases, "
    "clear single-string attacks, long natural resonance, left-hand gliding pressed tones "
    "and subtle yin-nao vibrato, occasional bright harmonics, generous silences. "
    "Only the guqin is audible in a quiet intimate recording."
)


def make_clip(handler: AceStepHandler, seed: int, name: str, reference: str | None = None) -> None:
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
        reference_audio=reference,
        audio_cover_strength=0.2 if reference else 1.0,
    )
    config = GenerationConfig(batch_size=1, audio_format="flac", use_random_seed=False)
    result = generate_music(handler, None, params, config, save_dir=str(OUTPUT))
    if not result.success:
        raise RuntimeError(result.error or "Generation failed")
    shutil.move(result.audios[0]["path"], str(OUTPUT / name))
    print(OUTPUT / name, flush=True)


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
    if not (OUTPUT / "01_修正提示词_基础模型.flac").exists():
        make_clip(handler, 5070, "01_修正提示词_基础模型.flac")
    handler.load_lora(str(ADAPTER))
    if not handler.lora_loaded:
        raise RuntimeError("LoRA loading failed")
    handler.use_lora = True
    if not (OUTPUT / "02_修正提示词_旧微调.flac").exists():
        make_clip(handler, 5070, "02_修正提示词_旧微调.flac")
    reference = str(ROOT / "guqin_v0" / "audio" / "CD第4册_13_1.flac")
    make_clip(handler, 5070, "03_修正提示词_参考音色.flac", reference)


if __name__ == "__main__":
    main()
