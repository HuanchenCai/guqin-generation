"""Generate a fixed-seed, pre-fine-tuning guqin reference clip."""

from pathlib import Path

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music


PROJECT = Path(__file__).parent / "ACE-Step-1.5"
OUTPUT = Path(__file__).parents[1] / "outputs" / "baseline"

handler = AceStepHandler()
status, ok = handler.initialize_service(
    project_root=str(PROJECT),
    config_path="acestep-v15-turbo",
    device="cuda",
    use_flash_attention=True,
)
print(status, flush=True)
if not ok:
    raise SystemExit(1)

params = GenerationParams(
    task_type="text2music",
    caption=(
        "Solo traditional Chinese guqin zither, sparse free rhythm, "
        "warm wooden plucked strings, resonant low notes, delicate sliding ornaments, "
        "natural decays, intimate quiet room, unaccompanied instrumental, no singing"
    ),
    lyrics="[Instrumental]",
    instrumental=True,
    duration=30,
    seed=5070,
    thinking=False,
    use_cot_caption=False,
    use_cot_language=False,
    use_cot_metas=False,
)
config = GenerationConfig(batch_size=1, audio_format="flac", use_random_seed=False)
OUTPUT.mkdir(parents=True, exist_ok=True)
result = generate_music(handler, None, params, config, save_dir=str(OUTPUT))
print(result.status_message, flush=True)
if not result.success:
    raise SystemExit(result.error or "Generation failed")
for item in result.audios:
    print(item["path"], flush=True)
