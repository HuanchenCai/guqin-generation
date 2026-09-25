"""Run ACE-Step Base repaint on the same held-out guqin opening as SA3."""

import json
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUT = PROJECT / "outputs" / "古琴夜间训练"
SOURCE = OUT / "未见录音_原曲开头.wav"
SCRATCH = ROOT / "ace_same_source"
LENGTH = 35


def main() -> None:
    SCRATCH.mkdir(exist_ok=True)
    source, rate = sf.read(SOURCE, dtype="float32", always_2d=True)
    assert rate == 44100
    prefix = source[: 10 * rate]
    padded = np.zeros((LENGTH * rate, prefix.shape[1]), dtype=np.float32)
    padded[: len(prefix)] = prefix
    guide = SCRATCH / "guqin_heldout_35s.wav"
    sf.write(guide, padded, rate)

    start = time.perf_counter()
    handler = AceStepHandler()
    _, okay = handler.initialize_service(
        project_root=str(ROOT / "ACE-Step-1.5"),
        config_path="acestep-v15-base",
        device="cuda",
        use_flash_attention=True,
    )
    if not okay:
        raise RuntimeError("ACE-Step Base failed to initialize")
    params = GenerationParams(
        task_type="repaint", src_audio=str(guide),
        caption="Solo guqin instrumental.", lyrics="[Instrumental]", instrumental=True,
        duration=LENGTH, seed=94017, shift=1.0, inference_steps=50,
        guidance_scale=7.0, repainting_start=10.0, repainting_end=float(LENGTH),
        chunk_mask_mode="explicit", thinking=False, use_cot_caption=False,
        use_cot_language=False, use_cot_metas=False,
    )
    result = generate_music(handler, None, params,
                            GenerationConfig(batch_size=1, audio_format="flac", use_random_seed=False),
                            save_dir=str(SCRATCH))
    if not result.success:
        raise RuntimeError(result.error or "ACE-Step generation failed")
    target = OUT / "ACE基础版_同原曲续写_35秒.flac"
    shutil.move(result.audios[0]["path"], target)
    waveform, actual_rate = sf.read(target, dtype="float32", always_2d=True)
    generated = waveform[int(10 * actual_rate):]
    metrics = {
        "model": "ACE-Step 1.5 Base",
        "task": "repaint after the same held-out 10-second guqin source",
        "text": "Solo guqin instrumental. [Instrumental]",
        "source": str(SOURCE),
        "output": str(target),
        "duration_seconds": round(len(waveform) / actual_rate, 2),
        "generation_seconds": round(time.perf_counter() - start, 2),
        "generated_rms_dbfs": round(20 * np.log10(max(float(np.sqrt(np.mean(generated ** 2))), 1e-10)), 2),
        "generated_near_clip_fraction": round(float(np.mean(np.abs(generated) >= 0.999)), 6),
    }
    (OUT / "ACE基础版_同原曲续写_记录.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
