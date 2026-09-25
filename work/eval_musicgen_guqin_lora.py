"""Matched-seed continuation samples before and after guqin-only LoRA."""

import json
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from peft import PeftModel
from transformers import MusicgenForConditionalGeneration


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "musicgen-small"
ADAPTER_PATH = ROOT / "musicgen_guqin_lora" / "adapter"
OUT = ROOT.parent / "outputs" / "古琴交叉实验"
RATE = 32000
SEEDS = [
    ("1995较新录音", ROOT / "guqin_v2" / "audio" / "source_01_part_04.flac", 4),
    ("未参与训练录音", ROOT / "guqin_holdout_reference.flac", 5),
    ("1991中期录音", ROOT / "guqin_v2" / "audio" / "source_06_part_01.flac", 4),
    ("1988较旧录音", ROOT / "guqin_v2" / "audio" / "source_08_part_01.flac", 4),
]


def load_seed(path, start):
    x, rate = torchaudio.load(str(path))
    x = x.mean(0, keepdim=True)[:, int(start * rate) : int((start + 10) * rate)]
    return torchaudio.functional.resample(x, rate, RATE) if rate != RATE else x


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = MusicgenForConditionalGeneration.from_pretrained(
        str(MODEL_PATH), local_files_only=True, torch_dtype=torch.float16
    ).to("cuda").eval()
    records = []
    for label, source, start in SEEDS:
        seed_audio = load_seed(source, start)
        with torch.inference_mode():
            codes = model.audio_encoder.encode(seed_audio[None].half().to("cuda")).audio_codes
        for variant in ("原版", "古琴微调"):
            if variant == "古琴微调":
                model = PeftModel.from_pretrained(model, str(ADAPTER_PATH)).merge_and_unload().eval()
            torch.manual_seed(84913 + start)
            inputs = model.get_unconditional_inputs(num_samples=1)
            with torch.inference_mode():
                result = model.generate(**inputs, decoder_input_ids=codes[0, 0].long().to("cuda"),
                                        do_sample=True, max_new_tokens=600)
            audio = result[0].detach().float().cpu().reshape(-1).numpy()
            name = f"{label}_{variant}.wav"
            sf.write(OUT / name, audio, RATE)
            records.append({"file": name, "source": str(source), "source_start_seconds": start,
                            "condition": "10 seconds audio only, no text", "generation_seed": 84913 + start,
                            "seconds": round(len(audio) / RATE, 2), "continuation_starts_seconds": 10})
            print(name, flush=True)
        # Reload the base to keep both source prompts matched.
        model = MusicgenForConditionalGeneration.from_pretrained(
            str(MODEL_PATH), local_files_only=True, torch_dtype=torch.float16
        ).to("cuda").eval()
    (OUT / "生成记录.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
