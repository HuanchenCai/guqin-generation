"""Convert all newly trained Lightning LoRAs to inference safetensors."""

from pathlib import Path

from stable_audio_3.models.lora.utils import convert_lora_ckpt_to_safetensors


ROOT = Path(__file__).resolve().parent / "sa3_overnight"


def main() -> None:
    for ckpt in sorted(ROOT.glob("*_run/*.ckpt")):
        target = ckpt.with_suffix(".safetensors")
        if target.exists() and target.stat().st_mtime >= ckpt.stat().st_mtime:
            continue
        print(convert_lora_ckpt_to_safetensors(ckpt, target), flush=True)


if __name__ == "__main__":
    main()
