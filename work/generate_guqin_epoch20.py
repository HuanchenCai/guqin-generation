"""Generate the same strict-prompt seed from the intermediate SFT checkpoint."""

from acestep.handler import AceStepHandler

from generate_guqin_sft_v2 import OUTPUT, PROJECT, make_clip
from generate_guqin_strict import PROMPT

from pathlib import Path


ROOT = Path(__file__).parent
ADAPTER = ROOT / "guqin_v2" / "epoch20_adapter"


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
        raise RuntimeError("Checkpoint LoRA loading failed")
    handler.use_lora = True
    make_clip(handler, 5070, PROMPT, "新版SFT_第20轮_严格同提示_5070.flac")


if __name__ == "__main__":
    main()
