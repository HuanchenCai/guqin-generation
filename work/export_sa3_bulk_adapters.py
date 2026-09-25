"""Export Lightning Stable Audio 3 adapter checkpoints for inference."""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file


RUN = Path(__file__).resolve().parent / "sa3_feedback_bulk" / "medium_run"


def main() -> None:
    checkpoints = sorted(RUN.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {RUN}")
    for path in checkpoints:
        target = path.with_suffix(".safetensors")
        data = torch.load(path, map_location="cpu", weights_only=False)
        weights = {key: value.detach().contiguous().cpu() for key, value in data["state_dict"].items()}
        metadata = {"lora_config": json.dumps(data["lora_config"])}
        save_file(weights, str(target), metadata=metadata)
        print(f"Exported {target.name}: {len(weights)} tensors", flush=True)


if __name__ == "__main__":
    main()
