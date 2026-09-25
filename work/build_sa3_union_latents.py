"""Combine the previously encoded clean and expanded guqin clips."""

import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "sa3_overnight"
DEST = ROOT / "union_latents"


def link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    count = 0
    for group in ("small", "large"):
        source_dir = ROOT / f"{group}_latents"
        for latent in sorted(source_dir.glob("*.npy")):
            if latent.name == "silence.npy":
                continue
            name = f"{count:010d}"
            link_or_copy(latent, DEST / f"{name}.npy")
            metadata = json.loads(latent.with_suffix(".json").read_text(encoding="utf-8"))
            metadata["source_group"] = group
            (DEST / f"{name}.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
            count += 1
    link_or_copy(ROOT / "small_latents" / "silence.npy", DEST / "silence.npy")
    print(f"Combined {count} audio clips", flush=True)


if __name__ == "__main__":
    main()
