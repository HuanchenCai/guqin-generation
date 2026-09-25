"""Hard-link pre-encoded latents into piece-level pilot training groups."""

import csv
import json
import os
from pathlib import Path


WORK = Path(__file__).resolve().parent
ROOT = WORK / "guqin_aligned_412"
LATENTS = ROOT / "latents"


def link(source: Path, target: Path) -> None:
    if not target.exists():
        os.link(source, target)


def main() -> None:
    with (ROOT / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        assignment = {row["file"]: row["split"] for row in csv.DictReader(handle)}
    skipped = json.loads((ROOT / "silent_windows_skipped.json").read_text(
        encoding="utf-8"))["pure_digital_silence_skipped"]
    if not set(skipped).issubset(assignment):
        raise RuntimeError("Skipped windows are absent from source manifest")
    paths = list(LATENTS.glob("*.json"))
    expected = len(assignment) - len(skipped)
    if len(paths) != expected:
        raise RuntimeError(f"Encoded {len(paths)} / expected {expected} audible windows")
    counts = {"train": 0, "validation": 0, "test": 0}
    seen = set()
    for md_path in paths:
        metadata = json.loads(md_path.read_text(encoding="utf-8"))
        source_name = Path(metadata["path"]).name
        split = assignment[source_name]
        if source_name in seen:
            raise RuntimeError(f"Repeated encoded source {source_name}")
        seen.add(source_name)
        target_dir = ROOT / f"latents_{split}"
        target_dir.mkdir(exist_ok=True)
        link(md_path, target_dir / md_path.name)
        latent_path = md_path.with_suffix(".npy")
        link(latent_path, target_dir / latent_path.name)
        counts[split] += 1
    link(LATENTS / "silence.npy", ROOT / "latents_train" / "silence.npy")
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
