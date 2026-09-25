"""Make a second enlarged corpus after a conservative noise-proxy screen."""

import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "sa3_overnight"
DEST = ROOT / "quieter_latents"
LIMIT_DB = -35.0


def link_or_copy(source, target):
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main():
    audit = json.loads((ROOT / "noise_audit.json").read_text(encoding="utf-8"))
    allowed = {group: {row["file"] for row in audit[group]
                       if row["quiet_highpass_relative_db"] <= LIMIT_DB}
               for group in ("small", "large")}
    DEST.mkdir(parents=True, exist_ok=True)
    counts = {}
    count = 0
    for group in ("small", "large"):
        counts[group] = 0
        for latent in sorted((ROOT / f"{group}_latents").glob("*.npy")):
            if latent.name == "silence.npy":
                continue
            metadata = json.loads(latent.with_suffix(".json").read_text(encoding="utf-8"))
            if Path(metadata["relpath"]).name not in allowed[group]:
                continue
            name = f"{count:010d}"
            link_or_copy(latent, DEST / f"{name}.npy")
            metadata["source_group"] = group
            (DEST / f"{name}.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
            counts[group] += 1
            count += 1
    link_or_copy(ROOT / "small_latents" / "silence.npy", DEST / "silence.npy")
    result = {"threshold_relative_db": LIMIT_DB, "count": count, "by_group": counts,
              "caution": "A high-frequency energy proxy, not a verified human noise rating."}
    (ROOT / "quieter_corpus.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
