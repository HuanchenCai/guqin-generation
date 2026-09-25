"""Create a conservative solo-only candidate set calibrated on confirmed clips."""

import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "sa3_overnight"
MAX_KNOWN_SOLO_MARGIN = 0.11
OTHER = ("guqin_flute", "guqin_voice", "ensemble", "bass_drums", "speech")


def margin(row):
    return max(row[key] for key in OTHER) - max(row["solo_guqin"], row["solo_zither"])


def link_or_copy(source, target):
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def make_set(source_dir, target_dir, medium, rejected):
    target_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    retained = []
    for source in sorted(source_dir.glob("*.npy")):
        if source.name == "silence.npy":
            continue
        metadata = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
        filename = Path(metadata["relpath"]).name
        group = "large" if filename.startswith("large_") else "small" if filename.startswith("small_") else metadata["source_group"]
        if medium:
            filename = filename.split("_", 1)[1]
        if group == "large" and filename in rejected:
            continue
        name = f"{count:010d}"
        link_or_copy(source, target_dir / f"{name}.npy")
        (target_dir / f"{name}.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        count += 1
        retained.append({"group": group, "file": filename})
    link_or_copy(source_dir / "silence.npy", target_dir / "silence.npy")
    return retained


def main():
    audit = json.loads((ROOT / "solo_clap_audit.json").read_text(encoding="utf-8"))
    rejected = {row["file"] for row in audit if margin(row) > MAX_KNOWN_SOLO_MARGIN}
    small = make_set(ROOT / "quieter_latents", ROOT / "strict_same_s_latents", False, rejected)
    medium = make_set(ROOT / "quieter_same_l_latents", ROOT / "strict_same_l_latents", True, rejected)
    if len(small) != len(medium) or set((row["group"], row["file"]) for row in small) != set((row["group"], row["file"]) for row in medium):
        raise RuntimeError("SAME-S and SAME-L sets do not contain the same audio")
    report = {"confirmed_solo_calibration_max_margin": MAX_KNOWN_SOLO_MARGIN,
              "small_model_clips": len(small), "medium_model_clips": len(medium),
              "small_original": sum(row["group"] == "small" for row in small),
              "large_added": sum(row["group"] == "large" for row in small),
              "excluded_large_files": sorted(rejected),
              "caution": "CLAP flags are uncertain; this is a conservative experiment, not proof of voice."}
    (ROOT / "strict_solo_corpus.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "excluded_large_files"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
