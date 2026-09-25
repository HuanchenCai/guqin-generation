"""Build new SA3 training subsets using the owner's performer noise feedback."""

import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "sa3_overnight"
EXCLUDED = {"张育瑾", "谢孝苹"}


def link_file(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def source_lookup() -> dict[str, tuple[str, str]]:
    manifest = json.loads((BASE / "corpus_manifest.json").read_text(encoding="utf-8"))
    lookup = {}
    for row in manifest["large_files"]:
        if row.get("file"):
            lookup[row["file"]] = (row["performer"], row["source"])
    with (ROOT / "guqin_v2" / "clip_report.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["accepted"].lower() == "true":
                performer = Path(row["source"].replace("\\", "/")).name.split("-", 1)[0]
                lookup[row["clip"]] = (performer, row["source"])
    return lookup


def canonical_name(relpath: str) -> str:
    name = Path(relpath.replace("\\", "/")).name
    for prefix in ("large_", "small_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def build_subset(source_dir: Path, target_dir: Path, lookup: dict[str, tuple[str, str]]):
    target_dir.mkdir(exist_ok=False)
    link_file(source_dir / "silence.npy", target_dir / "silence.npy")
    kept = []
    excluded = []
    for meta_path in sorted(source_dir.glob("*.json")):
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        key = canonical_name(metadata["relpath"])
        if key not in lookup:
            raise ValueError(f"Unmapped audio: {meta_path}: {key}")
        performer, source = lookup[key]
        row = {"latent": meta_path.name, "file": key, "performer": performer,
               "source": source, "seconds": metadata["seconds_total"]}
        if performer in EXCLUDED:
            excluded.append(row)
            continue
        kept.append(row)
        link_file(meta_path, target_dir / meta_path.name)
        latent_path = meta_path.with_suffix(".npy")
        if not latent_path.exists():
            raise FileNotFoundError(latent_path)
        link_file(latent_path, target_dir / latent_path.name)
    return kept, excluded


def main() -> None:
    lookup = source_lookup()
    variants = {
        "same_s": (BASE / "quieter_latents", BASE / "artist_clean_same_s_latents"),
        "same_l": (BASE / "quieter_same_l_latents", BASE / "artist_clean_same_l_latents"),
    }
    results = {}
    for name, (source_dir, target_dir) in variants.items():
        kept, excluded = build_subset(source_dir, target_dir, lookup)
        kept_names = Counter(row["file"] for row in kept)
        excluded_names = Counter(row["file"] for row in excluded)
        if any(count != 1 for count in kept_names.values()):
            raise AssertionError(f"Repeated latent source in {name}")
        results[name] = {"kept": kept, "excluded": excluded,
                         "kept_recordings": len(set(row["source"] for row in kept)),
                         "kept_performers": dict(Counter(row["performer"] for row in kept)),
                         "excluded_performers": dict(Counter(row["performer"] for row in excluded)),
                         "excluded_files": dict(excluded_names)}
    s = results["same_s"]
    l = results["same_l"]
    if Counter(row["file"] for row in s["kept"]) != Counter(row["file"] for row in l["kept"]):
        raise AssertionError("SAME-S and SAME-L retained different source clips")
    if Counter(row["file"] for row in s["excluded"]) != Counter(row["file"] for row in l["excluded"]):
        raise AssertionError("SAME-S and SAME-L excluded different source clips")
    report = {"excluded_performers": sorted(EXCLUDED),
              "source_dataset": "123 high-frequency-noise-screened clips",
              "kept_clips": len(s["kept"]),
              "kept_minutes": round(sum(row["seconds"] for row in s["kept"]) / 60, 2),
              "kept_recordings": s["kept_recordings"],
              "excluded_clips": len(s["excluded"]),
              "excluded_by_performer": s["excluded_performers"],
              "retained_by_performer": s["kept_performers"],
              "excluded_rows": s["excluded"],
              "retained_rows": s["kept"]}
    (BASE / "artist_clean_corpus.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("kept_clips", "kept_minutes", "kept_recordings", "excluded_clips", "excluded_by_performer", "retained_by_performer")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
