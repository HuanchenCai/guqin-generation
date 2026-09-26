"""Captions with piece stories, mood tags and the estimated pentatonic mode.

Stories and tags come from guqin_piece_stories_draft.csv (researched notes on
each piece). Each window gets one caption style by a stable hash:
40% tags + mode, 40% story + tags + mode, 20% story only. Training blanks a
further 10% of prompts (cfg dropout).

Split for the long run: validation pieces are held out; train and test
pieces are both trained on.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from build_guqin_captions import HEAD_TEXT, MOOD, ROOT, SCENE, SOURCE, mode_text, normalize, window_modes

WORK = Path(__file__).resolve().parent
MOOD = {**MOOD, "隐逸": "reclusive", "知音": "kindred spirits", "幽冥": "eerie, ghostly"}
TABLE = WORK / "guqin_piece_stories_draft.csv"


def load_pieces() -> dict:
    with TABLE.open(encoding="utf-8-sig", newline="") as handle:
        rows = {r["曲名"]: r for r in csv.DictReader(handle)}
    for r in rows.values():
        r["tags"] = [t for t in r["标签"].split(";") if t]
        unknown = [t for t in r["tags"] if t not in MOOD and t not in SCENE]
        if unknown:
            raise ValueError(f"{r['曲名']}: unknown tags {unknown}")
    return rows


def tag_text(tags) -> str:
    mood = [MOOD[t] for t in tags if t in MOOD]
    scene = [SCENE[t] for t in tags if t in SCENE]
    parts = ([f"Mood: {', '.join(mood)}."] if mood else []) + ([f"Scene: {', '.join(scene)}."] if scene else [])
    return " ".join(parts) + " 意境：" + "、".join(tags) + "。"


def story_text(row) -> str:
    return f"{row['英文意译']} ({row['曲名']}). {row['story_en']} {row['故事']}"


def caption(style: str, row: dict, gong: int) -> str:
    if style == "story":
        return f"{HEAD_TEXT} {story_text(row)}"
    if style == "story_tags":
        return f"{HEAD_TEXT} {story_text(row)} {tag_text(row['tags'])} {mode_text(gong)}"
    return f"{HEAD_TEXT} {tag_text(row['tags'])} {mode_text(gong)}"


def style_for(name: str) -> str:
    bucket = int(hashlib.sha1(("story|" + name).encode()).hexdigest(), 16) % 10
    return "tags" if bucket < 4 else "story_tags" if bucket < 8 else "story"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", required=True, help="manifest splits to include")
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    target = Path(args.target)
    target.mkdir(exist_ok=True)
    pieces = load_pieces()
    with (ROOT / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        manifest = {row["file"]: row for row in csv.DictReader(handle)}
    rows = {}
    for md_path in sorted(SOURCE.glob("*.json")):
        row = manifest[Path(json.loads(md_path.read_text(encoding="utf-8"))["path"]).name]
        if row["split"] in args.splits:
            rows[md_path] = row
    modes = window_modes([r["file"] for r in rows.values()])
    counts = {"tags": 0, "story_tags": 0, "story": 0}
    for md_path, row in rows.items():
        metadata = json.loads(md_path.read_text(encoding="utf-8"))
        style = style_for(md_path.stem)
        metadata["prompt"] = caption(style, pieces[normalize(row["piece"])], modes[row["file"]]["gong"])
        counts[style] += 1
        # ASCII-escaped JSON: the loader reads metadata with the Windows default codec.
        (target / md_path.name).write_text(json.dumps(metadata), encoding="utf-8")
        latent = target / md_path.with_suffix(".npy").name
        if not latent.exists():
            os.link(md_path.with_suffix(".npy"), latent)
    silence = target / "silence.npy"
    if not silence.exists():
        os.link(SOURCE / "silence.npy", silence)
    print(json.dumps({"windows": len(rows), **counts}), flush=True)


if __name__ == "__main__":
    main()
