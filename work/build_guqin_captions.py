"""Write text captions (piece name + mood/scene tags) into pre-encoded latents.

The original latents keep their empty prompts. Captioned copies of the .json
metadata go to guqin_aligned_412/latents_captioned; the .npy latents are
hard-linked, so nothing is re-encoded.

Each window gets one caption style, chosen by a stable hash so reruns agree:
50% tags + piece + performer, 30% tags only, 20% piece + performer only.
Training additionally blanks 10% of prompts (cfg dropout), so empty-prompt
generation keeps working.
"""

import csv
import hashlib
import json
import os
import re
from pathlib import Path

WORK = Path(__file__).resolve().parent
ROOT = WORK / "guqin_aligned_412"
SOURCE = ROOT / "latents"
TARGET = ROOT / "latents_captioned"

MOOD = {"平静": "calm", "恬淡": "tranquil", "孤寂": "lonely", "忧伤": "melancholy",
        "悲愤": "sorrowful and indignant", "思念": "longing", "离别": "farewell",
        "激昂": "impassioned", "欢快": "joyful", "洒脱": "carefree", "庄严": "solemn",
        "禅意": "meditative", "雄浑": "majestic", "苍茫": "vast", "灵动": "lively",
        "清雅": "elegant", "醉": "drunken"}
SCENE = {"流水": "flowing water", "江河": "river", "高山": "mountains", "云": "clouds",
         "月夜": "moonlit night", "夜": "night", "秋": "autumn", "春": "spring",
         "冬雪": "snow", "风": "wind", "雨": "rain", "飞鸟": "birds",
         "渔樵": "fisherman and woodcutter", "塞外大漠": "frontier desert",
         "宫廷": "palace", "仙境": "immortals", "田园": "countryside"}
ALIASES = {"醉渔晚唱": "醉渔唱晚"}


def normalize(piece: str) -> str:
    name = re.sub(r"-残$", "", piece.strip()).lstrip("-")
    name = re.sub(r"\d+$", "", name)
    return ALIASES.get(name, name)


def load_tags() -> dict:
    with (WORK / "guqin_piece_tags.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = {row["曲名"]: row for row in csv.DictReader(handle)}
    for row in rows.values():
        row["tags"] = [t for t in row["标签"].split(";") if t]
        unknown = [t for t in row["tags"] if t not in MOOD and t not in SCENE]
        if unknown:
            raise ValueError(f"{row['曲名']}: unknown tags {unknown}")
    return rows


def tag_text(tags: list[str]) -> str:
    mood = [MOOD[t] for t in tags if t in MOOD]
    scene = [SCENE[t] for t in tags if t in SCENE]
    parts = []
    if mood:
        parts.append("Mood: " + ", ".join(mood) + ".")
    if scene:
        parts.append("Scene: " + ", ".join(scene) + ".")
    return " ".join(parts) + " 意境：" + "、".join(tags) + "。"


def caption(style: str, row: dict, performer: str) -> str:
    head = "Solo guqin, Chinese seven-string zither."
    piece = f"Piece: {row['拼音']} ({row['英文意译']}) {row['曲名']}. Performer: {performer}."
    if style == "tags":
        return f"{head} {tag_text(row['tags'])}"
    if style == "piece":
        return f"{head} {piece}"
    return f"{head} {tag_text(row['tags'])} {piece}"


def style_for(name: str) -> str:
    bucket = int(hashlib.sha1(name.encode()).hexdigest(), 16) % 10
    return "full" if bucket < 5 else "tags" if bucket < 8 else "piece"


def main() -> None:
    tags = load_tags()
    with (ROOT / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        manifest = {row["file"]: row for row in csv.DictReader(handle)}
    TARGET.mkdir(exist_ok=True)
    counts = {"full": 0, "tags": 0, "piece": 0}
    missing = set()
    for md_path in sorted(SOURCE.glob("*.json")):
        metadata = json.loads(md_path.read_text(encoding="utf-8"))
        row = manifest[Path(metadata["path"]).name]
        piece = normalize(row["piece"])
        if piece not in tags:
            missing.add(piece)
            continue
        style = style_for(md_path.stem)
        metadata["prompt"] = caption(style, tags[piece], row["artist"])
        counts[style] += 1
        (TARGET / md_path.name).write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        latent = TARGET / md_path.with_suffix(".npy").name
        if not latent.exists():
            os.link(md_path.with_suffix(".npy"), latent)
    if missing:
        raise SystemExit(f"Pieces without tags: {sorted(missing)}")
    silence = TARGET / "silence.npy"
    if not silence.exists():
        os.link(SOURCE / "silence.npy", silence)
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
