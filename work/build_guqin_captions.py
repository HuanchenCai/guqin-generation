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
HEAD_TEXT = "Solo guqin, Chinese seven-string zither."
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
    head = HEAD_TEXT
    piece = f"Piece: {row['拼音']} ({row['英文意译']}) {row['曲名']}. Performer: {performer}."
    if style == "tags":
        return f"{head} {tag_text(row['tags'])}"
    if style == "piece":
        return f"{head} {piece}"
    return f"{head} {tag_text(row['tags'])} {piece}"


def style_for(name: str) -> str:
    bucket = int(hashlib.sha1(name.encode()).hexdigest(), 16) % 10
    return "full" if bucket < 5 else "tags" if bucket < 8 else "piece"


NOTE_NAMES = ("A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#")


def window_modes(files: list[str]) -> dict:
    """Pentatonic mode of each window, estimated from its audio (cached)."""
    from guqin_pitch_metric import best_pentatonic, cent_histogram, load_mono, pitch_classes, tuning_offset
    cache_path = ROOT / "window_modes.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    for i, name in enumerate(files):
        if name in cache:
            continue
        hist = cent_histogram(load_mono(ROOT / "audio" / name))
        fit, mode = best_pentatonic(pitch_classes(hist, tuning_offset(hist)))
        gong = int(mode[0])  # PENTATONIC starts at 0, so mode[0] is the gong note
        cache[name] = {"gong": gong, "fit": round(float(fit), 4)}
        if i % 200 == 0:
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
            print(f"modes {i}/{len(files)}", flush=True)
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return cache


def mode_text(gong: int) -> str:
    notes = [NOTE_NAMES[(gong + step) % 12] for step in (0, 2, 4, 7, 9)]
    return f"Pentatonic mode, gong on {notes[0]}: {', '.join(notes)}. 五声调式，{notes[0]} 宫。"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="all", choices=("all", "train", "validation"))
    parser.add_argument("--mode", action="store_true", help="append the estimated pentatonic mode")
    parser.add_argument("--target", default=str(TARGET))
    args = parser.parse_args()
    target = Path(args.target)
    tags = load_tags()
    with (ROOT / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        manifest = {row["file"]: row for row in csv.DictReader(handle)}
    target.mkdir(exist_ok=True)
    sources = sorted(SOURCE.glob("*.json"))
    rows = {p: manifest[Path(json.loads(p.read_text(encoding="utf-8"))["path"]).name] for p in sources}
    if args.split != "all":
        rows = {p: r for p, r in rows.items() if r["split"] == args.split}
    modes = window_modes([r["file"] for r in rows.values()]) if args.mode else {}
    counts = {"full": 0, "tags": 0, "piece": 0}
    missing = set()
    for md_path, row in rows.items():
        metadata = json.loads(md_path.read_text(encoding="utf-8"))
        piece = normalize(row["piece"])
        if piece not in tags:
            missing.add(piece)
            continue
        style = style_for(md_path.stem)
        metadata["prompt"] = caption(style, tags[piece], row["artist"])
        if args.mode:
            metadata["prompt"] += " " + mode_text(modes[row["file"]]["gong"])
        counts[style] += 1
        (target / md_path.name).write_text(json.dumps(metadata), encoding="utf-8")
        latent = target / md_path.with_suffix(".npy").name
        if not latent.exists():
            os.link(md_path.with_suffix(".npy"), latent)
    if missing:
        raise SystemExit(f"Pieces without tags: {sorted(missing)}")
    silence = target / "silence.npy"
    if not silence.exists():
        os.link(SOURCE / "silence.npy", silence)
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
