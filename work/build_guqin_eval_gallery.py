"""Package the unified listening and scoring gallery as a local offline page."""

import json
import random
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
RUN = ROOT / "guqin_eval_v3"
DEST = ROOT.parent / "outputs" / "古琴试听台"
BATCH = DEST / "batch"
LABELS = {
    "ace_sft_old": "ACE-Step · 旧版古琴微调",
    "ace_sft_new": "ACE-Step · 新版古琴微调",
    "ace_base": "ACE-Step Base · 原模型",
    "ace_turbo": "ACE-Step Turbo · 原模型",
    "musicgen_small": "MusicGen Small · 原模型",
    "audioldm2_music": "AudioLDM2 Music · 原模型",
}
REPS = {
    "waveform_latent": "波形潜变量",
    "codec_tokens": "音频编码器离散符号",
    "mel_spectrogram_latent": "梅尔频谱潜变量",
}
BRIEFS = {
    "夜深·留白": "听长音、按音与句子之间的呼吸。",
    "流水·行进": "听旋律能否流动，又不变成密集的琶音。",
    "日暮·回旋": "听短小主题是否自然回返与变化。",
    "清晨·泛音": "听泛音、空弦和木质共鸣是否自然。",
}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    BATCH.mkdir(parents=True, exist_ok=True)
    old = DEST / "index.html"
    history = DEST / "历史对照.html"
    if old.exists() and not history.exists():
        shutil.copy2(old, history)

    free = []
    for model in ("ace_sft_old", "ace_base", "musicgen_small", "audioldm2_music"):
        path = RUN / f"records_{model}.json"
        free.extend(json.loads(path.read_text(encoding="utf-8")))
    continuation = []
    for model in ("ace_sft_old", "ace_sft_new", "ace_base", "ace_turbo", "musicgen_small"):
        path = RUN / f"record_{model}_continuation.json"
        continuation.append(json.loads(path.read_text(encoding="utf-8")))

    candidates = []
    for r in free + continuation:
        source = Path(r["path"])
        assert source.is_file() and source.stat().st_size > 10000, source
        target = BATCH / source.name
        shutil.copy2(source, target)
        item = dict(id=r["id"], model=r["model"], model_label=LABELS[r["model"]],
                    representation=r["representation"], representation_label=REPS[r["representation"]],
                    task=r.get("task", "free"), src=f"batch/{source.name}",
                    duration_s=r["duration_s"], brief=r.get("brief", "同曲续写"),
                    brief_description=BRIEFS.get(r.get("brief"), ""))
        candidates.append(item)
    shutil.copy2(RUN / "reference_10s.mp3", BATCH / "reference_10s.mp3")

    rng = random.Random(20260923)
    cont = [x for x in candidates if x["task"] == "continuation"]
    rng.shuffle(cont)
    ordered = cont
    for title in BRIEFS:
        group = [x for x in candidates if x["task"] == "free" and x["brief"] == title]
        rng.shuffle(group)
        ordered.extend(group)
    assert len(ordered) == 21, len(ordered)
    (DEST / "candidates.js").write_text("window.GUQIN_CANDIDATES = " + json.dumps(ordered, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    shutil.copy2(ROOT / "guqin_eval_gallery_template.html", old)
    (DEST / "试听说明.json").write_text(json.dumps({
        "candidate_count": len(ordered), "continuation_count": len(cont),
        "free_count": len(free), "source": "蔡德允《平沙落雁》训练集之外的30秒片段",
        "reference_audio": "batch/reference_10s.mp3",
        "description": "相同文字提示的四种意境与相同10秒音频开头续写；一个音频一个总分。",
        "limitation": "不同模型训练语料、参数量及生成时长不同，不能由试听结果独立推断表征优劣。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    zip_path = DEST.parent / "古琴试听台.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for file in DEST.rglob("*"):
            if file.is_file():
                z.write(file, file.relative_to(DEST.parent))
    print(json.dumps({"page": str(old), "zip": str(zip_path), "candidates": len(ordered)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
