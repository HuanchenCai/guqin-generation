"""Collect generated guqin samples into one portable local listening gallery."""

import json
import shutil
from pathlib import Path

import numpy as np
import torchaudio


ROOT = Path(__file__).parent.parent
OUTPUTS = ROOT / "outputs"
GALLERY = OUTPUTS / "古琴试听台"
ASSETS = GALLERY / "audio"


PAIRS = [
    {
        "id": "strict-5070",
        "title": "版本对比｜同条件生成",
        "description": "两版使用同一段文字描述、相同随机起点和 30 秒时长。新版中段有你指出的急促声。",
        "old": ("古琴SFT试听/SFT微调_严格同提示_5070.mp3", "旧版"),
        "new": ("古琴新SFT试听/新版SFT_严格同提示_5070.mp3", "新版（30 轮）"),
        "note": "目前偏好：旧版更好听；新版中段出现突兀的急促声。",
    },
    {
        "id": "rounds-20-30",
        "title": "训练轮次对比｜20 轮与 30 轮",
        "description": "同一个新训练集、同一段文字描述和随机起点；只比较训练中途与最终版本。",
        "old": ("古琴新SFT试听/新版SFT_第20轮_严格同提示_5070.mp3", "训练 20 轮"),
        "new": ("古琴新SFT试听/新版SFT_严格同提示_5070.mp3", "训练 30 轮"),
    },
    {
        "id": "tag-5070",
        "title": "版本对比｜强调古琴独奏",
        "description": "文字描述额外强调只由一人弹古琴；两版仍使用相同的生成条件。",
        "old": ("古琴SFT试听/SFT微调_带标签_5070.mp3", "旧版"),
        "new": ("古琴新SFT试听/新版SFT_带标签_5070.mp3", "新版（30 轮）"),
    },
    {
        "id": "tag-5071",
        "title": "版本对比｜另一段独奏",
        "description": "和上一组使用相同文字描述，换一个随机起点，检查听感是否稳定。",
        "old": ("古琴SFT试听/SFT微调_带标签_5071.mp3", "旧版"),
        "new": ("古琴新SFT试听/新版SFT_带标签_5071.mp3", "新版（30 轮）"),
    },
    {
        "id": "continue-5074",
        "title": "续写对比｜听前 10 秒，再听续写",
        "description": "两版都保留同一原曲开头，10 秒后各自续写；原曲录音本身较旧。",
        "old": ("古琴第二版试听/SFT续写_前10秒原曲.mp3", "旧版续写"),
        "new": ("古琴新SFT试听/新版SFT_续写对照_前10秒原曲.mp3", "新版续写"),
        "note": "新版接缝仍偏安静，暂不能视为续写已修好。",
    },
]


EXTRAS = [
    ("古琴试听对比/01_基础模型.mp3", "最早的基础模型", "早期探索"),
    ("古琴试听对比/02_古琴微调_同种子.mp3", "第一次古琴微调", "早期探索"),
    ("古琴试听对比/03_古琴微调_另一种子.mp3", "第一次古琴微调（另一段）", "早期探索"),
    ("古琴单乐器试验/01_修正提示词_基础模型.mp3", "强调独奏后的基础模型", "独奏尝试"),
    ("古琴单乐器试验/02_修正提示词_旧微调.mp3", "强调独奏后的早期微调", "独奏尝试"),
    ("古琴单乐器试验/03_修正提示词_参考音色.mp3", "加入参考音色", "独奏尝试"),
    ("古琴单乐器试验/04_ACE基础版_高约束.mp3", "基础模型（更详细的古琴描述）", "独奏尝试"),
    ("古琴单乐器试验/05_ACE指令版_高约束.mp3", "指令模型（更详细的古琴描述）", "独奏尝试"),
    ("古琴第二版试听/单古琴_5070.mp3", "独奏尝试 1", "后续探索"),
    ("古琴第二版试听/单古琴_5071.mp3", "独奏尝试 2", "后续探索"),
    ("古琴第二版试听/单古琴_5072.mp3", "独奏尝试 3", "后续探索"),
    ("古琴第二版试听/单古琴_5073.mp3", "独奏尝试 4", "后续探索"),
    ("古琴第二版试听/单古琴_细节提示_5070.mp3", "独奏尝试（加入演奏细节）", "后续探索"),
    ("古琴第二版试听/单古琴_严格同提示_5070.mp3", "独奏尝试（统一文字描述）", "后续探索"),
    ("古琴第二版试听/音频续写_前10秒原曲.mp3", "早期续写（前 10 秒原曲）", "后续探索"),
]


REFERENCES = [
    ("古琴新素材筛选/较晚录音候选审听.mp3", "10 首候选原曲 · 审听合辑"),
    ("古琴素材审听/训练素材逐段审听.mp3", "早期训练素材 · 审听合辑"),
]


def waveform_summary(waveform: np.ndarray, bars: int = 72) -> list[int]:
    mono = waveform.mean(axis=0).astype(np.float64)
    chunks = np.array_split(mono, bars)
    rms = np.array([np.sqrt(np.mean(chunk * chunk)) for chunk in chunks])
    scale = max(float(np.percentile(rms, 95)), 1e-8)
    return np.clip(np.sqrt(rms / scale) * 100, 6, 100).astype(int).tolist()


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    catalog = {}
    path_keys = {}

    def add(relative_path: str, label: str) -> str:
        if relative_path in path_keys:
            return path_keys[relative_path]
        key = f"sample-{len(catalog) + 1:02d}"
        source = OUTPUTS / relative_path
        if not source.is_file():
            raise FileNotFoundError(source)
        target = ASSETS / f"{key}.mp3"
        shutil.copy2(source, target)
        audio, rate = torchaudio.load(str(source))
        catalog[key] = {
            "id": key,
            "title": label,
            "src": f"audio/{target.name}",
            "duration": round(audio.shape[1] / rate, 1),
            "waveform": waveform_summary(audio.numpy()),
            "source": relative_path,
        }
        path_keys[relative_path] = key
        return key

    pairs = []
    for pair in PAIRS:
        old_path, old_label = pair["old"]
        new_path, new_label = pair["new"]
        pairs.append({
            "id": pair["id"],
            "title": pair["title"],
            "description": pair["description"],
            "note": pair.get("note", ""),
            "old": add(old_path, old_label),
            "new": add(new_path, new_label),
        })

    extras = [
        {"sample": add(path, title), "group": group}
        for path, title, group in EXTRAS
    ]
    references = [
        {"sample": add(path, title), "group": "原曲审听"}
        for path, title in REFERENCES
    ]
    data = {"pairs": pairs, "extras": extras, "references": references, "samples": catalog}
    (GALLERY / "data.js").write_text(
        "window.GUQIN_GALLERY_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    print(f"Gallery: {GALLERY / 'index.html'}")
    print(f"Copied {len(catalog)} audio files; {len(pairs)} matched comparisons")


if __name__ == "__main__":
    main()
