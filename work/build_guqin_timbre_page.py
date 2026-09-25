"""Build the local timbre-first listening page from finished audio probes."""

import json
from pathlib import Path

ROOT = Path(__file__).parent
OUTPUT = ROOT.parent / "outputs" / "古琴音色诊断"
RADIO = ROOT.parent / "outputs" / "古琴连续生成"


def bars(values, maximum, bad):
    return "".join(
        f'<div class="barcell{" bad" if bad else ""}"><span style="width:{max(1, 100 * value / maximum):.1f}%"></span><em>{value:.3f}</em></div>'
        for value in values
    )


def main():
    wave = json.loads((RADIO / "生成记录.json").read_text(encoding="utf-8"))
    token = json.loads((RADIO / "音频符号接续记录.json").read_text(encoding="utf-8"))
    wave_values = [row["rms_new"] for row in wave["chunks"]]
    token_values = [row["rms_new"] for row in token["chunks"]]
    assert len(wave_values) == len(token_values) == 6
    maximum = max(wave_values + token_values)
    page = (ROOT / "guqin_timbre_page_template.html").read_text(encoding="utf-8")
    page = page.replace("__WAVE_BARS__", bars(wave_values, maximum, True))
    page = page.replace("__TOKEN_BARS__", bars(token_values, maximum, False))
    audio_only = RADIO / "纯音频接续_无文字提示.mp3"
    if audio_only.exists():
        content = '<section><h2>三、完全不使用文字，会怎样？</h2><p class="section-note">只用前面的古琴声音引导。试听已确认：开始反复重复，随后出现强烈金属噪声。</p><div class="card"><span class="tag">失败样本</span><h3>无文字的音频符号接续</h3><audio controls preload="none" src="../古琴连续生成/纯音频接续_无文字提示.mp3"></audio></div></section>'
    else:
        content = ""
    page = page.replace("__AUDIO_ONLY_SECTION__", content)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "index.html"
    path.write_text(page, encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
