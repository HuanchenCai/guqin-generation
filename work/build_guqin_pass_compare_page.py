"""Listening page: pass 14 vs pass 2 (and C), usual page format."""

import html
import json
import random
from pathlib import Path

from build_guqin_inference_page import SCRIPT, STYLES

OUT = Path(__file__).resolve().parent.parent / "outputs" / "古琴第14遍对比"
ISSUES = ("半音多", "拉锯/金属杂音", "不像古琴", "重复开头", "没旋律", "接缝突兀", "音量不稳")
SCENE_ISSUES = ("不符合描述", "半音多", "拉锯/金属杂音", "不像古琴", "没旋律", "音量不稳")
LEGEND = {"pass2": "故事版第 2 遍", "pass14": "故事版第 14 遍", "C": "旧版 C（329 首，标签 + 调式，3 遍）"}


def rating(rid: str, issues) -> str:
    return (f'<div class="rating" data-id="{html.escape(rid, quote=True)}">'
            '<label>综合评分 <select><option value="">未评分</option>'
            + ''.join(f'<option value="{n}">{n}</option>' for n in range(1, 6))
            + '</select></label><div class="issues">'
            + ''.join(f'<label><input type="checkbox" value="{x}">{x}</label>' for x in issues)
            + '</div><input class="comment" placeholder="其他听感（可选）"></div>')


def candidate(label, code, file, note, issues, jump) -> str:
    return (f'<article class="card"><div class="badge">候选 {label}</div>'
            f'<p class="who" data-who="{html.escape(LEGEND[code], quote=True)}">{note}</p>'
            f'<audio controls preload="none" src="audio/{file}"></audio>'
            + ('<button class="jump" type="button">从续写处听</button>' if jump else '')
            + rating(file, issues) + '</article>')


def main() -> None:
    data = json.loads((OUT / "样本记录.json").read_text(encoding="utf-8"))
    sections = []
    for c in data["continuations"]:
        i = c["index"]
        cands = [(code, f"o{i:02d}_{code}.flac") for code in ("pass2", "pass14", "C")]
        random.Random(c["file"]).shuffle(cands)
        parts = [f'<article class="card"><div class="badge">原曲 · 10 秒</div>'
                 f'<p>{html.escape(c["artist"])}《{html.escape(c["piece"])}》</p>'
                 f'<audio controls preload="none" src="audio/{c["file"]}"></audio></article>']
        parts += [candidate(l, code, f, "前 10 秒是原曲开头，后 25 秒为续写。", ISSUES, True)
                  for l, (code, f) in zip("甲乙丙", cands)]
        sections.append(f'<section class="case"><h2>{html.escape(c["piece"])} · {html.escape(c["artist"])}</h2>'
                        '<p class="fine">三个版本都没训练过这首曲子，顺序已打乱；与上一轮用的是同一开头和种子。</p>'
                        f'<div class="grid">{"".join(parts)}</div></section>')
    scene_sections = []
    for s in data["scenes"]:
        k = s["index"]
        cands = [(code, f"s{k}_{code}.flac") for code in ("pass2", "pass14")]
        random.Random(s["scene"]).shuffle(cands)
        text = s["prompt"].split("zither. ", 1)[1]
        parts = [candidate(l, code, f, "不给开头，只写场景描述。", SCENE_ISSUES, False)
                 for l, (code, f) in zip("甲乙", cands)]
        scene_sections.append(f'<section class="case"><h2>{html.escape(s["scene"])}</h2>'
                              f'<p class="fine">{html.escape(text)}（同一种子；其中一版是你上次入选的那段）</p>'
                              f'<div class="grid">{"".join(parts)}</div></section>')
    page = ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>古琴第 14 遍对比</title><style>' + STYLES + '</style></head><body>'
            '<header><div><h1>古琴第 14 遍对比</h1>'
            '<p>故事版训练到第 14 遍（验证误差最低）与第 2 遍、旧版 C 的对比。</p></div></header><main>'
            '<div class="intro">前半部分：8 首验证集曲目的续写，三个候选。后半部分：展示 v0 的五个场景，'
            '第 2 遍和第 14 遍用同一种子各一版。</div>'
            + ''.join(sections)
            + '<h2 style="margin:10px 0 16px">只写场景，不给开头</h2>' + ''.join(scene_sections)
            + '<button class="export" id="export">下载统一评分</button>'
            '<p class="fine">评分自动保存在当前浏览器；下载 JSON 后可以直接发给我。'
            '<a href="#" id="reveal">评完后可显示候选对应关系</a></p></main>'
            '<script>' + SCRIPT.replace("guqin_inference_sweep_v4", "guqin_pass14_v1")
            .replace("古琴推理调参统一评分.json", "古琴第14遍对比统一评分.json") + '</script></body></html>')
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
