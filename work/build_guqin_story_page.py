"""Listening page for the story-caption version, in the usual page format."""

import html
import json
import random
from pathlib import Path

from build_guqin_inference_page import SCRIPT, STYLES

OUT = Path(__file__).resolve().parent.parent / "outputs" / "古琴故事版试听"
ISSUES = ("半音多", "拉锯/金属杂音", "不像古琴", "重复开头", "没旋律", "接缝突兀", "音量不稳")
TEXT_ISSUES = ("不符合描述",) + ISSUES
LEGEND = {"new": "新版：370 首、故事 + 标签 + 调式，第 2 遍", "C": "旧版 C：329 首、标签 + 调式，3 遍"}


def rating(rid: str, issues) -> str:
    return (f'<div class="rating" data-id="{html.escape(rid, quote=True)}">'
            '<label>综合评分 <select><option value="">未评分</option>'
            + ''.join(f'<option value="{n}">{n}</option>' for n in range(1, 6))
            + '</select></label><div class="issues">'
            + ''.join(f'<label><input type="checkbox" value="{x}">{x}</label>' for x in issues)
            + '</div><input class="comment" placeholder="其他听感（可选）"></div>')


def main() -> None:
    data = json.loads((OUT / "样本记录.json").read_text(encoding="utf-8"))
    sections = []
    for i, item in enumerate(data["openings"], 1):
        cands = [("new", f"o{i:02d}_new.flac"), ("C", f"o{i:02d}_C.flac")]
        random.Random(item["file"]).shuffle(cands)
        parts = [f'<article class="card"><div class="badge">原曲 · 10 秒</div>'
                 f'<p>{html.escape(item["artist"])}《{html.escape(item["piece"])}》</p>'
                 f'<audio controls preload="none" src="audio/{item["file"]}"></audio></article>']
        parts += [f'<article class="card"><div class="badge">候选 {label}</div>'
                  f'<p class="who" data-who="{html.escape(LEGEND[code], quote=True)}">前 10 秒是原曲开头，后 25 秒为续写。</p>'
                  f'<audio controls preload="none" src="audio/{f}"></audio>'
                  f'<button class="jump" type="button">从续写处听</button>{rating(f, ISSUES)}</article>'
                  for label, (code, f) in zip("甲乙", cands)]
        sections.append(f'<section class="case"><h2>{html.escape(item["piece"])} · {html.escape(item["artist"])}</h2>'
                        '<p class="fine">这首曲子没有参与两个版本的训练，两个候选顺序已打乱。</p>'
                        f'<div class="grid">{"".join(parts)}</div></section>')
    text_cards = [f'<article class="card"><div class="badge">只写文字</div><p>{html.escape(t["label"])}</p>'
                  f'<audio controls preload="none" src="audio/{t["file"]}"></audio>{rating(t["file"], TEXT_ISSUES)}</article>'
                  for t in data["texts"]]
    page = ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>古琴故事版试听</title><style>' + STYLES + '</style></head><body>'
            '<header><div><h1>古琴故事版试听</h1>'
            '<p>370 首训练、文字里写入曲目故事的新版本（第 2 遍检查点），与上一版最好的 C 对比。</p></div></header><main>'
            '<div class="intro">前半部分：8 首验证集曲目、8 位演奏者，两个版本都没训练过。'
            '后半部分：不给开头，只写意境标签或一段场景描述，看新版能否按文字生成。</div>'
            + ''.join(sections)
            + '<section class="case"><h2>只写文字，不给开头</h2><p class="fine">第 1–5 条是意境标签，第 6 条是一段自由描写的场景。</p>'
            f'<div class="grid">{"".join(text_cards)}</div></section>'
            '<button class="export" id="export">下载统一评分</button>'
            '<p class="fine">评分自动保存在当前浏览器；下载 JSON 后可以直接发给我。'
            '<a href="#" id="reveal">评完后可显示候选对应关系</a></p></main>'
            '<script>' + SCRIPT.replace("guqin_inference_sweep_v4", "guqin_story_v1")
            .replace("古琴推理调参统一评分.json", "古琴故事版统一评分.json") + '</script></body></html>')
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
