"""Listening page for the semitone experiment, in the usual page format."""

import html
import json
import random
from pathlib import Path

from build_guqin_inference_page import SCRIPT, STYLES

OUT = Path(__file__).resolve().parent.parent / "outputs" / "古琴半音对照"
ISSUES = ("半音多", "拉锯/金属杂音", "不像古琴", "重复开头", "没旋律", "接缝突兀", "音量不稳")
LEGEND = {"A": "A：训练 1 遍，无文字", "B": "B：训练 3 遍，写意境标签", "C": "C：训练 3 遍，写意境标签和调式"}


def rating(rid: str) -> str:
    return (f'<div class="rating" data-id="{html.escape(rid, quote=True)}">'
            '<label>综合评分 <select><option value="">未评分</option>'
            + ''.join(f'<option value="{n}">{n}</option>' for n in range(1, 6))
            + '</select></label><div class="issues">'
            + ''.join(f'<label><input type="checkbox" value="{x}">{x}</label>' for x in ISSUES)
            + '</div><input class="comment" placeholder="其他听感（可选），比如第几秒出现半音"></div>')


def main() -> None:
    rows = json.loads((OUT / "样本记录.json").read_text(encoding="utf-8"))
    items = json.loads((OUT / "openings.json").read_text(encoding="utf-8"))
    summary = json.loads((OUT / "客观指标.json").read_text(encoding="utf-8"))
    first_seed = min(r["seed"] for r in rows)
    sections = []
    for item in items:
        cards = [r for r in rows if r["opening"] == item["file"] and r["seed"] == first_seed]
        random.Random(item["file"]).shuffle(cards)
        parts = [f'<article class="card"><div class="badge">原曲 · 10 秒</div>'
                 f'<p>{html.escape(item["artist"])}《{html.escape(item["piece"])}》</p>'
                 f'<audio controls preload="none" src="audio/{item["file"]}"></audio></article>']
        for label, r in zip("甲乙丙", cards):
            parts.append(f'<article class="card"><div class="badge">候选 {label}</div>'
                         f'<p class="who" data-who="{html.escape(LEGEND[r["model"]], quote=True)}">前 10 秒是原曲开头，后 25 秒为续写。</p>'
                         f'<audio controls preload="none" src="audio/{r["file"]}"></audio>'
                         f'<button class="jump" type="button">从续写处听</button>{rating(r["file"])}</article>')
        sections.append(f'<section class="case"><h2>{html.escape(item["piece"])} · {html.escape(item["artist"])}</h2>'
                        '<p class="fine">这首曲子没有参与三个模型的训练。三个候选顺序已打乱。</p>'
                        f'<div class="grid">{"".join(parts)}</div></section>')
    rate = lambda code: f'{summary[code]["attack_out_of_mode"] * 100:.0f}%'
    page = ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>古琴半音对照试听</title><style>' + STYLES + '</style></head><body>'
            '<header><div><h1>古琴半音对照试听</h1>'
            '<p>多训练几遍、把调式写进文字，能不能减少弹下去就是半音的情况。</p></div></header><main>'
            '<div class="intro">三个模型都只用同一批 329 首训练，下面 8 首曲子、8 位演奏者都没有参与训练。'
            '<strong>重点听半音：</strong>弹下去的音是不是跑出调式，滑音、吟猱里的半音不算。'
            f'程序测得起音落在调式外的比例：真实原曲 {summary["real_openings_attack_out_of_mode"] * 100:.0f}%；'
            '三个模型分别为 ' + '、'.join(rate(c) for c in "ABC") + '（评完再看对应关系）。'
            '这只是粗略测量，以你的听感为准。</div>'
            + ''.join(sections)
            + '<button class="export" id="export">下载统一评分</button>'
            '<p class="fine">评分自动保存在当前浏览器；下载 JSON 后可以直接发给我。'
            '<a href="#" id="reveal">评完后可显示候选对应关系</a></p></main>'
            '<script>' + SCRIPT.replace("guqin_inference_sweep_v4", "guqin_semitone_v1")
            .replace("古琴推理调参统一评分.json", "古琴半音对照统一评分.json") + '</script></body></html>')
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
