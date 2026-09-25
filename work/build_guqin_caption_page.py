"""Listening page for the text-caption experiment, in the usual page format."""

import html
import json
import random
from pathlib import Path

from build_guqin_inference_page import SCRIPT, STYLES

OUT = Path(__file__).resolve().parent.parent / "outputs" / "古琴文字标签试听"
ISSUES = ("拉锯/金属杂音", "不像古琴", "重复", "没旋律", "杂音", "接缝突兀", "音量不稳")
TEXT_ISSUES = ("不符合标签",) + ISSUES
LEGEND = {"old": "原 412 首版本（训练时无文字）", "none": "新版本 · 不写文字",
          "match": "新版本 · 写对应意境", "opposite": "新版本 · 故意写相反意境",
          "new": "新版本（训练时带文字）"}


def rating(rid: str, issues) -> str:
    return (f'<div class="rating" data-id="{html.escape(rid, quote=True)}">'
            '<label>综合评分 <select><option value="">未评分</option>'
            + ''.join(f'<option value="{n}">{n}</option>' for n in range(1, 6))
            + '</select></label><div class="issues">'
            + ''.join(f'<label><input type="checkbox" value="{x}">{x}</label>' for x in issues)
            + '</div><input class="comment" placeholder="其他听感（可选）"></div>')


def candidate(label: str, row: dict, note: str, issues, jump: bool) -> str:
    who = LEGEND[row["code"]] + (f"：{'、'.join(row['tags'])}" if row["tags"] else "")
    return (f'<article class="card"><div class="badge">候选 {label}</div>'
            f'<p class="who" data-who="{html.escape(who, quote=True)}">{note}</p>'
            f'<audio controls preload="none" src="audio/{row["file"]}"></audio>'
            + ('<button class="jump" type="button">从续写处听</button>' if jump else '')
            + rating(row["file"], issues) + '</article>')


def main() -> None:
    rows = json.loads((OUT / "样本记录.json").read_text(encoding="utf-8"))
    openings = json.loads((OUT / "openings.json").read_text(encoding="utf-8"))
    sections = []
    for o in openings:
        cards = [r for r in rows if r["part"] == "continue" and r["opening"] == o["file"]]
        random.Random(o["file"]).shuffle(cards)
        parts = [f'<article class="card"><div class="badge">原曲 · 10 秒</div>'
                 f'<p>{o["artist"]}《{o["piece"]}》{o["start_seconds"]} 秒起。</p>'
                 f'<audio controls preload="none" src="audio/{o["file"]}"></audio>'
                 f'<p>这首的意境标签：{"、".join(o["match"])}</p></article>']
        parts += [candidate(l, r, "前 10 秒是原曲开头，后 25 秒为续写。", ISSUES, True)
                  for l, r in zip("甲乙丙丁", cards)]
        sections.append(f'<section class="case"><h2>{o["piece"]} · {o["artist"]}</h2>'
                        '<p class="fine">这首曲子和演奏者都没有参与训练。同一开头和种子，四个候选顺序已打乱。</p>'
                        f'<div class="grid">{"".join(parts)}</div></section>')
    text_groups = {}
    for r in rows:
        if r["part"] == "text":
            text_groups.setdefault(tuple(r["tags"]), []).append(r)
    text_cards = []
    for tags, group in text_groups.items():
        random.Random("|".join(tags)).shuffle(group)
        text_cards.append(f'<article class="card"><div class="badge">标签：{"、".join(tags)}</div>'
                          '<p>没有开头，只凭文字从头生成 35 秒。</p></article>')
        text_cards += [candidate(l, r, "判断它是否符合这组标签。", TEXT_ISSUES, False)
                       for l, r in zip("甲乙", group)]
    text_section = ('<section class="case"><h2>只写标签，不给开头</h2>'
                    '<p class="fine">每行第一格是文字标签，后两格是新旧两个版本，顺序已打乱。</p>'
                    f'<div class="grid">{"".join(text_cards)}</div></section>')
    script = (SCRIPT.replace("guqin_inference_sweep_v4", "guqin_caption_test_v1")
              .replace("古琴推理调参统一评分.json", "古琴文字标签统一评分.json"))
    page = ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>古琴文字标签试听</title><style>' + STYLES + '</style></head><body>'
            '<header><div><h1>古琴文字标签试听</h1>'
            '<p>412 首重新训练时写入了曲名和意境标签（如平静、孤寂、塞外大漠），看文字能否影响生成。</p></div></header><main>'
            '<div class="intro">前半部分：10 首曲子、10 位演奏者各不相同，都没有参与训练。每组四个候选，'
            '其中有的写了对应意境、有的故意写了相反意境，听完可看对应关系。后半部分：不给开头，只写标签，比较新旧版本。'
            '<a href="../古琴推理调参/index.html">上一轮推理调参页</a></div>'
            + ''.join(sections) + text_section
            + '<button class="export" id="export">下载统一评分</button>'
            '<p class="fine">评分自动保存在当前浏览器；下载 JSON 后可以直接发给我。'
            '<a href="#" id="reveal">评完后可显示候选对应关系</a></p></main>'
            '<script>' + script + '</script></body></html>')
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(f"{len(sections)} opening groups, {len(text_groups)} text groups -> {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
