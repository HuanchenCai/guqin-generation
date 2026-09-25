"""Listening page for the latent LM prototype, in the usual page format."""

import html
import json
import random
from pathlib import Path

from build_guqin_inference_page import SCRIPT, STYLES

OUT = Path(__file__).resolve().parent.parent / "outputs" / "古琴潜空间续写"
ISSUES = ("拉锯/金属杂音", "不像古琴", "重复", "没旋律", "杂音", "接缝突兀", "音量不稳")
LEGEND = {"latent_lm": "新：潜空间语言模型", "sa3": "对照：Stable Audio 3 适配器（同一 329 首训练集）"}


def rating(rid: str) -> str:
    return (f'<div class="rating" data-id="{html.escape(rid, quote=True)}">'
            '<label>综合评分 <select><option value="">未评分</option>'
            + ''.join(f'<option value="{n}">{n}</option>' for n in range(1, 6))
            + '</select></label><div class="issues">'
            + ''.join(f'<label><input type="checkbox" value="{x}">{x}</label>' for x in ISSUES)
            + '</div><input class="comment" placeholder="其他听感（可选）"></div>')


def card(title: str, note: str, file: str, rid: str | None, jump: bool = False, who: str = "") -> str:
    return (f'<article class="card"><div class="badge">{title}</div>'
            f'<p class="who" data-who="{html.escape(who or note, quote=True)}">{note}</p>'
            f'<audio controls preload="none" src="audio/{file}"></audio>'
            + ('<button class="jump" type="button">从续写处听</button>' if jump else '')
            + (rating(rid) if rid else '') + '</article>')


def main() -> None:
    rec = json.loads((OUT / "样本记录.json").read_text(encoding="utf-8"))
    sections = []
    for i, c in enumerate(rec["continuations"], 1):
        cands = [(k, c[k]["file"]) for k in ("latent_lm", "sa3") if k in c]
        random.Random(c["file"]).shuffle(cands)
        parts = [card("原曲 · 10 秒", f'{c["artist"]}《{c["piece"]}》，经 SAME-L 编码再解码。', c["file"], None)]
        parts += [card(f"候选 {l}", "前 10 秒是原曲开头，后 50 秒为续写。", f, f, True, LEGEND[k])
                  for l, (k, f) in zip("甲乙", cands)]
        sections.append(f'<section class="case"><h2>{c["piece"]} · {c["artist"]}</h2>'
                        '<p class="fine">这首曲子没有参与两个模型的训练。两个候选顺序已打乱。</p>'
                        f'<div class="grid">{"".join(parts)}</div></section>')
    endless = [card(f"连续 5 分钟 · 随机度 {e['temperature']}",
                    f"从第一组的开头出发，一直往下生成。音量 {e['rms_dbfs']} dB。", e["file"], e["file"])
               for e in rec["endless"]]
    free = [card(f"不给开头 · 第 {i} 段", "从零开始生成 60 秒。", f["file"], f["file"])
            for i, f in enumerate(rec["free"], 1)]
    page = ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>古琴潜空间续写试听</title><style>' + STYLES + '</style></head><body>'
            '<header><div><h1>古琴潜空间续写试听</h1>'
            '<p>不用文字：模型读前面的声音表征、逐帧预测后面的声音，理论上可以一直播放。这是一晚训练的原型。</p></div></header><main>'
            f'<div class="intro">新模型只在 329 首上从零训练了一晚（{html.escape(rec["checkpoint"])}），'
            '对照是同一训练集的 Stable Audio 3 适配器。先听续写，判断音色和旋律；再听连续 5 分钟，重点听后段有没有越来越乱。'
            '<a href="../古琴AI电台策略.md">查看策略说明</a></div>'
            + ''.join(sections)
            + '<section class="case"><h2>一直播放</h2><p class="fine">两种随机度：0.8 更保守，1.0 更多变化。</p>'
            f'<div class="grid">{"".join(endless)}</div></section>'
            '<section class="case"><h2>不给开头</h2><p class="fine">完全从零开始，看它自己会弹出什么。</p>'
            f'<div class="grid">{"".join(free)}</div></section>'
            '<button class="export" id="export">下载统一评分</button>'
            '<p class="fine">评分自动保存在当前浏览器；下载 JSON 后可以直接发给我。'
            '<a href="#" id="reveal">评完后可显示候选对应关系</a></p></main>'
            '<script>' + SCRIPT.replace("guqin_inference_sweep_v4", "guqin_latent_lm_v1")
            .replace("古琴推理调参统一评分.json", "古琴潜空间续写统一评分.json") + '</script></body></html>')
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
