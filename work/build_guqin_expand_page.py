"""Listening page for the 渔舟晚唱 extension and extra takes, usual format."""

import html
import json
import os
from pathlib import Path

from build_guqin_inference_page import SCRIPT, STYLES
from generate_guqin_pass14_expand import CONTEXT, FAVOURITE, NEW

OUT = Path(__file__).resolve().parent.parent / "outputs" / "古琴渔舟晚唱拓展"
ISSUES = ("半音多", "拉锯/金属杂音", "不像古琴", "没旋律", "接缝突兀", "越来越乱", "音量不稳")


def rating(rid: str) -> str:
    return (f'<div class="rating" data-id="{html.escape(rid, quote=True)}">'
            '<label>综合评分 <select><option value="">未评分</option>'
            + ''.join(f'<option value="{n}">{n}</option>' for n in range(1, 6))
            + '</select></label><div class="issues">'
            + ''.join(f'<label><input type="checkbox" value="{x}">{x}</label>' for x in ISSUES)
            + '</div><input class="comment" placeholder="其他听感；好的请写“入选”，问题请写大概第几分几秒"></div>')


def main() -> None:
    data = json.loads((OUT / "样本记录.json").read_text(encoding="utf-8"))
    original = OUT / "audio" / "original_s3_pass14.flac"
    if not original.exists():
        os.link(FAVOURITE, original)
    seeks = lambda secs: '<div class="seeks">' + ''.join(
        f'<button type="button" data-seek="{t}">{t // 60}:{t % 60:02d}</button>' for t in secs) + '</div>'
    chain_cards = [f'<article class="card"><div class="badge">原版 · 35 秒</div><p>上次你评为“绝品”的那段。</p>'
                   f'<audio controls preload="none" src="audio/original_s3_pass14.flac"></audio></article>']
    for c in sorted(data["chains"], key=lambda c: ("lock_db" not in c, c["file"])):
        joins = [35 + k * NEW for k in range(len(c["rounds"]))]
        levels = "、".join(f'{r["new_rms_dbfs"]}' for r in c["rounds"])
        locked = "lock_db" in c
        title = f'接续版 {c["file"][7]}' + ("（音量锁定）" if locked else "（未锁定，越接越响）")
        note = ("整首先降到 -16 dB，每段接续都按前一段音量对齐。" if locked
                else "每段越来越响、后半削波明显，保留作对比。")
        chain_cards.append(f'<article class="card"><div class="badge">{title} · {int(c["seconds"] // 60)} 分 {int(c["seconds"] % 60)} 秒</div>'
                           f'<p>{note}前 35 秒是原版，之后每 30 秒接一段（按钮是各接缝处）。每段音量 dB：{levels}。</p>'
                           f'<audio controls preload="none" src="audio/{c["file"]}"></audio>{seeks(joins)}{rating(c["file"])}</article>')
    take_cards = [f'<article class="card"><div class="badge">第 {k} 版</div><p>种子 {t["seed"]}</p>'
                  f'<audio controls preload="none" src="audio/{t["file"]}"></audio>{rating(t["file"])}</article>'
                  for k, t in enumerate(data["takes"], 1)]
    script = (SCRIPT.replace("guqin_inference_sweep_v4", "guqin_fisher_expand_v1")
              .replace("古琴推理调参统一评分.json", "古琴渔舟晚唱拓展统一评分.json")
              + "document.querySelectorAll('[data-seek]').forEach(b=>b.onclick=()=>{const a=b.closest('.card')"
                ".querySelector('audio');a.currentTime=Math.max(0,Number(b.dataset.seek)-3);a.play()});")
    prompt = data["prompt"].split("zither. ", 1)[1]
    page = ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1"><title>古琴渔舟晚唱拓展</title>'
            '<style>' + STYLES + '.seeks{display:flex;gap:6px;flex-wrap:wrap}.seeks button{border:1px solid #cad9cc;'
            'background:#edf2ea;color:var(--accent);border-radius:6px;padding:4px 8px}</style></head><body>'
            '<header><div><h1>渔舟晚唱 · 拓展</h1>'
            f'<p>第 14 遍模型。场景：{html.escape(prompt)}</p></div></header><main>'
            f'<div class="intro">上半部分：从“绝品”那段出发，每次取最后 {CONTEXT} 秒当开头、再往下接 {NEW} 秒，共接 6 次，'
            '得到约 3 分半的整曲，两种随机种子各一版。重点听接缝处（按钮会跳到接缝前 3 秒）和后半段有没有越来越乱。'
            '</div>'
            '<section class="case"><h2>接续成 3 分半</h2><div class="grid">' + ''.join(chain_cards) + '</div></section>'
            + ('<section class="case"><h2>同一场景多版</h2><div class="grid">' + ''.join(take_cards) + '</div></section>' if take_cards else '')
            + '<button class="export" id="export">下载统一评分</button>'
            '<p class="fine">评分自动保存在当前浏览器；下载 JSON 后可以直接发给我。</p></main>'
            '<script>' + script + '</script></body></html>')
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
