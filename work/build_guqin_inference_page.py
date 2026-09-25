"""Blind listening page for the guqin inference sweep.

Uses the same layout, styles and rating format as the earlier listening
pages (see build_sa3_overnight_page.py) so exports stay comparable.
"""

import html
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "outputs" / "古琴推理调参"
ISSUES = ("拉锯/金属杂音", "不像古琴", "重复", "没旋律", "杂音", "接缝突兀", "音量不稳")


def rating(rid: str) -> str:
    return (f'<div class="rating" data-id="{html.escape(rid, quote=True)}">'
            '<label>综合评分 <select><option value="">未评分</option>'
            + ''.join(f'<option value="{n}">{n}</option>' for n in range(1, 6))
            + '</select></label><div class="issues">'
            + ''.join(f'<label><input type="checkbox" value="{x}">{x}</label>' for x in ISSUES)
            + '</div><input class="comment" placeholder="其他听感（可选）"></div>')


def main() -> None:
    rows = json.loads((OUT / "样本记录.json").read_text(encoding="utf-8"))
    openings = json.loads((OUT / "openings.json").read_text(encoding="utf-8"))
    legend = {r["variant"]: r["title"] for r in rows}
    sections = []
    for opening in openings:
        for seed in sorted({r["seed"] for r in rows}):
            cards = [r for r in rows if r["opening"] == opening["file"] and r["seed"] == seed]
            random.Random(f"{opening['file']}|{seed}").shuffle(cards)
            parts = [f'<article class="card"><div class="badge">原曲 · 10 秒</div>'
                     f'<p>蔡德允原曲 {opening["start_seconds"]} 秒起，先听原声。</p>'
                     f'<audio controls preload="none" src="audio/{opening["file"]}"></audio></article>']
            for label, row in zip("甲乙丙丁戊己", cards):
                parts.append(
                    f'<article class="card"><div class="badge">候选 {label}</div>'
                    f'<p class="who" data-who="{html.escape(legend[row["variant"]], quote=True)}">前 10 秒是原曲开头，后 25 秒为续写。</p>'
                    f'<audio controls preload="none" src="audio/{row["file"]}"></audio>'
                    f'<button class="jump" type="button">从续写处听</button>{rating(row["file"])}</article>')
            sections.append(f'<section class="case"><h2>{opening["piece"]} · 种子 {seed}</h2>'
                            f'<p class="fine">同一开头和随机种子，五个候选只差推理设置或适配器，顺序已打乱。</p>'
                            f'<div class="grid">{"".join(parts)}</div></section>')
    page = ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>古琴推理调参试听</title><style>' + STYLES + '</style></head><body>'
            '<header><div><h1>古琴推理调参试听</h1>'
            '<p>同一个 412 首适配器，比较不同推理设置，并加入旧版 222 段模型作参照。</p></div></header><main>'
            '<div class="intro">开头都来自蔡德允，她的录音没有进入任何一版训练。先听每组从第 10 秒开始的续写，'
            '<strong>重点判断擦弦噪声是否用过头、听起来像拉锯</strong>；同时留意是否仍像单人古琴、旋律是否自然。'
            '不必全部听完，评多少都有用。</div>'
            + ''.join(sections)
            + '<button class="export" id="export">下载统一评分</button>'
            '<p class="fine">评分自动保存在当前浏览器；下载 JSON 后可以直接发给我。'
            '<a href="#" id="reveal">评完后可显示候选对应关系</a></p></main>'
            '<script>' + SCRIPT + '</script></body></html>')
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(f"{len(sections)} groups, {len(rows)} clips -> {OUT / 'index.html'}")


# Styles and script follow build_sa3_overnight_page.py; the grid is 3 columns
# because each group holds the opening plus five candidates.
STYLES = '''
:root{--bg:#eeeae0;--paper:#fffdf8;--ink:#26372e;--muted:#68756b;--line:#d7ded4;--accent:#355c46}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 "Microsoft YaHei",system-ui,sans-serif}
header{background:linear-gradient(130deg,#1e3429,#456d55);color:#faf8ef;padding:43px 24px}header>div,main{max-width:1280px;margin:auto}
h1{font:normal clamp(33px,5vw,55px)/1.25 "STSong",serif;margin:0 0 10px}header p{margin:0;max-width:920px;color:#e2e8de}
main{padding:28px 24px 70px}.intro{padding:15px 18px;background:#f8f7f0;border-left:4px solid var(--accent);margin-bottom:25px}
.case{margin:0 0 34px}h2{font:normal 27px "STSong",serif;margin:0 0 3px}.fine{font-size:13px;color:var(--muted);margin:0 0 12px}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.card{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:15px;min-width:0}
.badge{font-size:17px;font-weight:700}.card p{font-size:12px;color:var(--muted);margin:3px 0 8px;min-height:22px}audio{width:100%;margin:8px 0}
button{font:inherit;cursor:pointer}.jump{background:#edf2ea;color:var(--accent);border:1px solid #cad9cc;border-radius:6px;padding:5px 10px}
.rating{border-top:1px solid var(--line);margin-top:12px;padding-top:10px;font-size:13px}select,.comment{border:1px solid var(--line);background:white;border-radius:6px;padding:4px;font:inherit}
.issues{display:flex;gap:4px;flex-wrap:wrap;margin-top:8px}.issues label{border:1px solid var(--line);border-radius:99px;padding:2px 7px;white-space:nowrap}.issues input{margin-right:3px}
.comment{width:100%;margin-top:8px}.export{background:var(--accent);color:white;border:0;border-radius:8px;padding:10px 18px}
@media(max-width:1100px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:680px){.grid{grid-template-columns:1fr}main{padding:22px 14px}header{padding:31px 16px}}
'''

SCRIPT = '''
const key='guqin_inference_sweep_v4';let state={};try{state=JSON.parse(localStorage.getItem(key)||'{}')}catch{};
const save=()=>{try{localStorage.setItem(key,JSON.stringify(state))}catch{}};
document.querySelectorAll('audio').forEach(a=>a.onplay=()=>document.querySelectorAll('audio').forEach(b=>{if(b!==a)b.pause()}));
document.querySelectorAll('.jump').forEach(b=>b.onclick=()=>{const a=b.parentElement.querySelector('audio');a.currentTime=10;a.play()});
document.querySelectorAll('.rating').forEach(r=>{const id=r.dataset.id,old=state[id]||{},score=r.querySelector('select'),checks=[...r.querySelectorAll('input[type=checkbox]')],comment=r.querySelector('.comment');
score.value=old.score||'';checks.forEach(c=>c.checked=(old.issues||[]).includes(c.value));comment.value=old.comment||'';
const update=()=>{state[id]={score:score.value,issues:checks.filter(c=>c.checked).map(c=>c.value),comment:comment.value};save()};score.onchange=update;checks.forEach(c=>c.onchange=update);comment.oninput=update});
document.querySelector('#export').onclick=()=>{const blob=new Blob([JSON.stringify({experiment:'guqin_inference_sweep_v4',scoring:'1-5 higher is better',state},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='古琴推理调参统一评分.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
document.querySelector('#reveal').onclick=e=>{e.preventDefault();document.querySelectorAll('.who').forEach(p=>p.textContent=p.dataset.who)};
'''

if __name__ == "__main__":
    main()
