"""Blind listening page for the guqin inference sweep."""

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "outputs" / "古琴推理调参"


def main() -> None:
    rows = json.loads((OUT / "样本记录.json").read_text(encoding="utf-8"))
    openings = json.loads((OUT / "openings.json").read_text(encoding="utf-8"))
    groups = []
    for opening in openings:
        for seed in sorted({r["seed"] for r in rows}):
            cards = [{"id": r["file"], "file": r["file"], "variant": r["variant"]}
                     for r in rows if r["opening"] == opening["file"] and r["seed"] == seed]
            random.Random(f"{opening['file']}|{seed}").shuffle(cards)
            for label, card in zip("甲乙丙丁戊己", cards):
                card["label"] = label
            groups.append({"opening": opening, "seed": seed, "cards": cards})
    legend = {r["variant"]: r["title"] for r in rows}
    payload = json.dumps({"groups": groups, "legend": legend},
                         ensure_ascii=False).replace("</", "<\\/")
    html = PAGE.replace("__DATA__", payload)
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(f"{len(groups)} groups, {len(rows)} clips -> {OUT / 'index.html'}")


PAGE = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>古琴推理调参</title>
<style>
:root{--bg:#f5f1e9;--card:#fff;--soft:#faf8f3;--line:#e3d7c3;--ink:#302b23;--muted:#6b6255;--accent:#9a6a2f}
@media (prefers-color-scheme:dark){:root{--bg:#1d1b18;--card:#27241f;--soft:#2e2a24;--line:#443d33;--ink:#ece5d8;--muted:#a89e8f;--accent:#d9a45f}}
body{font-family:system-ui,"Microsoft YaHei",sans-serif;background:var(--bg);color:var(--ink);max-width:1180px;margin:auto;padding:24px 16px}
h1{font-size:26px;margin:0 0 6px}p{line-height:1.7}
.note{background:var(--card);border-left:4px solid var(--accent);padding:10px 16px;margin:16px 0}
.toolbar{position:sticky;top:0;background:var(--bg);padding:8px 0;z-index:2;display:flex;gap:10px;align-items:center;flex-wrap:wrap;border-bottom:1px solid var(--line)}
.set{background:var(--card);border:1px solid var(--line);border-radius:14px;margin:16px 0;padding:16px}
.set h2{margin:0 0 4px;font-size:19px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin-top:10px}
.clip{background:var(--soft);border:1px solid var(--line);border-radius:10px;padding:10px}
.clip.done{border-color:var(--accent)}audio{width:100%;margin-top:6px}
label{display:block;margin:6px 0;font-size:14px}select,button{font:inherit;padding:6px 9px}
.tag{font-size:12px;color:var(--muted)}small{color:var(--muted)}
</style>
<h1>古琴续写 · 推理调参盲听</h1>
<p>所有开头都来自<strong>蔡德允</strong>，她的录音没有进入任何一版训练，所以这里全部是未见录音。每组固定同一个 10 秒开头和随机种子，模型接 25 秒；五个候选只差推理设置或适配器，顺序已打乱。</p>
<div class="note">重点听：<b>擦弦/走手的短促噪声</b>是不是用过头、听起来像拉锯。有这个问题请勾“拉锯感”，这是我接下来训练自动筛选器的标签。其他问题（重复、机械音、跑调、音量突变）勾“其他故障”。不必全部听完，评多少都有用。</div>
<div class="toolbar"><button id="export">导出评分 JSON</button><button id="reveal">显示候选对应关系</button><span id="progress"></span></div>
<div id="mount"></div>
<script>
const DATA=__DATA__;const KEY='guqin_inference_sweep_v1';let ratings={};
try{ratings=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
let revealed=false;
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(ratings))}catch(e){}
 const n=Object.values(ratings).filter(x=>x.score).length;const total=DATA.groups.reduce((a,g)=>a+g.cards.length,0);
 document.getElementById('progress').textContent=`已评分 ${n}/${total} 段`};
const mount=document.getElementById('mount');
for(const g of DATA.groups){const box=document.createElement('section');box.className='set';
 box.innerHTML=`<h2>${g.opening.piece} · 种子 ${g.seed}</h2><small>蔡德允原曲 ${g.opening.start_seconds} 秒起的 10 秒开头</small><label>原曲开头 <audio controls preload="none" src="audio/${g.opening.file}"></audio></label><div class="grid"></div>`;
 const grid=box.querySelector('.grid');
 for(const c of g.cards){const e=ratings[c.id]||{};const d=document.createElement('div');d.className='clip'+(e.score?' done':'');
  d.innerHTML=`<strong>候选 ${c.label}</strong> <span class="tag" data-v="${c.variant}"></span><audio controls preload="none" src="audio/${c.file}"></audio>
  <label>总体 <select><option value="">待评分</option>${[1,2,3,4,5].map(n=>`<option value="${n}">${n} / 5</option>`).join('')}</select></label>
  <label><input type="checkbox" class="saw"> 拉锯感（擦弦噪声过多）</label><label><input type="checkbox" class="other"> 其他故障</label>`;
  const sel=d.querySelector('select'),saw=d.querySelector('.saw'),other=d.querySelector('.other');
  sel.value=e.score||'';saw.checked=!!e.saw;other.checked=!!e.other;
  const up=()=>{ratings[c.id]={score:Number(sel.value)||null,saw:saw.checked,other:other.checked,variant:c.variant};d.classList.toggle('done',!!sel.value);save()};
  sel.onchange=up;saw.onchange=up;other.onchange=up;grid.append(d)}
 mount.append(box)}
document.getElementById('export').onclick=()=>{const blob=new Blob([JSON.stringify({schema:'guqin-inference-sweep-v1',ratings,generated_at:new Date().toISOString()},null,2)],{type:'application/json'});
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='古琴推理调参评分.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
document.getElementById('reveal').onclick=()=>{revealed=!revealed;document.querySelectorAll('.tag').forEach(t=>t.textContent=revealed?DATA.legend[t.dataset.v]:'')};
save();
</script></html>'''

if __name__ == "__main__":
    main()
