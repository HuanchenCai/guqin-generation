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
:root{--bg:#f5f1e9;--card:#fff;--line:#e3d7c3;--ink:#302b23;--muted:#7a7061;--accent:#9a6a2f;--on:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#1d1b18;--card:#27241f;--line:#443d33;--ink:#ece5d8;--muted:#a89e8f;--accent:#d9a45f;--on:#1d1b18}}
*{box-sizing:border-box}
body{font-family:system-ui,"Microsoft YaHei",sans-serif;background:var(--bg);color:var(--ink);max-width:960px;margin:auto;padding:20px 16px 190px}
button{font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 12px;cursor:pointer}
button:hover{border-color:var(--accent)}
header{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
h1{font-size:22px;margin:0}.muted{color:var(--muted);font-size:14px}
.intro{margin:8px 0 18px;line-height:1.6}
.group{margin:0 0 22px}.bar{display:flex;align-items:center;gap:10px;margin-bottom:8px}
.bar h2{font-size:17px;margin:0;flex:1}.bar button{padding:3px 10px;font-size:13px}
.cards{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}
@media (max-width:700px){.cards{grid-template-columns:repeat(2,1fr)}}
.card{background:var(--card);border:2px solid var(--line);border-radius:12px;padding:8px 10px;cursor:pointer;text-align:center}
.card.current{border-color:var(--accent)}
.card .name{font-size:18px;font-weight:600}.card.playing .name::after{content:" ▶";color:var(--accent);font-size:15px}
.track{height:6px;background:var(--line);border-radius:3px;margin:8px 0;position:relative;cursor:pointer}
.track i{position:absolute;left:0;top:0;bottom:0;background:var(--accent);border-radius:3px}
.track b{position:absolute;top:-3px;bottom:-3px;width:1px;background:var(--muted);left:28.57%}
.score{font-size:15px;color:var(--muted);min-height:22px}.score.set{color:var(--accent);font-weight:600}
.flags{font-size:12px;color:var(--muted);min-height:16px}
.panel{position:fixed;left:0;right:0;bottom:0;background:var(--card);border-top:2px solid var(--accent);padding:10px 16px 12px;z-index:5;box-shadow:0 -4px 16px #0002}
.panel>div{max-width:960px;margin:auto}
.panel .row{margin-bottom:6px}
.row{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-bottom:10px}
.row .lab{width:44px;color:var(--muted);font-size:14px}
.pick.on{background:var(--accent);border-color:var(--accent);color:var(--on)}
.pick kbd{font:11px ui-monospace,Consolas,monospace;opacity:.6;margin-left:4px}
textarea{width:100%;font:inherit;font-size:14px;background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:8px;resize:vertical}
details{margin-top:18px;font-size:14px;color:var(--muted);line-height:1.9}
kbd{background:var(--card);border:1px solid var(--line);border-radius:4px;padding:0 5px;font:12px ui-monospace,Consolas,monospace}
.toast{position:fixed;bottom:18px;left:50%;transform:translateX(-50%);background:var(--ink);color:var(--bg);padding:6px 14px;border-radius:8px;font-size:14px;opacity:0;transition:opacity .2s;pointer-events:none}
.toast.show{opacity:.92}
</style>
<header><h1>古琴续写盲听</h1><div><span class="muted" id="progress"></span> <button id="export">导出评分</button></div></header>
<p class="intro muted">开头都来自没参与训练的蔡德允录音。每组 5 个候选，从第 10 秒（模型生成部分）开始播。<b>重点听擦弦噪声是否用过头、像拉锯。</b></p>
<div id="groups"></div>
<div class="panel"><div>
 <div class="row"><span class="lab" id="clabel"></span><span class="muted">总体</span><span id="scores"></span></div>
 <div class="row"><span class="lab">问题</span><span id="tags"></span></div>
 <textarea id="note" rows="1" placeholder="备注（可选）：例如“17 秒突然抽搐”"></textarea>
</div></div>
<details><summary>快捷键</summary>
<kbd>空格</kbd> 播放/暂停 · <kbd>1</kbd>–<kbd>5</kbd> 评分（评完自动下一段） · <kbd>←</kbd><kbd>→</kbd> 上/下一段 · <kbd>↑</kbd><kbd>↓</kbd> 上/下一组<br>
<kbd>S</kbd><kbd>D</kbd><kbd>M</kbd><kbd>G</kbd><kbd>C</kbd> 问题标签 · <kbd>Enter</kbd> 写备注，<kbd>Esc</kbd> 退出 · <kbd>O</kbd> 原曲开头 · <kbd>R</kbd> 重播 · <kbd>B</kbd> 从 0 秒播 · <kbd>J</kbd><kbd>L</kbd> ±5 秒<br>
<label><input type="checkbox" id="advance"> 评分后自动跳下一段</label> · <a href="#" id="reveal">显示候选对应关系</a>
</details>
<div class="toast" id="toast"></div>
<script>
const DATA=__DATA__;const KEY='guqin_inference_sweep_v1';const PREFIX=10;
// Merged from the user's own past listening notes.
const TAGS=[{k:'s',name:'拉锯/杂音'},{k:'d',name:'重复开头'},{k:'m',name:'旋律乱/半音'},{k:'g',name:'不像古琴'},{k:'c',name:'爆音/急促'}];
const LEGACY={'拉锯/金属杂音':'拉锯/杂音','没旋律/旋律乱':'旋律乱/半音','奇怪半音':'旋律乱/半音','伴音/其他乐器':'不像古琴','急促/抽搐':'爆音/急促','爆音/噼啪':'爆音/急促'};
let ratings={};try{ratings=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
for(const r of Object.values(ratings)){const s=new Set((r.issues||[]).map(x=>LEGACY[x]||x));if(r.saw)s.add('拉锯/杂音');r.issues=[...s].filter(x=>TAGS.some(t=>t.name===x))}
const $=id=>document.getElementById(id);
const advance=$('advance');try{advance.checked=localStorage.getItem(KEY+':advance')!=='0'}catch(e){advance.checked=true}
advance.onchange=()=>{try{localStorage.setItem(KEY+':advance',advance.checked?'1':'0')}catch(e){}};
const all=DATA.groups.flatMap((g,gi)=>g.cards.map((c,ci)=>({...c,gi,ci})));
let gi=0,ci=0,revealed=false,playingId=null;
const audio=new Audio();const blobs={};
let toastT;const toast=t=>{$('toast').textContent=t;$('toast').classList.add('show');clearTimeout(toastT);toastT=setTimeout(()=>$('toast').classList.remove('show'),900)};
const rec=c=>ratings[c.id]||(ratings[c.id]={score:null,issues:[],note:'',variant:c.variant});
function persist(){try{localStorage.setItem(KEY,JSON.stringify(ratings))}catch(e){}
 $('progress').textContent=`已评 ${Object.values(ratings).filter(r=>r.score).length} / ${all.length}`}
// Servers without HTTP Range support make audio unseekable; blob URLs always seek.
const ready=()=>audio.readyState>=1?Promise.resolve():new Promise(ok=>audio.addEventListener('loadedmetadata',ok,{once:true}));
let playToken=0;
async function play(file,id,t){const token=++playToken;playingId=id;audio.pause();
 const stale=()=>token!==playToken;
 // Over http, fetch once as a blob so seeking works even without Range support.
 // file:// pages cannot fetch, but the browser seeks local files natively.
 if(!blobs[file]&&t>0&&location.protocol!=='file:'){
  try{const blob=await (await fetch('audio/'+file)).blob();blobs[file]=blobs[file]||URL.createObjectURL(blob)}catch(e){}
  if(stale())return}
 const url=blobs[file]||('audio/'+file);
 if(audio.src!==url&&!audio.src.endsWith('/'+url)){audio.src=url;audio.load()}
 await ready();if(stale())return;
 audio.currentTime=Math.min(t,audio.duration-0.1);await audio.play().catch(()=>{});render()}
const cur=()=>DATA.groups[gi].cards[ci];
const playCard=(t=PREFIX)=>play(cur().file,cur().id,t);
function render(){const box=$('groups');box.innerHTML='';
 DATA.groups.forEach((g,gidx)=>{const sec=document.createElement('section');sec.className='group';
  sec.innerHTML=`<div class="bar"><h2>${gidx+1}. ${g.opening.piece} · 种子 ${g.seed}</h2><button>▶ 原曲开头</button></div><div class="cards"></div>`;
  sec.querySelector('button').onclick=()=>play(g.opening.file,'orig',0);
  const row=sec.querySelector('.cards');
  g.cards.forEach((c,i)=>{const r=ratings[c.id]||{};const d=document.createElement('div');
   const here=gidx===gi&&i===ci;
   d.className='card'+(here?' current':'')+(playingId===c.id&&!audio.paused?' playing':'');d.dataset.id=c.id;
   d.innerHTML=`<div class="name">${c.label}</div><div class="track"><b></b><i></i></div><div class="score ${r.score?'set':''}">${r.score?r.score+' 分':'未评'}</div><div class="flags">${(r.issues||[]).join(' · ')}</div>${revealed?`<div class="flags">${DATA.legend[c.variant]}</div>`:''}`;
   d.onclick=ev=>{if(ev.target.closest('.track'))return;const same=here;gi=gidx;ci=i;if(same&&!audio.paused&&playingId===c.id)audio.pause();else playCard();render()};
   d.querySelector('.track').onclick=ev=>{const f=(ev.clientX-ev.currentTarget.getBoundingClientRect().left)/ev.currentTarget.clientWidth;gi=gidx;ci=i;play(c.file,c.id,f*35)};
   row.append(d)});
  box.append(sec)});
 const c=cur(),r=ratings[c.id]||{};$('clabel').textContent=`${gi+1}·${c.label}`;
 $('scores').innerHTML=[1,2,3,4,5].map(n=>`<button class="pick ${r.score===n?'on':''}" data-n="${n}">${n}</button>`).join(' ');
 $('scores').querySelectorAll('button').forEach(b=>b.onclick=()=>score(+b.dataset.n));
 $('tags').innerHTML=TAGS.map(t=>`<button class="pick ${(r.issues||[]).includes(t.name)?'on':''}" data-k="${t.k}">${t.name}<kbd>${t.k.toUpperCase()}</kbd></button>`).join(' ');
 $('tags').querySelectorAll('button').forEach(b=>b.onclick=()=>tag(b.dataset.k));
 if(document.activeElement!==$('note'))$('note').value=r.note||'';
 tick()}
function reveal(){const el=document.querySelector('.card.current');if(!el)return;const b=el.getBoundingClientRect(),panel=document.querySelector('.panel').offsetHeight;
 if(b.top<60||b.bottom>innerHeight-panel-10)el.scrollIntoView({block:'center',behavior:'smooth'})}
function tick(){document.querySelectorAll('.card').forEach(el=>{const bar=el.querySelector('.track i');
 bar.style.width=(playingId===el.dataset.id&&audio.duration?audio.currentTime/audio.duration*100:0)+'%'})}
audio.ontimeupdate=tick;audio.onplay=audio.onpause=audio.onended=()=>render();
function go(g,c=0){gi=(g+DATA.groups.length)%DATA.groups.length;ci=c;playCard();render();reveal()}
function step(d){let i=all.findIndex(x=>x.gi===gi&&x.ci===ci)+d;i=(i+all.length)%all.length;go(all[i].gi,all[i].ci)}
function nextUnrated(){const s=all.findIndex(x=>x.gi===gi&&x.ci===ci);for(let k=1;k<=all.length;k++){const x=all[(s+k)%all.length];if(!(ratings[x.id]||{}).score)return x}return null}
function score(n){rec(cur()).score=n;persist();render();toast(`${cur().label}：${n} 分`);
 if(advance.checked){const x=nextUnrated();setTimeout(()=>x?go(x.gi,x.ci):toast('全部评完，记得导出'),300)}}
function tag(k){const t=TAGS.find(t=>t.k===k),r=rec(cur()),i=r.issues.indexOf(t.name);
 if(i>=0)r.issues.splice(i,1);else r.issues.push(t.name);persist();render();toast((i>=0?'取消：':'标记：')+t.name)}
$('note').oninput=()=>{rec(cur()).note=$('note').value.trim();persist()};
$('note').onblur=()=>render();
$('reveal').onclick=e=>{e.preventDefault();revealed=!revealed;render()};
document.addEventListener('keydown',ev=>{if(ev.ctrlKey||ev.metaKey||ev.altKey)return;
 if(ev.target.tagName==='TEXTAREA'){if(ev.key==='Escape'){ev.target.blur();ev.preventDefault()}return}
 const k=ev.key.toLowerCase();
 if(k===' '){ev.preventDefault();if(playingId===cur().id&&!audio.paused)audio.pause();else if(playingId===cur().id&&audio.currentTime>0&&!audio.ended)audio.play();else playCard()}
 else if(/^[1-5]$/.test(k))score(+k);
 else if(TAGS.some(t=>t.k===k))tag(k);
 else if(k==='enter'){ev.preventDefault();$('note').focus()}
 else if(k==='arrowright'||k==='n'){ev.preventDefault();step(1)}
 else if(k==='arrowleft'||k==='p'){ev.preventDefault();step(-1)}
 else if(k==='arrowdown'){ev.preventDefault();go(gi+1)}
 else if(k==='arrowup'){ev.preventDefault();go(gi-1)}
 else if(k==='o')play(DATA.groups[gi].opening.file,'orig',0);
 else if(k==='r')playCard(PREFIX);
 else if(k==='b')playCard(0);
 else if(k==='j'||k==='l'){if(audio.src)audio.currentTime=Math.max(0,audio.currentTime+(k==='j'?-5:5))}
});
$('export').onclick=()=>{const blob=new Blob([JSON.stringify({schema:'guqin-inference-sweep-v3',ratings,generated_at:new Date().toISOString()},null,2)],{type:'application/json'});
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='古琴推理调参评分.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
persist();render();
</script></html>'''

if __name__ == "__main__":
    main()
