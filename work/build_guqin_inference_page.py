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
:root{--bg:#f5f1e9;--card:#fff;--soft:#faf8f3;--line:#e3d7c3;--ink:#302b23;--muted:#6b6255;--accent:#9a6a2f;--focus:#c7842f;--kbd:#ece4d6}
@media (prefers-color-scheme:dark){:root{--bg:#1d1b18;--card:#27241f;--soft:#2e2a24;--line:#443d33;--ink:#ece5d8;--muted:#a89e8f;--accent:#d9a45f;--focus:#e8a652;--kbd:#3a342c}}
body{font-family:system-ui,"Microsoft YaHei",sans-serif;background:var(--bg);color:var(--ink);max-width:1180px;margin:auto;padding:24px 16px 80px}
h1{font-size:26px;margin:0 0 6px}p{line-height:1.7}
.note{background:var(--card);border-left:4px solid var(--accent);padding:10px 16px;margin:16px 0}
.toolbar{position:sticky;top:0;background:var(--bg);padding:8px 0;z-index:2;display:flex;gap:10px;align-items:center;flex-wrap:wrap;border-bottom:1px solid var(--line)}
.keys{font-size:13px;color:var(--muted);line-height:2}
kbd{background:var(--kbd);border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 5px;font:12px ui-monospace,Consolas,monospace;color:var(--ink)}
.set{background:var(--card);border:1px solid var(--line);border-radius:14px;margin:16px 0;padding:16px}
.set h2{margin:0 0 4px;font-size:19px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin-top:10px}
.clip{background:var(--soft);border:2px solid var(--line);border-radius:10px;padding:10px;cursor:pointer}
.clip.done{border-color:var(--accent)}.clip.current{border-color:var(--focus);box-shadow:0 0 0 3px color-mix(in srgb,var(--focus) 35%,transparent)}
.clip.playing strong::after{content:" ▶";color:var(--focus)}
audio{width:100%;margin-top:6px}
label{display:block;margin:6px 0;font-size:14px}select,button{font:inherit;padding:6px 9px}
.tag{font-size:12px;color:var(--muted)}small{color:var(--muted)}
.chips{display:flex;flex-wrap:wrap;gap:4px;margin:6px 0}.chip{font-size:12px;padding:2px 6px;border:1px solid var(--line);border-radius:999px;background:var(--card);color:var(--ink);cursor:pointer}
.chip kbd{margin-left:3px;font-size:10px;padding:0 3px}.chip.on{background:var(--accent);border-color:var(--accent);color:var(--bg)}.chip.on kbd{background:transparent;color:var(--bg);border-color:var(--bg)}
.memo{width:100%;box-sizing:border-box;font:13px/1.5 inherit;font-family:inherit;background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:5px;resize:vertical}
.opening{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.opening audio{flex:1;min-width:220px}
.toast{position:fixed;bottom:18px;left:50%;transform:translateX(-50%);background:var(--ink);color:var(--bg);padding:6px 14px;border-radius:8px;font-size:14px;opacity:0;transition:opacity .2s;pointer-events:none}
.toast.show{opacity:.92}
</style>
<h1>古琴续写 · 推理调参盲听</h1>
<p>所有开头都来自<strong>蔡德允</strong>，她的录音没有进入任何一版训练，所以这里全部是未见录音。每组固定同一个 10 秒开头和随机种子，模型接 25 秒；五个候选只差推理设置或适配器，顺序已打乱。</p>
<div class="note">重点听：<b>擦弦/走手的短促噪声</b>是不是用过头、听起来像拉锯。有这个问题请点“拉锯/金属杂音”（快捷键 S），这是我接下来训练自动筛选器的标签。其他问题标签取自你以前试听时最常写的备注；标签说不清的，直接写进备注框，比如在第几秒出了什么问题。不必全部听完，评多少都有用。</div>
<div class="toolbar">
 <button id="export">导出评分 JSON</button><button id="reveal">显示候选对应关系</button>
 <label style="margin:0"><input type="checkbox" id="skip" checked> 候选从第 10 秒开始</label>
 <label style="margin:0"><input type="checkbox" id="advance" checked> 评分后自动下一段</label>
 <span id="progress"></span>
 <div class="keys" style="flex-basis:100%">
  <kbd>空格</kbd> 播放/暂停 · <kbd>1</kbd>–<kbd>5</kbd> 评分 · 问题标签见每段下方的按键 · <kbd>Enter</kbd> 写备注，<kbd>Esc</kbd> 退出 ·
  <kbd>→</kbd>/<kbd>N</kbd> 下一段 · <kbd>←</kbd>/<kbd>P</kbd> 上一段 · <kbd>U</kbd> 下一段未评分 ·
  <kbd>R</kbd> 从第 10 秒重播 · <kbd>B</kbd> 从头播（含开头） · <kbd>O</kbd> 播原曲开头 ·
  <kbd>J</kbd>/<kbd>L</kbd> 后退/前进 5 秒 · <kbd>↓</kbd>/<kbd>↑</kbd> 下一组/上一组
 </div>
</div>
<div id="mount"></div><div class="toast" id="toast"></div>
<script>
const DATA=__DATA__;
// Issue tags come from the user's own past listening notes.
const ISSUES=[{key:'s',name:'拉锯/金属杂音',hint:'擦弦、走手噪声用过头，像锯子、弹棉花、咔咔声'},
 {key:'d',name:'重复开头',hint:'反复重复前 10 秒，像复制粘贴'},
 {key:'m',name:'没旋律/旋律乱',hint:'没有旋律、旋律混乱或草率'},
 {key:'h',name:'奇怪半音',hint:'像按错弦、练琴没弹好'},
 {key:'g',name:'不像古琴',hint:'像吉他扫弦、竖琴、西域弹拨乐、唱歌'},
 {key:'a',name:'伴音/其他乐器',hint:'奇怪的伴音、低音或其他乐器'},
 {key:'k',name:'急促/抽搐',hint:'音符连得过快、突然抽搐、突兀'},
 {key:'c',name:'爆音/噼啪',hint:'爆音、失真、噼啪声'},
 {key:'v',name:'太轻/太响',hint:'音量突变、动态过大或过小'},
 {key:'f',name:'过闷',hint:'声音发闷、高频被磨掉'},
 {key:'x',name:'其他',hint:'其他问题，请写备注'}];
const SAW='拉锯/金属杂音';const KEY='guqin_inference_sweep_v1';const PREFIX=10;
let ratings={};try{ratings=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
const pref=(k,d)=>{try{const v=localStorage.getItem(KEY+':'+k);return v===null?d:v==='1'}catch(e){return d}};
const setPref=(k,v)=>{try{localStorage.setItem(KEY+':'+k,v?'1':'0')}catch(e){}};
const skip=document.getElementById('skip'),advance=document.getElementById('advance');
skip.checked=pref('skip',true);advance.checked=pref('advance',true);
skip.onchange=()=>setPref('skip',skip.checked);advance.onchange=()=>setPref('advance',advance.checked);
let revealed=false,cur=-1;const cards=[];const openings=[];
const toastEl=document.getElementById('toast');let toastTimer;
const toast=t=>{toastEl.textContent=t;toastEl.classList.add('show');clearTimeout(toastTimer);toastTimer=setTimeout(()=>toastEl.classList.remove('show'),900)};
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(ratings))}catch(e){}
 const n=Object.values(ratings).filter(x=>x.score).length;
 document.getElementById('progress').textContent=`已评分 ${n}/${cards.length} 段`};
const allAudio=()=>document.querySelectorAll('audio');
const pauseOthers=a=>allAudio().forEach(x=>{if(x!==a)x.pause()});
const canSeek=(a,t)=>{for(let i=0;i<a.seekable.length;i++)if(a.seekable.start(i)<=t&&a.seekable.end(i)>=t)return true;return false};
const metadata=a=>a.readyState>=1?Promise.resolve():new Promise(ok=>{a.addEventListener('loadedmetadata',ok,{once:true});a.preload='auto';a.load()});
// Servers without HTTP Range support make audio unseekable; a blob URL is always seekable.
async function ensureSeekable(a,t){await metadata(a);if(t<=0||canSeek(a,t)||a.src.startsWith('blob:'))return;
 const blob=await (await fetch(a.src)).blob();a.src=URL.createObjectURL(blob);await metadata(a)}
async function playFrom(a,t){pauseOthers(a);await ensureSeekable(a,t);a.currentTime=Math.min(t,Math.max(0,a.duration-0.1));pauseOthers(a);a.play()}
function select(i,{play=true,scroll=true}={}){if(i<0||i>=cards.length)return;
 if(cur>=0)cards[cur].el.classList.remove('current');cur=i;const c=cards[i];c.el.classList.add('current');
 if(scroll)c.el.scrollIntoView({block:'center',behavior:'smooth'});
 if(play)playFrom(c.audio,skip.checked?PREFIX:0)}
function nextUnrated(from){for(let k=1;k<=cards.length;k++){const j=(from+k)%cards.length;if(!(ratings[cards[j].id]||{}).score)return j}return -1}
function update(c){const issues=ISSUES.filter(i=>c.chips[i.key].classList.contains('on')).map(i=>i.name);
 const e={score:Number(c.sel.value)||null,issues,note:c.note.value.trim(),saw:issues.includes(SAW),variant:c.variant};
 ratings[c.id]=e;c.el.classList.toggle('done',!!e.score);save()}
function toggleIssue(c,key){const chip=c.chips[key];chip.classList.toggle('on');update(c);
 toast((chip.classList.contains('on')?'已标记：':'取消：')+ISSUES.find(i=>i.key===key).name)}
const mount=document.getElementById('mount');
DATA.groups.forEach((g,gi)=>{const box=document.createElement('section');box.className='set';
 box.innerHTML=`<h2>${g.opening.piece} · 种子 ${g.seed}</h2><small>蔡德允原曲 ${g.opening.start_seconds} 秒起的 10 秒开头</small><div class="opening"><span>原曲开头</span><audio controls preload="none" src="audio/${g.opening.file}"></audio></div><div class="grid"></div>`;
 openings.push(box.querySelector('.opening audio'));const grid=box.querySelector('.grid');
 for(const c of g.cards){const e=ratings[c.id]||{};const d=document.createElement('div');d.className='clip'+(e.score?' done':'');
  d.innerHTML=`<strong>候选 ${c.label}</strong> <span class="tag" data-v="${c.variant}"></span><audio controls preload="none" src="audio/${c.file}"></audio>
  <label>总体 <select><option value="">待评分</option>${[1,2,3,4,5].map(n=>`<option value="${n}">${n} / 5</option>`).join('')}</select></label>
  <div class="chips">${ISSUES.map(i=>`<button type="button" class="chip" data-k="${i.key}" title="${i.hint}">${i.name}<kbd>${i.key.toUpperCase()}</kbd></button>`).join('')}</div>
  <textarea class="memo" rows="2" placeholder="备注：比如“17 秒突然抽搐”“延长音有咔咔声”（按 Enter 开始输入，Esc 退出）"></textarea>`;
  const item={...c,group:gi,el:d,audio:d.querySelector('audio'),sel:d.querySelector('select'),note:d.querySelector('.memo'),chips:{}};
  d.querySelectorAll('.chip').forEach(b=>{item.chips[b.dataset.k]=b;b.onclick=()=>{select(idx,{play:false,scroll:false});toggleIssue(item,b.dataset.k)}});
  const idx=cards.length;cards.push(item);
  item.sel.value=e.score||'';item.note.value=e.note||'';
  const had=new Set(e.issues||[]);if(e.saw)had.add(SAW);
  ISSUES.forEach(i=>item.chips[i.key].classList.toggle('on',had.has(i.name)));
  item.sel.onchange=()=>update(item);item.note.oninput=()=>update(item);
  item.note.onfocus=()=>{if(cur!==idx)select(idx,{play:false,scroll:false})};
  d.addEventListener('click',ev=>{if(ev.target.closest('audio,select,input,label,textarea,button'))return;select(idx,{scroll:false})});
  // Native play button: jump past the shared prefix when starting from the top.
  item.audio.addEventListener('play',()=>{pauseOthers(item.audio);if(cur!==idx)select(idx,{play:false,scroll:false});
   if(skip.checked&&item.audio.currentTime<0.5){item.audio.pause();playFrom(item.audio,PREFIX)}});
  item.audio.addEventListener('playing',()=>d.classList.add('playing'));
  ['pause','ended'].forEach(t=>item.audio.addEventListener(t,()=>d.classList.remove('playing')));
  grid.append(d)}
 mount.append(box)});
openings.forEach(a=>a.addEventListener('play',()=>pauseOthers(a)));
document.addEventListener('keydown',ev=>{
 if(ev.ctrlKey||ev.metaKey||ev.altKey)return;
 const tag=(ev.target.tagName||'').toLowerCase();if(tag==='textarea'){if(ev.key==='Escape'){ev.target.blur();ev.preventDefault()}return}
 if(tag==='select'||(tag==='input'&&ev.target.type!=='checkbox'))return;
 const k=ev.key.toLowerCase();const c=cur>=0?cards[cur]:null;
 const need=()=>{if(!c){select(0,{play:false});return true}return false};
 if(k===' '){ev.preventDefault();if(need())return;const a=c.audio;if(a.paused){if(a.currentTime<0.5||a.ended)playFrom(a,skip.checked?PREFIX:0);else{pauseOthers(a);a.play()}}else a.pause()}
 else if(/^[1-5]$/.test(k)){if(need())return;c.sel.value=k;update(c);toast(`候选 ${c.label}：${k} 分`);
  if(advance.checked){const j=nextUnrated(cur);setTimeout(()=>{if(j>=0)select(j);else{c.audio.pause();toast('全部评完了，记得导出')}},250)}}
 else if(ISSUES.some(i=>i.key===k)){if(need())return;toggleIssue(c,k)}
 else if(k==='enter'){ev.preventDefault();if(need())return;c.note.focus()}
 else if(k==='arrowright'||k==='n'){ev.preventDefault();select(cur+1)}
 else if(k==='arrowleft'||k==='p'){ev.preventDefault();select(Math.max(0,cur-1))}
 else if(k==='u'){const j=nextUnrated(cur);if(j>=0)select(j);else toast('没有未评分的了')}
 else if(k==='arrowdown'||k==='arrowup'){ev.preventDefault();const g=(c?c.group:-1)+(k==='arrowdown'?1:-1);const j=cards.findIndex(x=>x.group===g);if(j>=0)select(j)}
 else if(k==='r'){if(need())return;playFrom(c.audio,PREFIX)}
 else if(k==='b'){if(need())return;playFrom(c.audio,0)}
 else if(k==='o'){if(need())return;playFrom(openings[c.group],0)}
 else if(k==='j'||k==='l'){if(need())return;const a=c.audio;a.currentTime=Math.max(0,a.currentTime+(k==='j'?-5:5))}
});
document.getElementById('export').onclick=()=>{const blob=new Blob([JSON.stringify({schema:'guqin-inference-sweep-v2',ratings,generated_at:new Date().toISOString()},null,2)],{type:'application/json'});
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='古琴推理调参评分.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
document.getElementById('reveal').onclick=()=>{revealed=!revealed;document.querySelectorAll('.tag').forEach(t=>t.textContent=revealed?DATA.legend[t.dataset.v]:'')};
save();
</script></html>'''

if __name__ == "__main__":
    main()
