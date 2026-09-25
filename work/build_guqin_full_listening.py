"""Build a full-recording, single-player review page for guqin candidates."""

import csv
import html
import json
from pathlib import Path


WORK = Path(__file__).resolve().parent
ROOT = Path(r"Y:\Music\古琴曲")
SOURCE = WORK.parent / "outputs" / "古琴录音筛选"
OUT = WORK.parent / "outputs" / "古琴整曲筛选"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def scanned() -> dict[str, dict]:
    result = {}
    for path in (WORK / "sa3_feedback_bulk" / "whole_track_scan.jsonl",
                 WORK / "guqin_full_scan.jsonl"):
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                result[row["relative_path"]] = row
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = [row for row in read_csv(SOURCE / "逐首筛选清单_听感更新.csv")
               if row["筛选状态"].startswith("候选")]
    reviewed_source = {row["Y盘相对路径"] for name in ("同来源目录候选.csv", "跨目录候选.csv")
                       for row in read_csv(SOURCE / name)}
    same_folder = {row["Y盘相对路径"] for row in read_csv(SOURCE / "同来源目录候选.csv")}
    analysis = scanned()
    tracks = []
    for row in catalog:
        rel = row["Y盘相对路径"]
        info = analysis.get(rel, {})
        rms = info.get("one_second_rms", [])
        quiet_positions = [i for i, level in enumerate(rms) if 0.0018 < level < 0.03]
        quiet = min(quiet_positions, key=lambda i: rms[i]) if quiet_positions else None
        tracks.append({
            "id": rel,
            "artist": row["演奏者"],
            "title": row["曲名"],
            "album": rel.split("\\", 1)[0],
            "uri": (ROOT / Path(rel.replace("\\", "/"))).as_uri(),
            "seconds": round(float(row["时长分钟"]) * 60),
            "priority": "已认可来源·同目录" if rel in same_folder else
                        "已认可来源·其他目录" if rel in reviewed_source else "其他候选",
            "prior": row["首轮本曲判断"],
            "priorIssues": row["首轮问题标记"],
            "priorNote": row["首轮听感备注"],
            "technical": row["首轮层级"],
            "clipSeconds": info.get("seconds_with_clip_over_0_1pct"),
            "quietSeconds": info.get("seconds_under_minus55db"),
            "rms": info.get("rms_dbfs"),
            "quietSeek": quiet,
            "readError": info.get("error", ""),
        })
    tracks.sort(key=lambda x: (0 if x["priority"] == "已认可来源·同目录" else
                               1 if x["priority"] == "已认可来源·其他目录" else 2,
                               x["artist"], x["title"], x["id"]))
    data = json.dumps(tracks, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>古琴整曲筛选</title><style>
:root{--ink:#26372d;--green:#2f6549;--line:#d7dfd4;--muted:#677569;--paper:#fffdf8;--bg:#eff1eb}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 "Microsoft YaHei",system-ui,sans-serif}
header{padding:23px 26px;background:#23412f;color:#f8fcf7}header h1{font:normal 35px/1.2 "STSong",serif;margin:0 0 8px}header p{margin:0;color:#dbe8dd;max-width:1020px}
main{display:grid;grid-template-columns:minmax(310px,43%) minmax(350px,1fr);height:calc(100vh - 121px);min-height:600px}
.queue{border-right:1px solid var(--line);display:flex;flex-direction:column;min-height:0}.toolbar{background:var(--paper);padding:12px 15px;border-bottom:1px solid var(--line);display:grid;gap:8px}
.controls{display:flex;gap:6px;flex-wrap:wrap}input[type=search],select,textarea{font:inherit;border:1px solid #bdcbbd;border-radius:6px;background:white;padding:7px 9px}input[type=search]{width:100%}select{max-width:100%}
button{font:inherit;border:1px solid #b7c9ba;border-radius:7px;background:#fff;color:var(--ink);padding:7px 10px;cursor:pointer}button:hover,.row:hover{background:#e7eee7}button.primary{background:var(--green);color:white;border-color:var(--green)}button:focus-visible,.row:focus-visible{outline:3px solid #8eb99a}
.small{font-size:12px;color:var(--muted)}.count{font-size:13px;font-weight:700}.list{overflow:auto;min-height:0}.row{padding:10px 15px;border-bottom:1px solid var(--line);cursor:pointer;display:grid;grid-template-columns:42px 1fr auto;gap:8px;align-items:start;background:var(--paper)}.row.active{background:#dcebdc;border-left:4px solid var(--green);padding-left:11px}.row .num{color:var(--muted)}.row strong{display:block;font-size:15px}.row .meta{color:var(--muted);font-size:12px}.pill{font-size:12px;padding:2px 6px;border-radius:6px;background:#edf1e8;white-space:nowrap}.pill.good{background:#dcebdc}.pill.bad{background:#f4dddd}.pill.maybe{background:#f5edcf}
.detail{overflow:auto;padding:22px 25px}.panel{max-width:760px;margin:auto}.eyebrow{font-size:13px;color:var(--green);font-weight:700}.detail h2{font:normal 34px/1.25 "STSong",serif;margin:7px 0}.detail h3{font-size:18px;margin:24px 0 8px}.facts{color:var(--muted);margin:0 0 15px}.player{position:sticky;top:0;z-index:2;background:var(--paper);padding:16px;border:1px solid var(--line);border-radius:10px;box-shadow:0 6px 18px #15352016}.player audio{width:100%}.seek,.judge,.nav{display:flex;gap:7px;flex-wrap:wrap;margin-top:10px}.judge button{font-weight:700}.judge .keep{background:#deefdf}.judge .reject{background:#f6e1dd}.hint{font-size:13px;color:var(--muted);margin:10px 0}.prior,.note{padding:11px 13px;border:1px solid var(--line);border-radius:7px;background:var(--paper);margin:14px 0}.note textarea{width:100%;min-height:68px}.warn{color:#963e28}.shortcut{margin:20px 0;font-size:13px;color:var(--muted)}.shortcut kbd{background:white;border:1px solid var(--line);border-radius:4px;padding:1px 5px}
@media(max-width:780px){header{padding:18px}main{display:block;height:auto}.queue{height:43vh;border-right:0;border-bottom:1px solid var(--line)}.detail{overflow:visible;padding:17px}.player{top:0}.detail h2{font-size:27px}}
</style></head><body><header><h1>古琴整曲筛选</h1><p>一次列出全部 571 首单人古琴候选，播放原始整首录音。先前 16 秒的判断只作参考；这里单独记录整曲是否可用。前 400 首来自已认可的演奏者，随后是其他候选。</p></header>
<main><section class="queue"><div class="toolbar"><input id="search" type="search" placeholder="搜演奏者、曲名或册数"><div class="controls"><select id="group"><option value="all">全部 571 首</option><option value="approved">已认可来源 400 首</option><option value="other">其他候选 171 首</option></select><select id="filter"><option value="all">全部判断</option><option value="unrated">待整曲判断</option><option value="keep">整首可用</option><option value="partial">部分可用</option><option value="mood">有氛围</option><option value="reject">不适合</option></select></div><div class="controls"><button id="nextUnrated">下一首待判断 N</button><button id="export" class="primary">导出筛选结果</button><button id="import">导入结果</button><input id="importFile" type="file" accept="application/json,.json" hidden></div><div class="count" id="count"></div><div class="small">判断自动保存在这个浏览器；导出 JSON 可发给我。支持连续播放和键盘快捷键。</div></div><div id="list" class="list"></div></section>
<section class="detail"><div class="panel"><div class="eyebrow" id="place"></div><h2 id="title">选择一首录音</h2><p class="facts" id="facts"></p><div class="player"><audio id="audio" controls preload="none"></audio><div class="seek"><button data-seek="0">开头</button><button data-seek=".25">1/4</button><button data-seek=".5">一半</button><button data-seek=".75">3/4</button><button id="quiet">听较安静处</button></div><div class="nav"><button id="prev">上一首</button><button id="next">下一首</button><label><input id="auto" type="checkbox" checked> 标记后自动听下一首</label></div><div id="playError" class="hint warn"></div></div><div class="prior" id="prior"></div><h3>整曲判断</h3><div class="judge"><button data-decision="keep" class="keep">1 整首可用</button><button data-decision="mood">2 有氛围</button><button data-decision="partial">3 部分可用</button><button data-decision="reject" class="reject">4 不适合</button><button data-decision="">撤销</button></div><div class="note"><label for="ranges">若选择“部分可用”，写下可用时间段，每行一段</label><textarea id="ranges" placeholder="例如：0:30-3:20&#10;4:05-6:10"></textarea><label for="comment">其他备注</label><textarea id="comment" placeholder="例如：后半段底噪明显"></textarea></div><p class="hint" id="signal"></p><p class="shortcut">快捷键：<kbd>空格</kbd>播放/暂停，<kbd>J</kbd>/<kbd>L</kbd>退/进 30 秒，<kbd>N</kbd>下一首待判断，<kbd>1</kbd>–<kbd>4</kbd>标记。输入备注时快捷键暂停。</p><p class="hint">音频直接读取 Y: 盘原文件，不做降噪或一分钟截取。训练前会依据整曲判断，分段切出全部可用部分。</p></div></section></main>
<script id="tracks" type="application/json">TRACK_DATA</script><script>
const tracks=JSON.parse(document.querySelector('#tracks').textContent), byId=new Map(tracks.map((t,i)=>[t.id,i]));
const key='guqin_full_recordings_v1'; let state={};try{state=JSON.parse(localStorage.getItem(key)||'{}')}catch{};
let current=-1, visible=[]; const $=s=>document.querySelector(s), audio=$('#audio');
const labels={keep:'整首可用',mood:'有氛围',partial:'部分可用',reject:'不适合'};
const group=t=>t.priority==='其他候选'?'other':'approved';
function save(){localStorage.setItem(key,JSON.stringify(state))}
function filtered(){const q=$('#search').value.trim().toLowerCase(),g=$('#group').value,f=$('#filter').value;return tracks.map((t,i)=>i).filter(i=>{const t=tracks[i],d=state[t.id]?.decision||'';return(g==='all'||group(t)===g)&&(f==='all'||(f==='unrated'?!d:d===f))&&(!q||(t.artist+' '+t.title+' '+t.id).toLowerCase().includes(q))})}
function render(){visible=filtered();const total=Object.values(state).filter(x=>x?.decision).length;$('#count').textContent=`已判 ${total} / ${tracks.length} · 当前显示 ${visible.length}`;$('#list').innerHTML=visible.map((i,n)=>{const t=tracks[i],d=state[t.id]?.decision||'',cls=d==='keep'?'good':d==='reject'?'bad':d?'maybe':'';return `<div class="row ${i===current?'active':''}" tabindex="0" data-index="${i}"><span class="num">${n+1}</span><span><strong>${esc(t.artist)} · ${esc(t.title)}</strong><span class="meta">${esc(t.album)} · ${Math.round(t.seconds/60)} 分钟 · ${esc(t.priority)}</span></span><span class="pill ${cls}">${labels[d]||'待判断'}</span></div>`}).join('');}
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function choose(i,play=true){if(i<0||i>=tracks.length)return;const changed=current!==i;current=i;const t=tracks[i];$('#place').textContent=t.priority+' · '+t.album;$('#title').textContent=t.artist+' · '+t.title;$('#facts').textContent=`约 ${Math.floor(t.seconds/60)} 分 ${String(t.seconds%60).padStart(2,'0')} 秒 · ${t.id}`;$('#prior').textContent=t.prior?`先前短片段判断：${t.prior}${t.priorIssues?'；'+t.priorIssues:''}${t.priorNote?'；'+t.priorNote:''}。尚未代表整曲已通过。`:'先前没有逐曲听感判断。';$('#comment').value=state[t.id]?.comment||'';$('#ranges').value=state[t.id]?.ranges||'';$('#signal').textContent=`整曲机器提示：${t.clipSeconds??'未检查'} 秒出现较高削波比例；${t.quietSeconds??'未检查'} 秒电平很低。此指标不能判断底噪好坏。${t.readError?'读取异常：'+t.readError:''}`;$('#quiet').disabled=t.quietSeek===null;$('#playError').textContent='';if(changed){audio.pause();audio.src=t.uri;audio.load()}render();const active=$('#list .row.active');active?.scrollIntoView({block:'nearest'});if(play)audio.play().catch(()=>{$('#playError').textContent='无法直接播放 Y: 盘文件；请告诉我，我会改成本地试听副本。'})}
function next(delta,unrated=false){if(!visible.length)return;let pos=visible.indexOf(current);for(let k=1;k<=visible.length;k++){const i=visible[(pos+delta*k+visible.length*10)%visible.length];if(!unrated||!state[tracks[i].id]?.decision){choose(i);return}}}
function decide(decision){if(current<0)return;const t=tracks[current];state[t.id]={decision,comment:$('#comment').value,ranges:$('#ranges').value,updatedAt:new Date().toISOString()};save();render();if($('#auto').checked&&decision)next(1,true)}
$('#list').onclick=e=>{const row=e.target.closest('.row');if(row)choose(Number(row.dataset.index))};$('#list').onkeydown=e=>{if(e.key==='Enter'){const row=e.target.closest('.row');if(row)choose(Number(row.dataset.index))}};
['search','group','filter'].forEach(id=>$('#'+id).addEventListener(id==='search'?'input':'change',render));
$('#prev').onclick=()=>next(-1);$('#next').onclick=()=>next(1);$('#nextUnrated').onclick=()=>next(1,true);
document.querySelectorAll('[data-decision]').forEach(b=>b.onclick=()=>decide(b.dataset.decision));
['comment','ranges'].forEach(field=>$('#'+field).oninput=()=>{if(current<0)return;const id=tracks[current].id;state[id]={...(state[id]||{}),[field]:$('#'+field).value};save()});
document.querySelectorAll('[data-seek]').forEach(b=>b.onclick=()=>{if(current<0)return;const v=Number(b.dataset.seek);const go=()=>{audio.currentTime=v<1?(audio.duration||tracks[current].seconds)*v:v;audio.play()};if(audio.readyState)go();else audio.addEventListener('loadedmetadata',go,{once:true})});
$('#quiet').onclick=()=>{if(current<0)return;audio.currentTime=Math.max(0,tracks[current].quietSeek-3);audio.play()};
document.onkeydown=e=>{if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement?.tagName))return;const k=e.key.toLowerCase();if([' ','j','l','n','1','2','3','4'].includes(k))e.preventDefault();if(k===' ')audio.paused?audio.play():audio.pause();if(k==='j')audio.currentTime=Math.max(0,audio.currentTime-30);if(k==='l')audio.currentTime=Math.min(audio.duration||1e9,audio.currentTime+30);if(k==='n')next(1,true);if(k==='1')decide('keep');if(k==='2')decide('mood');if(k==='3')decide('partial');if(k==='4')decide('reject')};
audio.onended=()=>{if($('#auto').checked)next(1,true)};audio.onerror=()=>{$('#playError').textContent='音频加载失败；Y: 盘可能未连接。'};
$('#export').onclick=()=>{const blob=new Blob([JSON.stringify({experiment:'guqin_full_recordings_v1',source:'Y:\\Music\\古琴曲',state},null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='古琴整曲筛选反馈.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
$('#import').onclick=()=>$('#importFile').click();$('#importFile').onchange=async e=>{const f=e.target.files[0];if(!f)return;try{const data=JSON.parse(await f.text());if(!data.state||typeof data.state!=='object')throw Error('格式不符');state={...state,...data.state};save();render();if(current>=0)choose(current,false)}catch(err){alert('导入失败：'+err.message)}e.target.value=''};
render();choose(visible.find(i=>!state[tracks[i].id]?.decision)??visible[0]??0,false);
</script></body></html>'''.replace("TRACK_DATA", data)
    (OUT / "index.html").write_text(page, encoding="utf-8")
    (OUT / "候选整曲清单.json").write_text(json.dumps(tracks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"tracks": len(tracks),
                      "approved_sources": sum(group != "其他候选" for group in (t["priority"] for t in tracks)),
                      "whole_scanned": sum(t["clipSeconds"] is not None for t in tracks),
                      "html_bytes": len(page.encode("utf-8"))}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
