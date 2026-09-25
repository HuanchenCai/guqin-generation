"""Build one listening/rating page for dataset x representation x generator."""

import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "outputs" / "古琴交叉实验"
CASES = [
    ("1995较新录音", "本库相对较新；已用于 MusicGen 微调", True),
    ("1988较旧录音", "本库较早录音；已用于 MusicGen 微调", True),
    ("未参与训练录音", "独立于这次 MusicGen 微调", False),
    ("1991中期录音", "介于两组年代之间；已用于 MusicGen 微调", False),
]
CODECS = [("EnCodec", "MusicGen 使用的 32 kHz 声音符号"),
          ("DAC", "44.1 kHz 的另一套声音符号"),
          ("SAME", "Stable Audio 3 使用的 44.1 kHz 连续潜表示")]
GENERATORS = [("原版", "MusicGen 原版"), ("古琴微调", "MusicGen 古琴微调"),
              ("SA3纯音频续写", "Stable Audio 3 Small Music")]


def player(name, note, filename, jump=False, score_id=None):
    score = ""
    if score_id:
        score = (f'<div class="rating" data-id="{score_id}"><label>统一评分 '
                 '<select><option value="">未评分</option>'
                 + ''.join(f'<option>{n}</option>' for n in range(1, 6))
                 + '</select></label><div class="issues">'
                 + ''.join(f'<label><input type="checkbox" value="{x}">{x}</label>'
                           for x in ("不像古琴", "重复", "拉锯/金属杂音", "接缝突兀", "太轻/太响"))
                 + '</div><input class="comment" placeholder="其他听感（可选）"></div>')
    return (f'<article class="card"><h4>{html.escape(name)}</h4><p>{html.escape(note)}</p>'
            f'<audio controls preload="none" src="{html.escape(filename, quote=True)}"></audio>'
            + ('<button class="jump" type="button">从第 9 秒听</button>' if jump else '')
            + score + '</article>')


def section(label, description, expanded):
    title = html.escape(label)
    intro = player("原始 10 秒", "同一段古琴输入；以下模型从第 10 秒开始续写。", f"{label}_原曲开头.wav")
    codecs = ''.join(player(codec, note, f"{label}_{codec}还原.wav") for codec, note in CODECS)
    gens = ''.join(player(name, "空文字描述，只给 10 秒古琴开头；后续约 12 秒由模型生成。",
                          f"{label}_{variant}.wav", True, f"{label}/{variant}")
                   for variant, name in GENERATORS)
    return (f'<details class="case" {"open" if expanded else ""}><summary>{title}<small>{html.escape(description)}</small></summary>'
            '<div class="casebody"><h3>先听声音表示：没有生成新音符</h3>'
            f'<div class="grid four">{intro}{codecs}</div>'
            f'<label class="codec-choice">最接近原曲的还原：<select data-codec="{title}"><option value="">未选</option>'
            + ''.join(f'<option>{name}</option>' for name, _ in CODECS)
            + '</select></label><h3>再听纯音频续写</h3>'
            f'<div class="grid three">{gens}</div></div></details>')


def main():
    expected = []
    for label, _, _ in CASES:
        expected.extend([f"{label}_原曲开头.wav", *(f"{label}_{c}还原.wav" for c, _ in CODECS),
                         *(f"{label}_{v}.wav" for v, _ in GENERATORS)])
    expected.extend(["1995较新录音_SA3一分钟.wav",
                     *(f"1995较新录音_SA3一分钟_{n}秒.wav" for n in (10, 30, 50))])
    missing = [name for name in expected if not (OUT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} cross experiment samples: {missing[:4]}")
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>古琴生成模型交叉试听</title><style>
:root{--bg:#f3f0e8;--paper:#fffdfa;--ink:#263a32;--muted:#61736b;--green:#355e4c;--line:#d6ded5;--gold:#c5a77a}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 "Microsoft YaHei",system-ui,sans-serif}header{background:linear-gradient(120deg,#1d372d,#385e4d);color:#faf8ee;padding:40px 24px}header>div,main{max-width:1200px;margin:auto}h1{font:normal clamp(32px,5vw,53px)/1.2 "STSong",serif;margin:4px 0 12px}header p{max-width:850px;margin:0;color:#e3eade}main{padding:30px 24px 70px}h2{font:normal 30px "STSong",serif;margin:0 0 10px}h3{font-size:18px;margin:24px 0 10px}.intro{background:#e9eee8;border-left:4px solid var(--green);padding:13px 17px;margin:0 0 24px}.facts{display:flex;gap:10px;flex-wrap:wrap;margin:20px 0 28px}.fact{background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:10px 14px}.fact strong{font-size:19px}.case{background:var(--paper);border:1px solid var(--line);border-radius:14px;margin:15px 0;overflow:hidden}summary{cursor:pointer;padding:17px 21px;font-size:21px;font-weight:650;list-style-position:inside;background:#fbfaf5}summary small{font-size:13px;color:var(--muted);font-weight:400;margin-left:12px}.casebody{padding:0 20px 23px}.grid{display:grid;gap:12px}.grid.four{grid-template-columns:repeat(4,minmax(0,1fr))}.grid.three{grid-template-columns:repeat(3,minmax(0,1fr))}.card{border:1px solid var(--line);border-radius:10px;padding:14px;min-width:0;background:#fff}.card h4{margin:0 0 3px;font-size:15px}.card p{font-size:12px;color:var(--muted);margin:0;min-height:38px}audio{width:100%;display:block;margin:10px 0 5px}.jump{border:1px solid #c9d6ca;background:#eef3ed;color:var(--green);border-radius:7px;padding:4px 9px;cursor:pointer}.codec-choice{display:inline-block;margin:13px 0 4px;font-weight:650}select{font:inherit;border:1px solid var(--line);background:white;border-radius:7px;padding:3px 7px}.rating{font-size:12px;margin-top:10px;border-top:1px solid var(--line);padding-top:10px}.issues{display:flex;gap:5px;flex-wrap:wrap;margin-top:7px}.issues label{border:1px solid var(--line);border-radius:99px;padding:2px 6px;white-space:nowrap}.issues input{margin:0 3px 0 0;vertical-align:middle}.comment{margin-top:7px;width:100%;border:1px solid var(--line);border-radius:6px;padding:5px 7px;font:inherit}.fine{font-size:13px;color:var(--muted)}#export{background:var(--green);color:white;border:0;border-radius:9px;padding:10px 18px;font:inherit;cursor:pointer;margin:25px 0 8px}a{color:var(--green)}@media(max-width:930px){.grid.four{grid-template-columns:repeat(2,minmax(0,1fr))}.grid.three{grid-template-columns:1fr}}@media(max-width:600px){.grid.four{grid-template-columns:1fr}header{padding:32px 16px}main{padding:22px 14px}summary small{display:block;margin:2px 0 0 22px}}
</style></head><body><header><div><div style="color:#e4cfa9;font-size:12px;letter-spacing:.15em">AUDIO × REPRESENTATION × GENERATOR</div><h1>古琴生成方向：交叉试听</h1><p>同一段古琴开头，先比较三套声音表示能否还原音色，再比较三个生成器能否自然续写。全部没有意境提示词。每段生成音频前约 10 秒是输入，之后约 12 秒才是模型新生成。</p></div></header><main><div class="intro"><strong>怎么听：</strong>先听原曲和三种“编码后还原”，判断是否丢失泛音、滑音与尾音；再从第 9 秒跳到接缝，给三个生成器各打一个 1–5 分。MusicGen 微调只用了本库 10 首独奏共 36 分钟；Stable Audio 3 目前是未微调的原版。DAC 目前只有还原试听，还没有与之配套的已训练生成器。</div><div class="facts"><div class="fact"><strong>10 首</strong><br>确认单人独奏</div><div class="fact"><strong>36 分钟</strong><br>本次微调素材</div><div class="fact"><strong>1988–1995</strong><br>录音年代</div><div class="fact"><strong>44.1 kHz / 16 位</strong><br>转存文件规格，不能代表原始母带品质</div></div><h2>四段同条件实验</h2><p class="fine">默认展开年代两端。1991 年和未参与微调的录音也可展开。音量按模型原样保留，过轻或过响请标记。</p>
__CASES__
<button id="export">下载统一评分</button><p class="fine">评分只保存在当前浏览器，可下载 JSON 发回给我。<a href="素材质量统计.json">素材统计</a> · <a href="还原频谱比较.json">频谱辅助比较</a>。频谱数字只作辅助，不代替听感。</p></main><script>
const key='guqin_cross_v1';let state={};try{state=JSON.parse(localStorage.getItem(key)||'{}')}catch{};const save=()=>localStorage.setItem(key,JSON.stringify(state));document.querySelectorAll('audio').forEach(a=>a.addEventListener('play',()=>document.querySelectorAll('audio').forEach(b=>{if(b!==a)b.pause()})));document.querySelectorAll('.jump').forEach(b=>b.onclick=()=>{const a=b.parentElement.querySelector('audio');a.currentTime=9;a.play()});document.querySelectorAll('[data-codec]').forEach(s=>{const id=s.dataset.codec;s.value=state[id]?.best_codec||'';s.onchange=()=>{state[id]={...(state[id]||{}),best_codec:s.value};save()}});document.querySelectorAll('.rating').forEach(r=>{const id=r.dataset.id,old=state[id]||{},score=r.querySelector('select'),checks=[...r.querySelectorAll('input[type=checkbox]')],comment=r.querySelector('.comment');score.value=old.score||'';checks.forEach(c=>c.checked=(old.issues||[]).includes(c.value));comment.value=old.comment||'';const update=()=>{state[id]={score:score.value,issues:checks.filter(c=>c.checked).map(c=>c.value),comment:comment.value};save()};score.onchange=update;checks.forEach(c=>c.onchange=update);comment.oninput=update});document.querySelector('#export').onclick=()=>{const blob=new Blob([JSON.stringify({experiment:'guqin_audio_cross_v1',scoring:'1-5 higher is better',state},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='古琴交叉实验统一评分.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
</script></body></html>'''
    page = page.replace("__CASES__", "".join(section(*case) for case in CASES))
    long_section = ('<section style="margin-top:32px"><h2>单次生成一分钟</h2>'
                    '<p class="fine">使用同一段 1995 年古琴开头，Stable Audio 3 一次生成到 60 秒；'
                    '没有反复回送生成结果。模型加载后的生成与解码耗时 17.4 秒。'
                    '最后十秒的噪声指标升高，听感需要再判断。</p>'
                    + player("完整一分钟", "前十秒是输入，之后五十秒是生成。",
                             "1995较新录音_SA3一分钟.wav", True, "long/sa3")
                    + '<div class="grid three" style="margin-top:12px">'
                    + ''.join(player(f"第 {n} 秒开始", "抽取十秒，便于比较有没有逐渐跑偏。",
                                     f"1995较新录音_SA3一分钟_{n}秒.wav") for n in (10, 30, 50))
                    + '</div></section>')
    page = page.replace('<button id="export">', long_section + '<button id="export">')
    page = page.replace('<a href="素材质量统计.json">素材统计</a>',
                        '<a href="实验记录.md">实验记录与结论</a> · <a href="素材质量统计.json">素材统计</a>')
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
