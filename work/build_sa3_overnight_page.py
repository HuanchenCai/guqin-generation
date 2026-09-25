"""Build a single, plainly named listening page from completed SA3 runs."""

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "outputs" / "古琴夜间训练"
TAGS = [
    ("原版", "原版模型", "未用本曲库微调"),
    ("Medium原版", "更大模型原版", "不同声音编码；未微调"),
    ("48段500步", "原来 48 段", "36 分钟，训练 500 步"),
    ("139段500步", "扩充 139 段", "127 分钟，训练 500 步"),
    ("139段1000步", "扩充 139 段，多训练", "127 分钟，训练 1000 步"),
    ("123段低噪1000步", "扩充且筛底噪", "123 段，训练 1000 步"),
    ("Medium123段500步", "更大模型，先停训练", "123 段，训练 500 步"),
    ("Medium123段1000步", "更大模型，同素材", "123 段，训练 1000 步"),
    ("Medium222段500步", "更大模型，扩充录音", "222 首各取一段，训练 500 步"),
    ("Medium222段1000步", "更大模型，扩充录音并多训练", "222 首各取一段，训练 1000 步"),
    ("102段保守独奏1000步", "小模型，旧102段组", "训练 1000 步；当时的自动乐器筛选有误判"),
    ("Medium102段保守独奏1000步", "大模型，旧102段组", "训练 1000 步；当时的自动乐器筛选有误判"),
]
SOURCES = ["较新录音", "未见录音"]
SEEDS = [84917, 94017]
RADIO_TAGS = [("原版", "原版模型连续生成"),
              ("139段1000步", "古琴微调后直接连续生成"),
              ("139段1000步电平控制", "控制音量后连续生成"),
              ("102段保守独奏1000步", "旧102段组连续生成"),
              ("Medium原版单次", "更大模型原版单次长段"),
              ("Medium123段1000步单次", "更大模型单次长段生成"),
              ("Medium102段保守独奏1000步单次", "大模型旧102段组单次长段")]

SOURCE_LABELS = {"较新录音": "旧对照：谢孝苹录音（底噪较大）", "未见录音": "未参与训练的录音"}


def card(tag, title, note, filename, source, seed, signal):
    score_id = f"{source}/{seed}/{tag}"
    anchor = ' id="medium-heldout-score"' if (source, seed, tag) == ("未见录音", 94017, "Medium123段500步") else ''
    loud_warning = ('<p class="warning">这段可能过响，请先调低音量。</p>'
                    if signal.get("clipped_fraction", 0) > 0.001 else '')
    return (f'<article class="card"{anchor}><div class="badge">{html.escape(title)}</div>'
            f'<p>{html.escape(note)}</p><audio controls preload="none" src="{html.escape(filename, quote=True)}"></audio>'
            + loud_warning
            + '<button class="jump" type="button">从续写处听</button>'
            f'<div class="rating" data-id="{html.escape(score_id, quote=True)}">'
            '<label>综合评分 <select><option value="">未评分</option>'
            + ''.join(f'<option value="{n}">{n}</option>' for n in range(1, 6))
            + '</select></label><div class="issues">'
            + ''.join(f'<label><input type="checkbox" value="{x}">{x}</label>'
                      for x in ("不像古琴", "重复", "没旋律", "杂音", "接缝突兀", "音量不稳"))
            + '</div><input class="comment" placeholder="其他听感（可选）"></div></article>')


def main():
    runs = {}
    for tag, _, _ in TAGS:
        report = OUT / f"{tag}_记录.json"
        if report.exists():
            runs[tag] = {(row["source"], row["seed"]): row for row in json.loads(report.read_text(encoding="utf-8"))}
    if not runs:
        raise RuntimeError("No completed listening runs")
    codecs = []
    for source in SOURCES:
        files = [("原曲", f"{source}_原曲开头.wav"),
                 ("DAC", f"{source}_DAC还原.wav"),
                 ("SAME-S", f"{source}_SAME还原.wav"),
                 ("SAME-L", f"{source}_SAME-L还原.flac")]
        if all((OUT / filename).exists() for _, filename in files):
            players = ''.join(f'<div class="codec"><strong>{name}</strong><audio controls preload="none" src="{html.escape(filename, quote=True)}"></audio></div>'
                              for name, filename in files)
            codecs.append(f'<section class="case"><h2>{html.escape(SOURCE_LABELS[source])} · 音色还原</h2>'
                          '<p class="fine">这里都没有生成新音符，只比较声音编码后能否还原原曲。SAME-L 属于更大的 Medium 模型。</p>'
                          + f'<div class="grid">{players}</div>'
                          + f'<label class="codec-choice">最喜欢的还原 <select data-codec="{html.escape(source, quote=True)}">'
                          '<option value="">未选择</option><option>DAC</option><option>SAME-S</option><option>SAME-L</option></select></label></section>')
    sections = []
    for source in SOURCES:
        for seed in SEEDS:
            cards = []
            for tag, title, note in TAGS:
                if tag.startswith("Medium222段"):
                    continue
                row = runs.get(tag, {}).get((source, seed))
                if row and (OUT / row["file"]).exists():
                    cards.append(card(tag, title, note, row["file"], source, seed, row["signal"]))
            if cards:
                sections.append(f'<section class="case"><h2>{html.escape(SOURCE_LABELS[source])} · 选段 {SEEDS.index(seed)+1}</h2>'
                                '<p class="fine">各组以同一段原曲、同一种子作条件；前 10 秒经各模型重新编码还原，第 10 秒后才是新生成。原曲原声可在上方“音色还原”中听。</p>'
                                + f'<div class="grid">{"".join(cards)}</div></section>')
    bulk_sections = []
    for seed in SEEDS:
        cards = []
        for tag in ("Medium123段500步", "Medium123段1000步", "Medium222段500步", "Medium222段1000步"):
            row = runs.get(tag, {}).get(("未见录音", seed))
            if row and (OUT / row["file"]).exists():
                title, note = next((title, note) for item_tag, title, note in TAGS if item_tag == tag)
                cards.append(card(tag, title, note, row["file"], "未见录音", seed, row["signal"]))
        if cards:
            bulk_sections.append(f'<section class="case"><h2>扩充录音对照 · 选段 {SEEDS.index(seed)+1}</h2>'
                                 '<p class="fine">四段使用同一未参与训练的原曲开头和同一种子；从第 10 秒开始听生成。请比较 123 段与 222 段、500 步与 1000 步。录音数量增加不保证效果更好。</p>'
                                 + f'<div class="grid">{"".join(cards)}</div></section>')
    radio_cards = []
    for tag, title in RADIO_TAGS:
        report_path = OUT / f"电台接续_{tag}_记录.json"
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not (OUT / report["file"]).exists():
            continue
        loud_warning = ('<p class="warning">这段有明显音量过高的风险，播放前请先调低音量。</p>'
                        if any(row["signal"]["clipped_fraction"] > 0.001 for row in report["chunks"]) else '')
        issue_options = ("不像古琴", "重复", "没旋律", "杂音", "接缝突兀", "音量爬升")
        radio_cards.append(
            f'<article class="card radio"><div class="badge">{html.escape(title)}</div>'
            f'<p>共 {report["total_seconds"] / 60:.1f} 分钟，{len(report["chunks"])} 段生成；从头、中、尾检查有没有跑偏。</p>'
            + loud_warning +
            f'<audio controls preload="none" src="{html.escape(report["file"], quote=True)}"></audio>'
            '<div class="seeks">'
            + ''.join(f'<button type="button" data-seek="{excerpt["start_seconds"]}">听{label}</button>'
                      for excerpt, label in zip(report["excerpts"], ("开头", "中段", "末段")))
            + '</div>'
            f'<div class="rating" data-id="radio/{html.escape(tag, quote=True)}">'
            '<label>连续播放评分 <select><option value="">未评分</option>'
            + ''.join(f'<option value="{n}">{n}</option>' for n in range(1, 6))
            + '</select></label><div class="issues">'
            + ''.join(f'<label><input type="checkbox" value="{x}">{x}</label>' for x in issue_options)
            + '</div><input class="comment" placeholder="其他听感（可选）"></div></article>')
    radio_section = ('<section class="case"><h2>长段连续播放</h2>'
                     '<p class="fine">比较原版、多次接续、音量控制和更大模型的一次生成。请特别听末段是否变成怪声。</p>'
                     + f'<div class="grid">{"".join(radio_cards)}</div></section>') if radio_cards else ''
    new_models_section = ''
    magenta_file = "Magenta实时2_纯音频参考_60秒.wav"
    magenta_base_file = "Magenta实时2_Base纯音频参考_40秒.wav"
    ace_file = "ACE基础版_同原曲续写_35秒.flac"
    sa3_file = "未见录音_Medium123段500步_94017.flac"
    original_file = "未见录音_原曲开头.wav"
    if all((OUT / name).exists() for name in (magenta_file, sa3_file, original_file)):
        issues = ("不像古琴", "其他乐器/低音", "重复", "旋律杂乱", "杂音", "音量不稳")
        rating = ('<div class="rating" data-id="mrt2_small/audio_style/heldout/60s">'
                  '<label>Magenta 综合评分 <select><option value="">未评分</option>'
                  + ''.join(f'<option value="{n}">{n}</option>' for n in range(1, 6))
                  + '</select></label><div class="issues">'
                  + ''.join(f'<label><input type="checkbox" value="{html.escape(x, quote=True)}">{html.escape(x)}</label>'
                            for x in issues)
                  + '</div><input class="comment" placeholder="音色、旋律与第 20/40 秒附近的听感"></div>')
        base_card = ''
        if (OUT / magenta_base_file).exists():
            base_rating = ('<div class="rating" data-id="mrt2_base/audio_style/heldout/40s">'
                           '<label>Base 综合评分 <select><option value="">未评分</option>'
                           + ''.join(f'<option value="{n}">{n}</option>' for n in range(1, 6))
                           + '</select></label><div class="issues">'
                           + ''.join(f'<label><input type="checkbox" value="{html.escape(x, quote=True)}">{html.escape(x)}</label>'
                                     for x in issues)
                           + '</div><input class="comment" placeholder="与 Small 的音色和旋律比较"></div>')
            base_card = (f'<article class="card"><div class="badge">Magenta RealTime 2 Base</div>'
                         '<p>同一原曲的纯音频风格参考，连续生成 40 秒；使用官方权重和半精度运行。</p>'
                         f'<audio controls preload="none" src="{html.escape(magenta_base_file, quote=True)}"></audio>{base_rating}</article>')
        ace_card = ''
        if (OUT / ace_file).exists():
            ace_rating = ('<div class="rating" data-id="ace_base/same_source/heldout/35s">'
                          '<label>ACE 综合评分 <select><option value="">未评分</option>'
                          + ''.join(f'<option value="{n}">{n}</option>' for n in range(1, 6))
                          + '</select></label><div class="issues">'
                          + ''.join(f'<label><input type="checkbox" value="{html.escape(x, quote=True)}">{html.escape(x)}</label>'
                                    for x in issues)
                          + '</div><input class="comment" placeholder="从第 10 秒开始听续写；与 Medium 比较"></div>')
            ace_card = (f'<article class="card"><div class="badge">ACE-Step 1.5 Base</div>'
                        '<p>同一原曲，前 10 秒保留、后 25 秒续写；用了最简“独奏古琴”文字标签。旧版其他原曲的评分是 1–2 分。</p>'
                        f'<audio controls preload="none" src="{html.escape(ace_file, quote=True)}"></audio>'
                        f'<button class="jump" type="button">从续写处听</button>{ace_rating}</article>')
        new_models_section = (
            '<section class="case"><h2>新模型路线：纯音频引导与续写</h2>'
            '<p class="fine">这些试听使用同一段未参与训练的古琴原曲。Medium 与 ACE 保留前 10 秒并续写；Magenta 只提取原曲的音色风格，然后从头生成，不能当作同开头续写。Medium、Magenta 没有文字或 MIDI；ACE 使用最简文字标签。旧 ACE 的 1–2 分来自另一段原曲。</p>'
            '<div class="grid">'
            f'<article class="card"><div class="badge">原曲 · 10 秒</div><p>先听古琴的原声音色。</p><audio controls preload="none" src="{html.escape(original_file, quote=True)}"></audio></article>'
            f'<article class="card"><div class="badge">Stable Audio 3 Medium</div><p>旧基线：前 10 秒是原曲条件，后 25 秒由模型续写。<a href="#medium-heldout-score">到原有评分位置</a></p><audio controls preload="none" src="{html.escape(sa3_file, quote=True)}"></audio></article>'
            f'<article class="card"><div class="badge">Magenta RealTime 2 Small</div><p>新测试：以原曲作纯音频风格参考，连续生成 60 秒；第 20、40 秒保持同一生成状态。生成耗时约 18 秒，无削波。</p><audio controls preload="none" src="{html.escape(magenta_file, quote=True)}"></audio>{rating}</article>'
            + base_card + ace_card +
            '</div></section>'
        )
    styles = '''
    :root{--bg:#eeeae0;--paper:#fffdf8;--ink:#26372e;--muted:#68756b;--line:#d7ded4;--accent:#355c46}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 "Microsoft YaHei",system-ui,sans-serif}
    header{background:linear-gradient(130deg,#1e3429,#456d55);color:#faf8ef;padding:43px 24px}header>div,main{max-width:1280px;margin:auto}
    h1{font:normal clamp(33px,5vw,55px)/1.25 "STSong",serif;margin:0 0 10px}header p{margin:0;max-width:920px;color:#e2e8de}
    main{padding:28px 24px 70px}.intro{padding:15px 18px;background:#f8f7f0;border-left:4px solid var(--accent);margin-bottom:25px}
    .case{margin:0 0 34px}h2{font:normal 27px "STSong",serif;margin:0 0 3px}.fine{font-size:13px;color:var(--muted);margin:0 0 12px}
    .grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.card{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:15px;min-width:0}
    .badge{font-size:17px;font-weight:700}.card p{font-size:12px;color:var(--muted);margin:3px 0 8px;min-height:22px}audio{width:100%;margin:8px 0}
    button{font:inherit;cursor:pointer}.jump{background:#edf2ea;color:var(--accent);border:1px solid #cad9cc;border-radius:6px;padding:5px 10px}
    .rating{border-top:1px solid var(--line);margin-top:12px;padding-top:10px;font-size:13px}select,.comment{border:1px solid var(--line);background:white;border-radius:6px;padding:4px;font:inherit}
    .issues{display:flex;gap:4px;flex-wrap:wrap;margin-top:8px}.issues label{border:1px solid var(--line);border-radius:99px;padding:2px 7px;white-space:nowrap}.issues input{margin-right:3px}
    .comment{width:100%;margin-top:8px}.export{background:var(--accent);color:white;border:0;border-radius:8px;padding:10px 18px}
    .codec{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:14px}.codec-choice{display:inline-block;margin:11px 0}.codec audio{margin-bottom:0}
    .radio{grid-column:span 2}.seeks{display:flex;gap:6px;flex-wrap:wrap}.seeks button{border:1px solid #cad9cc;background:#edf2ea;color:var(--accent);border-radius:6px;padding:5px 9px}
    .warning{color:#8d391e!important;font-weight:700;min-height:0!important}
    .review{background:#f8f7f0;border:1px solid var(--line);border-radius:12px;padding:15px;margin:28px 0}.review summary{font-size:19px;cursor:pointer}.review .fine{margin:10px 0}
    @media(max-width:1100px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:680px){.grid{grid-template-columns:1fr}.radio{grid-column:auto}main{padding:22px 14px}header{padding:31px 16px}}
    '''
    script = '''
    const key='guqin_sa3_overnight_v1';let state={};try{state=JSON.parse(localStorage.getItem(key)||'{}')}catch{};
    const save=()=>localStorage.setItem(key,JSON.stringify(state));
    document.querySelectorAll('audio').forEach(a=>a.onplay=()=>document.querySelectorAll('audio').forEach(b=>{if(b!==a)b.pause()}));
    document.querySelectorAll('[data-codec]').forEach(s=>{const id='codec/'+s.dataset.codec;s.value=state[id]||'';s.onchange=()=>{state[id]=s.value;save()}});
    document.querySelectorAll('[data-review]').forEach(s=>{const id='review/'+s.dataset.review;s.value=state[id]||'';s.onchange=()=>{state[id]=s.value;save()}});
    document.querySelectorAll('.jump').forEach(b=>b.onclick=()=>{const a=b.parentElement.querySelector('audio');a.currentTime=9;a.play()});
    document.querySelectorAll('[data-seek]').forEach(b=>b.onclick=()=>{const a=b.closest('.card').querySelector('audio');a.currentTime=Number(b.dataset.seek);a.play()});
    document.querySelectorAll('.rating').forEach(r=>{const id=r.dataset.id,old=state[id]||{},score=r.querySelector('select'),checks=[...r.querySelectorAll('input[type=checkbox]')],comment=r.querySelector('.comment');
    score.value=old.score||'';checks.forEach(c=>c.checked=(old.issues||[]).includes(c.value));comment.value=old.comment||'';
    const update=()=>{state[id]={score:score.value,issues:checks.filter(c=>c.checked).map(c=>c.value),comment:comment.value};save()};score.onchange=update;checks.forEach(c=>c.onchange=update);comment.oninput=update});
    document.querySelector('#export').onclick=()=>{const blob=new Blob([JSON.stringify({experiment:'guqin_sa3_overnight_v1',scoring:'1-5 higher is better',state},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='古琴夜间训练统一评分.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
    '''
    page = ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>古琴夜间训练试听</title><style>' + styles + '</style></head><body><header><div><h1>古琴夜间训练试听</h1>'
            '<p>同一段古琴开头，比较原版、原曲库微调和扩充曲库微调，并加入其他生成模型试听。</p></div></header><main>'
            '<div class="intro">先听每组从第 10 秒开始的续写，重点判断是否仍是单人古琴、旋律是否自然、有没有重复或怪声。'
            '这是模型方向实验，尚未选定能连续播放数小时的版本。</div>'
            '<div class="intro"><strong>新一轮批量试验：</strong>已从 47 位演奏者的同来源录音中，整曲检查 244 首并选出 222 首各一分钟，共 3 小时 42 分钟。下方四格比较旧 123 段与新 222 段训练结果；无需逐首听完整曲库。新模型的长段表现还未验证。'
            '<a href="实验记录.md">查看分析与下一步</a> · <a href="全曲库粗筛清单.csv">查看旧版粗筛清单</a> · <a href="../古琴整曲筛选/index.html">打开整曲试听页</a>。原有续写开头未更换。</div>'
            + ''.join(bulk_sections) + new_models_section + ''.join(codecs) + ''.join(sections) + radio_section + '<button class="export" id="export">下载统一评分</button>'
            '<p class="fine">评分自动保存在当前浏览器；下载 JSON 后可以直接发给我。</p></main><script>' + script + '</script></body></html>')
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
