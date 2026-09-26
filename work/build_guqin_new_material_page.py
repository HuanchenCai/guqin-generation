"""Review page for new material: openly licensed downloads and DVD audio.

Nothing listed here is in training. The user decides per file; exported
decisions (with usable time ranges) drive any later corpus build.
"""

import html
import json
import os
import subprocess
from pathlib import Path

from build_guqin_inference_page import STYLES

WORK = Path(__file__).resolve().parent
OUT = WORK.parent / "outputs" / "古琴新素材待审"
DECISIONS = ("可用于训练", "部分可用（填时间段）", "不适合训练", "有人声或讲解")
ISSUES = ("人声/讲解", "底噪明显", "其他乐器", "压缩音质差", "爆音或失真", "录音太短")


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(path)], capture_output=True, text=True).stdout.strip()
    return float(out or 0)


def clock(seconds: float) -> str:
    return f"{int(seconds // 3600)}:{int(seconds % 3600 // 60):02d}:{int(seconds % 60):02d}" if seconds >= 3600 \
        else f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def link(source: Path) -> str:
    target = OUT / "audio" / source.name
    if not target.exists():
        os.link(source, target)
    return f"audio/{source.name}"


def card(rid: str, title: str, note: str, src: str, seconds: float) -> str:
    seeks = "".join(f'<button type="button" data-seek="{int(seconds * f)}">{label}</button>'
                    for f, label in ((0, "开头"), (0.25, "1/4"), (0.5, "一半"), (0.75, "3/4"))) if seconds > 120 else ""
    return (f'<article class="card"><div class="badge">{html.escape(title)}</div><p>{note}</p>'
            f'<audio controls preload="none" src="{src}"></audio><div class="seeks">{seeks}</div>'
            f'<div class="rating" data-id="{html.escape(rid, quote=True)}"><label>判断 <select><option value="">未判断</option>'
            + "".join(f"<option>{d}</option>" for d in DECISIONS) + '</select></label><div class="issues">'
            + "".join(f'<label><input type="checkbox" value="{x}">{x}</label>' for x in ISSUES)
            + '</div><input class="comment" placeholder="备注；部分可用请写时间段，如 0:30-3:20, 5:10-12:00"></div></article>')


def main() -> None:
    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    public = json.loads((WORK / "guqin_public_candidates.json").read_text(encoding="utf-8"))
    pieces, clips = [], []
    for row in public:
        path = WORK / "guqin_public_candidates" / row["file"]
        seconds = duration(path)
        note = (f'{html.escape(row["source"])} · {html.escape(str(row["author"]))} · '
                f'<a href="{html.escape(row["url"], quote=True)}">{html.escape(str(row["license"]))}</a> · {clock(seconds)}')
        target = pieces if seconds >= 60 else clips
        target.append(card(f'public/{row["file"]}', row["file"].split("_", 1)[1].rsplit(".", 1)[0].replace("_", " "),
                           note, link(path), seconds))
    dvds = []
    dedupe = json.loads((WORK / "guqin_dvd_dedupe_summary.json").read_text(encoding="utf-8"))
    for path in sorted((WORK / "guqin_dvd_audio").glob("DVD*.flac")):
        seconds = duration(path)
        info = dedupe.get(path.stem, {})
        dup = sorted(set(info.get("dup_training", [])) | set(info.get("dup_other_cd", [])))
        ranges = []
        for t in dup:  # each query covers t..t+12 s; merge queries 30 s apart
            if ranges and t - ranges[-1][1] <= 30:
                ranges[-1][1] = t + 12
            else:
                ranges.append([t, t + 12])
        first_pass = [h for h in json.loads((WORK / "guqin_dvd_dedupe.json").read_text(encoding="utf-8"))
                      if h["dvd"] == path.stem and h["dvd_seconds"] in set(info.get("dup_training", []))]
        sources = sorted({h["cd_artist"] + "《" + h["cd_piece"] + "》（已在训练集）" for h in first_pass}
                         | {Path(x.replace("\\", "/")).stem for x in info.get("other_cd_sources", [])})
        note = f"Y 盘 DVD 音轨，MP2 有损 48 kHz · {clock(seconds)}"
        if ranges:
            note += (" · <strong>与 CD 重复：</strong>" + "、".join(f"{clock(a)}–{clock(b)}" for a, b in ranges)
                     + "（" + "、".join(sources) + "）")
        else:
            note += " · 未发现与 CD 重复"
        dvds.append(card(f"dvd/{path.name}", path.stem, note, link(path), seconds))
    script = '''
const key='guqin_new_material_v1';let state={};try{state=JSON.parse(localStorage.getItem(key)||'{}')}catch{};
const save=()=>{try{localStorage.setItem(key,JSON.stringify(state))}catch{}};
document.querySelectorAll('audio').forEach(a=>a.onplay=()=>document.querySelectorAll('audio').forEach(b=>{if(b!==a)b.pause()}));
document.querySelectorAll('[data-seek]').forEach(b=>b.onclick=()=>{const a=b.closest('.card').querySelector('audio');a.currentTime=Number(b.dataset.seek);a.play()});
document.querySelectorAll('.rating').forEach(r=>{const id=r.dataset.id,old=state[id]||{},decision=r.querySelector('select'),checks=[...r.querySelectorAll('input[type=checkbox]')],comment=r.querySelector('.comment');
decision.value=old.decision||'';checks.forEach(c=>c.checked=(old.issues||[]).includes(c.value));comment.value=old.comment||'';
const update=()=>{state[id]={decision:decision.value,issues:checks.filter(c=>c.checked).map(c=>c.value),comment:comment.value};save()};decision.onchange=update;checks.forEach(c=>c.onchange=update);comment.oninput=update});
document.querySelector('#export').onclick=()=>{const blob=new Blob([JSON.stringify({experiment:'guqin_new_material_v1',state},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='古琴新素材审听反馈.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
'''
    section = lambda title, fine, cards: (f'<section class="case"><h2>{title}</h2><p class="fine">{fine}</p>'
                                          f'<div class="grid">{"".join(cards)}</div></section>') if cards else ""
    page = ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1"><title>古琴新素材待审</title>'
            '<style>' + STYLES + '.seeks{display:flex;gap:6px;flex-wrap:wrap}.seeks button{border:1px solid #cad9cc;'
            'background:#edf2ea;color:var(--accent);border-radius:6px;padding:5px 9px}</style></head><body>'
            '<header><div><h1>古琴新素材待审</h1><p>公开授权下载的录音和 Y 盘 DVD 音轨。这里的内容都没有进入训练，由你逐个判断。</p></div></header><main>'
            '<div class="intro">公开授权录音大多来自 Wikimedia Commons 上琴人 Charlie Huang 自己上传的演奏（CC BY / CC BY-SA / 公有领域），'
            '另有 Internet Archive 两条；每张卡片标了来源、作者和授权链接，若使用需要署名，CC BY-SA 与 CC BY-NC-SA 另有附加条件。'
            'DVD 音轨是整张盘的连续音频，可能夹有讲解或其他乐器，请用“部分可用”并写出时间段。已自动与 638 首 CD 录音比对：12 张 DVD 与 412 首训练录音基本不重复，只有 DVD01、DVD06 有几段与未入训的 CD 录音（含合奏）重复，卡片上已标出。</div>'
            + section("公开授权 · 完整曲目", "一分钟以上的录音。", pieces)
            + section("DVD 音轨", "每张 DVD 一条连续音轨，可用按钮跳到不同位置抽查。", dvds)
            + section("公开授权 · 短片段", "调弦、泛音等示范短片段，通常太短不适合训练，列出供参考。", clips)
            + '<button class="export" id="export">下载审听反馈</button>'
            '<p class="fine">判断自动保存在当前浏览器；下载 JSON 后发给我。</p></main>'
            '<script>' + script + '</script></body></html>')
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(f"{len(pieces)} pieces, {len(clips)} clips, {len(dvds)} DVDs -> {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
