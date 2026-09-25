"""Publish a provisional guqin training list and source-recording listening page."""

import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


WORK = Path(__file__).resolve().parent
SOURCE = Path(r"Y:\Music\古琴曲")
OUTPUT = WORK.parent / "outputs" / "古琴录音筛选"
CLIPS = OUTPUT / "试听片段"
BAD_ARTISTS = {"张育瑾", "谢孝苹"}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def strict_candidate(row: dict) -> bool:
    return (row["筛选状态"].startswith("候选")
            and float(row["时长分钟"]) > 1
            and float(row["抽样削波比例"]) < 0.0001
            and float(row["安静处高频相对全曲dB"]) < -50
            and float(row["安静处总声相对全曲dB"]) < -9)


def listen_priority(row: dict) -> float:
    clip = float(row["抽样削波比例"])
    hf = float(row["安静处高频相对全曲dB"])
    gap = float(row["安静处总声相对全曲dB"])
    return min(clip * 2000, 20) + max(hf + 50, 0) / 6 + max(gap + 9, 0) / 3


def sample_at(stream: sf.SoundFile, fraction: float, seconds: float) -> np.ndarray:
    count = min(round(seconds * stream.samplerate), stream.frames)
    start = round(max(stream.frames - count, 0) * fraction)
    stream.seek(start)
    return stream.read(count, dtype="float32", always_2d=True)


def make_excerpt(relative_path: str, target: Path) -> None:
    path = SOURCE / Path(relative_path.replace("\\", "/"))
    with sf.SoundFile(path) as stream:
        active_options = [(float(np.sqrt(np.mean(x.astype("float64") ** 2))), fraction)
                          for fraction in (0.18, 0.38, 0.58, 0.78)
                          for x in [sample_at(stream, fraction, 4)]]
        active_fraction = max(active_options)[1]
        quiet_options = [(float(np.sqrt(np.mean(x.astype("float64") ** 2))), fraction)
                         for fraction in (0.1, 0.3, 0.5, 0.7, 0.9)
                         for x in [sample_at(stream, fraction, 4)]]
        audible = [item for item in quiet_options if item[0] > 10 ** (-55 / 20)]
        quiet_fraction = min(audible or quiet_options)[1]
        active = sample_at(stream, active_fraction, 12)
        quiet = sample_at(stream, quiet_fraction, 4)
        gap = np.zeros((round(stream.samplerate * 0.3), stream.channels), dtype="float32")
        excerpt = np.concatenate((active, gap, quiet))
        sf.write(target, excerpt, stream.samplerate, format="FLAC", subtype="PCM_16")


def review_card(row: dict, number: int) -> str:
    escape = lambda value: html.escape(str(value), quote=True)
    labels = ("底噪明显", "爆音或失真", "过闷", "声音不稳定", "音色自然", "有氛围可保留")
    anchor = str(row["已知噪声参照"]).lower() == "true"
    preferred = str(row["本曲技术优先"]).lower() == "true"
    source_label = "已知噪声参照" if anchor else (
        "技术风险较少" if preferred else "需要重点听")
    previous = (f'<p class="hint">同一演奏者的上一段：{escape(row["前一段评价"])}</p>'
                if row.get("前一段评价") else '')
    return (
        f'<article class="card" data-search="{escape(row["演奏者"] + " " + row["曲名"])}">'
        f'<div class="top"><span class="number">{number:02d}</span><strong>{escape(row["演奏者"])}</strong>'
        f'<span class="status">{source_label}</span></div>'
        f'<p class="song">{escape(row["曲名"])} · 该演奏者候选 {row["该演奏者候选首数"]} 首，'
        f'其中技术优先 {row["该演奏者技术优先首数"]} 首</p>'
        '<p class="hint">前 12 秒听演奏；短暂停顿后 4 秒听较安静处的底声。</p>'
        + previous +
        f'<audio controls preload="none" src="{escape(row["试听文件"])}"></audio>'
        f'<div class="rating" data-id="{escape(row["Y盘相对路径"])}">'
        '<label>录音判断 <select><option value="">未评</option><option value="可用于训练">可用于训练</option>'
        '<option value="有氛围，待处理">有氛围，待处理</option><option value="不适合训练">不适合训练</option></select></label>'
        '<div class="issues">' + ''.join(
            f'<label><input type="checkbox" value="{escape(label)}">{escape(label)}</label>' for label in labels)
        + '</div><input class="comment" placeholder="其他听感（可选）"></div></article>'
    )


def build_page(review_rows: list[dict], second_rows: list[dict] | None = None,
               seed_state: dict | None = None) -> None:
    priority = review_rows[:25]
    remaining = review_rows[25:-2]
    anchors = review_rows[-2:]
    styles = """
    :root{--ink:#2c372d;--muted:#6b756a;--green:#385b41;--line:#d7dfd3;--paper:#fffdf8}
    *{box-sizing:border-box}body{margin:0;color:var(--ink);background:#eeefe8;font:15px/1.55 system-ui,"Microsoft YaHei",sans-serif}
    header{background:#26422f;color:#fff;padding:32px 22px}header div,main{max-width:1120px;margin:auto}
    h1{font:normal 42px "STSong",serif;margin:0 0 8px}header p{max-width:850px;margin:0;color:#e4ede1}
    main{padding:20px 22px 70px}.notice{background:#fff8dd;border-left:4px solid #ad8640;padding:12px 16px;margin:0 0 16px}
    .bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;position:sticky;top:0;background:#eeefe8;padding:10px 0;z-index:2}
    input[type=search]{font:inherit;padding:9px 12px;min-width:min(330px,100%);border:1px solid #abbca9;border-radius:6px}
    button{font:inherit;border:0;border-radius:7px;padding:10px 17px;cursor:pointer;color:#fff;background:var(--green)}
    section{margin:24px 0}h2,summary{font:normal 27px "STSong",serif;margin:0 0 8px}summary{cursor:pointer}
    .grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.card{background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:16px;min-width:0}
    .top{display:flex;align-items:center;gap:9px}.top strong{font-size:19px}.number{color:var(--muted)}.status{margin-left:auto;background:#e5ede2;padding:2px 8px;border-radius:20px;font-size:12px}
    .song{margin:6px 0;color:var(--green)}.hint{font-size:12px;color:var(--muted);margin:0}audio{width:100%;margin:7px 0}
    .rating select,.comment{font:inherit;border:1px solid #cad6c8;border-radius:5px;padding:6px}.issues{display:flex;flex-wrap:wrap;gap:3px 10px;margin:8px 0}.issues label{font-size:12px;white-space:nowrap}
    .comment{width:100%}.fine{font-size:12px;color:var(--muted)}.hidden{display:none!important}@media(max-width:780px){.grid{grid-template-columns:1fr}h1{font-size:32px}}
    """
    seed_json = json.dumps(seed_state or {}, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e')
    script = "const imported=" + seed_json + ";" + """
    const key='guqin_recording_screen_v1';let state={...imported};try{state={...imported,...JSON.parse(localStorage.getItem(key)||'{}')}}catch(e){}
    function save(){localStorage.setItem(key,JSON.stringify(state));document.querySelector('#count').textContent=Object.values(state).filter(x=>x.decision).length+' 段已评'}
    document.querySelectorAll('.rating').forEach(r=>{const id=r.dataset.id,old=state[id]||{},select=r.querySelector('select'),checks=[...r.querySelectorAll('input[type=checkbox]')],comment=r.querySelector('.comment');
      select.value=old.decision||'';checks.forEach(c=>c.checked=(old.issues||[]).includes(c.value));comment.value=old.comment||'';
      const update=()=>{state[id]={decision:select.value,issues:checks.filter(c=>c.checked).map(c=>c.value),comment:comment.value};save()};
      select.onchange=update;checks.forEach(c=>c.onchange=update);comment.oninput=update});save();
    document.querySelector('#search').oninput=e=>{const q=e.target.value.trim().toLowerCase();document.querySelectorAll('.card').forEach(c=>c.classList.toggle('hidden',!!q&&!c.dataset.search.toLowerCase().includes(q)));
      if(q){document.querySelector('#other').open=true;const second=document.querySelector('#second');if(second)second.open=true}};
    document.querySelector('#export').onclick=()=>{const blob=new Blob([JSON.stringify({experiment:'guqin_recording_screen_v1',state},null,2)],{type:'application/json'});
      const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='古琴录音筛选反馈.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
    """
    page = ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>古琴录音筛选</title><style>' + styles + '</style></head><body><header><div><h1>古琴录音筛选</h1>'
            '<p>先辨别录音质感，再扩大 Stable Audio 3 Medium 的训练集。每位演奏者先抽一首代表录音；第二轮再换一首核对。评分只适用于听过的曲目，不自动推断该演奏者所有录音。</p>'
            '</div></header><main><div class="notice">机器指标无法可靠认出已知的噪声录音。这里的“技术风险较少”只表示抽样未见明显削波、底声偏高等信号；请按听感决定是否用于训练。'
            '张育瑾、谢孝苹仅作噪声参照，不进入训练。</div>'
            '<div class="bar"><input id="search" type="search" placeholder="找演奏者或曲名">'
            '<span id="count">0 段已评</span><button id="export" type="button">下载筛选反馈</button></div>'
            '<section><h2>先听这 25 位</h2><p class="fine">按可用曲目数量排序，先判断最能影响训练集的一批来源。</p><div class="grid">'
            + ''.join(review_card(row, i) for i, row in enumerate(priority, 1)) + '</div></section>'
            '<details id="other"><summary>其他演奏者 · 点击展开</summary><div class="grid">'
            + ''.join(review_card(row, i) for i, row in enumerate(remaining, 26)) + '</div></details>'
            '<section><h2>已知噪声参照</h2><div class="grid">'
            + ''.join(review_card(row, i) for i, row in enumerate(anchors, len(review_rows)-1)) + '</div></section>'
            + ('<details id="second"><summary>可选：同来源换一首复核 · 点击展开</summary>'
               '<p class="fine">这组不阻塞训练，也无需逐首听完。已按你认可的录音来源批量准备更多曲目；这里只在想核对某位演奏者时使用。</p><div class="grid">'
               + ''.join(review_card(row, i) for i, row in enumerate(second_rows, len(review_rows)+1))
               + '</div></details>' if second_rows else '') +
            '<p class="fine">评分保存在当前浏览器；可选试听并下载 JSON。批量训练筛选已经独立进行，无需完成全部卡片。试听片段保留原始电平，没有降噪或美化。</p>'
            '<p><a href="首轮筛选说明.md">查看筛选说明</a> · '
            + ('<a href="反馈分析.md">查看反馈分析</a> · <a href="逐首筛选清单_听感更新.csv">查看听感更新后的逐首状态</a>'
               if second_rows else '<a href="逐首筛选清单.csv">查看 638 首逐首状态</a>') + '</p>'
            '</main><script>' + script + '</script></body></html>')
    (OUTPUT / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CLIPS.mkdir(parents=True, exist_ok=True)
    source_rows = read_csv(WORK.parent / "outputs" / "古琴夜间训练" / "全曲库粗筛清单.csv")
    floor = {row["relative_path"]: row for row in read_csv(WORK / "guqin_noise_floor.csv")}
    all_rows = []
    grouped = defaultdict(list)
    for row in source_rows:
        if row["筛选状态"].startswith("暂不训练"):
            tier = "暂不训练"
        elif strict_candidate(row):
            tier = "技术优先，待耳听"
        else:
            tier = "重点复听，暂缓训练"
        item = dict(row)
        item["首轮层级"] = tier
        item["稳定底声比dB_仅供参考"] = floor[row["Y盘相对路径"]]["stationary_floor_vs_music_db"]
        item["高频谱平坦度_仅供参考"] = floor[row["Y盘相对路径"]]["quiet_high_spectral_flatness"]
        all_rows.append(item)
        if tier != "暂不训练":
            grouped[row["演奏者"]].append(item)
    with (OUTPUT / "逐首筛选清单.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    selections = []
    for artist, tracks in grouped.items():
        strict = [row for row in tracks if row["首轮层级"] == "技术优先，待耳听"]
        pool = strict or tracks
        chosen = min(pool, key=listen_priority)
        selections.append((chosen, len(tracks), len(strict), False))
    selections.sort(key=lambda item: (-item[2], -item[1], item[0]["演奏者"]))
    for artist in ("张育瑾", "谢孝苹"):
        tracks = [row for row in all_rows if row["演奏者"] == artist]
        chosen = min(tracks, key=listen_priority)
        selections.append((chosen, len(tracks), 0, True))
    review_rows = []
    for number, (chosen, count, strict_count, anchor) in enumerate(selections, 1):
        filename = f"{number:03d}.flac"
        target = CLIPS / filename
        if not target.exists():
            make_excerpt(chosen["Y盘相对路径"], target)
        review_rows.append({"演奏者": chosen["演奏者"], "曲名": chosen["曲名"],
                            "Y盘相对路径": chosen["Y盘相对路径"], "试听文件": f"试听片段/{filename}",
                            "该演奏者候选首数": count, "该演奏者技术优先首数": strict_count,
                            "本曲技术优先": chosen["首轮层级"] == "技术优先，待耳听",
                            "已知噪声参照": anchor})
        if number % 20 == 0 or number == len(selections):
            print(f"Prepared {number}/{len(selections)} listening excerpts", flush=True)
    with (OUTPUT / "审听队列.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)
    build_page(review_rows)
    counts = Counter(row["首轮层级"] for row in all_rows)
    hours = {key: round(sum(float(row["时长分钟"]) for row in all_rows if row["首轮层级"] == key) / 60, 2)
             for key in counts}
    summary = {"tracks": len(all_rows), "tiers": dict(counts), "hours": hours,
               "artist_excerpts": len(review_rows)-2, "known_noisy_anchors": 2}
    (WORK / "guqin_recording_screen_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
