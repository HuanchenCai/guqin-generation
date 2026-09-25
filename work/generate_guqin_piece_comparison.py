"""Export trained adapters and build a matched local listening comparison."""

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import save_file
from stable_audio_3 import StableAudioModel


WORK = Path(__file__).resolve().parent
ROOT = WORK / "guqin_aligned_412"
OUTPUT = WORK.parent / "outputs" / "古琴曲目分组对照"
AUDIO = OUTPUT / "audio"
SEEDS = (84917, 94017)
VARIANTS = (("A", "pilot_default", "官方默认遮蔽"),
            ("B", "pilot_continuation", "偏重前缀续写"),
            ("C", "full_412", "412 首全量"))


def checkpoint(directory: Path) -> Path:
    found = list(directory.glob("*.ckpt"))
    if not found:
        raise FileNotFoundError(f"No checkpoint in {directory}")
    return max(found, key=lambda path: int(re.search(r"step=(\d+)", path.name).group(1)))


def export(path: Path) -> Path:
    target = path.with_suffix(".safetensors")
    if not target.exists():
        data = torch.load(path, map_location="cpu", weights_only=False)
        weights = {key: value.detach().contiguous().cpu()
                   for key, value in data["state_dict"].items()}
        save_file(weights, str(target),
                  metadata={"lora_config": json.dumps(data["lora_config"])})
    return target


def signal(audio: np.ndarray) -> dict:
    generated = audio[441000:]
    rms = float(np.sqrt(np.mean(generated.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(generated)))
    return {"rms_dbfs": round(20 * np.log10(max(rms, 1e-8)), 2),
            "peak": round(peak, 4),
            "clipped_fraction": round(float(np.mean(np.abs(generated) >= .999)), 5)}


def main() -> None:
    AUDIO.mkdir(parents=True, exist_ok=True)
    contexts = [row for row in json.loads((ROOT / "eval_contexts" / "index.json").read_text(
        encoding="utf-8")) if row["分组"] == "validation"]
    for source in contexts:
        origin = ROOT / "eval_contexts" / source["file"]
        destination = AUDIO / source["file"]
        if not destination.exists():
            os.link(origin, destination)
    rows = []
    for code, directory, title in VARIANTS:
        ckpt = checkpoint(ROOT / directory)
        adapter = export(ckpt)
        print(f"Loading {code}: {ckpt.name}", flush=True)
        model = StableAudioModel.from_pretrained("medium", device="cuda")
        model.load_lora([str(adapter)])
        model.set_lora_strength(1.0)
        for index, source in enumerate(contexts, 1):
            context, rate = sf.read(ROOT / "eval_contexts" / source["file"],
                                    dtype="float32", always_2d=True)
            prefix = torch.from_numpy(context.T.copy())
            for seed in SEEDS:
                name = f"v{index:02d}_{seed}_{code}.flac"
                target = AUDIO / name
                started = time.monotonic()
                if not target.exists():
                    with torch.inference_mode():
                        result = model.generate(
                            prompt="", duration=35, steps=8, seed=seed,
                            inpaint_audio=(rate, prefix), inpaint_mask_start_seconds=10,
                            inpaint_mask_end_seconds=35, chunked_decode=True,
                        )
                    audio = result[0].detach().float().cpu().T.numpy()
                    sf.write(target, audio, rate, format="FLAC", subtype="PCM_16")
                audio, _ = sf.read(target, dtype="float32", always_2d=True)
                record = {"variant": code, "training": title, "source": source["file"],
                          "piece": source["归一曲目"], "artist": source["演奏者"],
                          "seed": seed, "file": name, "checkpoint": ckpt.name,
                          "generated_seconds": round(time.monotonic() - started, 2),
                          "signal": signal(audio)}
                rows.append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
        del model
        torch.cuda.empty_cache()
    (OUTPUT / "样本记录.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    build_page(rows, contexts)
    print(f"Ready: {OUTPUT / 'index.html'}", flush=True)


def build_page(rows: list[dict], contexts: list[dict]) -> None:
    payload = json.dumps({"rows": rows, "contexts": contexts}, ensure_ascii=False).replace("</", "<\\/")
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>古琴：按曲目分组的续写对照</title>
<style>body{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#f5f1e9;color:#302b23;max-width:1100px;margin:auto;padding:28px}h1{font-size:28px}p{line-height:1.7}.note{background:#fff8e8;border-left:4px solid #bc8b4c;padding:12px 16px;margin:20px 0}.set{background:#fff;border:1px solid #ddd0b8;border-radius:16px;margin:18px 0;padding:20px;box-shadow:0 2px 9px #0001}.set h2{margin:0 0 8px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}.clip{background:#faf8f3;border:1px solid #e8ddcb;border-radius:10px;padding:12px}audio{width:100%;margin-top:5px}select,button{font:inherit;padding:7px 9px}label{display:block;margin:9px 0}.toolbar{position:sticky;top:0;background:#f5f1e9eF;padding:8px 0;z-index:2;display:flex;gap:9px;align-items:center;flex-wrap:wrap}small{color:#655d52}</style>
<h1>古琴续写 · 曲目分组对照</h1><p>同一曲目的录音不会同时出现在训练、验证或测试组。下面固定 10 秒原曲开头，让模型接 25 秒；各候选使用同一个开头、种子和生成设置。每段只需给一个总体分数，听到明显故障可勾选。</p>
<div class="note">A、B 是只用 329 首训练的可比对照，这里的曲目来自未参与训练的验证组。C 是随后用全部 412 首训练的模型，因此 C 对这些曲目属于已见数据，只供检查最终音色，不能当作未见曲目成绩。</div>
<div class="toolbar"><button id="export">导出评分 JSON</button><button id="reveal">显示模型对应关系</button><span id="progress"></span></div><div id="mount"></div>
<script>const DATA=__DATA__;const KEY='guqin_piece_comparison_v1';const ratings=JSON.parse(localStorage.getItem(KEY)||'{}');const save=()=>{localStorage.setItem(KEY,JSON.stringify(ratings));document.getElementById('progress').textContent=`已评分 ${Object.values(ratings).filter(x=>x.score).length}/${DATA.rows.length} 段`};const mount=document.getElementById('mount');for(const [i,source] of DATA.contexts.entries()){if(source['分组']!=='validation')continue;for(const seed of [84917,94017]){const cards=DATA.rows.filter(r=>r.source===source.file&&r.seed===seed);const box=document.createElement('section');box.className='set';box.innerHTML=`<h2>${source['归一曲目']} · 种子 ${seed}</h2><small>${source['演奏者']}演奏的原曲开头；候选前 10 秒均为同一开头</small><label>原曲 10 秒 <audio controls preload="none" src="audio/${source.file}"></audio></label><div class="grid"></div>`;const grid=box.querySelector('.grid');for(const row of cards){const id=`${row.source}|${row.seed}|${row.variant}`;const entry=ratings[id]||{};const card=document.createElement('div');card.className='clip';card.innerHTML=`<strong>候选 ${row.variant}</strong><audio controls preload="none" src="audio/${row.file}"></audio><label>总体评分 <select><option value="">待评分</option>${[1,2,3,4,5].map(n=>`<option value="${n}">${n} / 5</option>`).join('')}</select></label><label><input type="checkbox"> 明显杂音、机械音或重复</label>`;const sel=card.querySelector('select'),check=card.querySelector('input');sel.value=entry.score||'';check.checked=!!entry.fault;sel.onchange=()=>{ratings[id]={score:Number(sel.value)||null,fault:check.checked};save()};check.onchange=sel.onchange;grid.append(card)}mount.append(box)}}document.getElementById('export').onclick=()=>{const blob=new Blob([JSON.stringify({schema:'guqin-piece-comparison-v1',ratings,generated_at:new Date().toISOString()},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='古琴曲目分组统一评分.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};document.getElementById('reveal').onclick=()=>alert('A：官方默认遮蔽（随机/全遮蔽/续写 10/80/10）\\nB：偏重前缀续写（10/30/60）\\nC：全部 412 首训练，采用 10/30/60');save();</script></html>'''.replace('__DATA__', payload)
    (OUTPUT / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
