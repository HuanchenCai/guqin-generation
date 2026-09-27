"""Pick page for showcase v0: three takes per scene, usual page format."""

import html
import json
from pathlib import Path

from build_guqin_inference_page import SCRIPT, STYLES

OUT = Path(__file__).resolve().parent.parent / "outputs" / "古琴展示v0"
ISSUES = ("半音多", "拉锯/金属杂音", "不像古琴", "没旋律", "不符合描述", "音量不稳")


def rating(rid: str) -> str:
    return (f'<div class="rating" data-id="{html.escape(rid, quote=True)}">'
            '<label>综合评分 <select><option value="">未评分</option>'
            + ''.join(f'<option value="{n}">{n}</option>' for n in range(1, 6))
            + '</select></label><div class="issues">'
            + ''.join(f'<label><input type="checkbox" value="{x}">{x}</label>' for x in ISSUES)
            + '</div><input class="comment" placeholder="其他听感；选定展示用的这段请写“入选”"></div>')


def main() -> None:
    records = json.loads((OUT / "样本记录.json").read_text(encoding="utf-8"))
    sections = []
    for scene in dict.fromkeys(r["scene"] for r in records):
        takes = [r for r in records if r["scene"] == scene]
        text = takes[0]["prompt"].split("zither. ", 1)[1]
        cards = [f'<article class="card"><div class="badge">第 {k} 版</div><p>种子 {r["seed"]}</p>'
                 f'<audio controls preload="none" src="audio/{r["file"]}"></audio>{rating(r["file"])}</article>'
                 for k, r in enumerate(takes, 1)]
        sections.append(f'<section class="case"><h2>{html.escape(scene)}</h2>'
                        f'<p class="fine">{html.escape(text)}</p><div class="grid">{"".join(cards)}</div></section>')
    page = ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>古琴展示 v0</title><style>' + STYLES + '</style></head><body>'
            '<header><div><h1>古琴展示 v0</h1>'
            '<p>不给开头，只写一段场景描述，由故事版模型（第 2 遍）生成 35 秒。每个场景三版，挑一版入选。</p></div></header><main>'
            '<div class="intro">写法参照上一轮得 5 分的“秋夜泊舟，月落江寒，怀念远方故人”。五个场景都是新写的，训练里没有这些描述。'
            '每个场景请给入选的那一版在备注里写“入选”。</div>'
            + ''.join(sections)
            + '<button class="export" id="export">下载统一评分</button>'
            '<p class="fine">评分自动保存在当前浏览器；下载 JSON 后可以直接发给我。</p></main>'
            '<script>' + SCRIPT.replace("guqin_inference_sweep_v4", "guqin_showcase_v0")
            .replace("古琴推理调参统一评分.json", "古琴展示v0统一评分.json") + '</script></body></html>')
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
