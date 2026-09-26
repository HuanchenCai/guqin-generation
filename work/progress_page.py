"""Self-refreshing progress page for long background jobs.

Jobs are listed in work/progress_jobs.json; this loop rewrites
outputs/任务进度/index.html every 15 s until every job is finished.

Job kinds:
  count - progress = matches of `pattern` in `log` out of `total`; the
          running item is estimated from the average time per finished item
          (or `item_seconds` before the first one finishes).
  train - progress = last step in a Lightning metrics.csv out of `total`.
"""

import glob
import html
import json
import re
import time
from datetime import datetime
from pathlib import Path

WORK = Path(__file__).resolve().parent
JOBS = WORK / "progress_jobs.json"
OUT = WORK.parent / "outputs" / "任务进度" / "index.html"


def clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}" if seconds >= 3600 \
        else f"{seconds // 60}:{seconds % 60:02d}"


def status(job: dict, now: float) -> dict:
    started = datetime.fromisoformat(job["started"]).timestamp()
    elapsed = now - started
    if job["kind"] == "count":
        text = Path(job["log"]).read_text(encoding="utf-8", errors="replace") if Path(job["log"]).exists() else ""
        done = len(re.findall(job["pattern"], text))
        total = job["total"]
        per_item = elapsed / done if done else job.get("item_seconds", 600)
        partial = 0 if done >= total else min(0.95, (elapsed - done * per_item) / per_item)
        fraction = min(1.0, (done + max(0.0, partial)) / total)
        detail = f"已完成 {done} / {total}"
    else:
        files = glob.glob(job["metrics"])
        step = 0
        if files:
            lines = Path(files[0]).read_text(encoding="utf-8").strip().splitlines()
            step = int(lines[-1].split(",")[0]) + 1 if len(lines) > 1 else 0
        total = job["total"]
        fraction = min(1.0, step / total)
        detail = f"第 {step} / {total} 步"
    finished = fraction >= 1.0 or bool(job.get("done_file") and Path(job["done_file"]).exists())
    fraction = 1.0 if finished else fraction
    remaining = elapsed / fraction - elapsed if 0 < fraction < 1 else 0
    return {"name": job["name"], "note": job.get("note", ""), "fraction": fraction, "detail": detail,
            "elapsed": clock(elapsed), "remaining": "已完成" if finished else f"约剩 {clock(remaining)}",
            "eta": "" if finished else datetime.fromtimestamp(now + remaining).strftime("%H:%M"),
            "finished": finished}


def render(rows: list[dict], now: float) -> str:
    bars = "".join(
        f'<section><div class="head"><strong>{html.escape(r["name"])}</strong>'
        f'<span>{r["fraction"] * 100:.0f}%</span></div>'
        f'<div class="bar"><i style="width:{r["fraction"] * 100:.1f}%"></i></div>'
        f'<p>{html.escape(r["detail"])} · 已用 {r["elapsed"]} · {r["remaining"]}'
        + (f' · 预计 {r["eta"]} 完成' if r["eta"] else "") + "</p>"
        + (f'<p class="note">{html.escape(r["note"])}</p>' if r["note"] else "") + "</section>"
        for r in rows)
    refresh = "" if all(r["finished"] for r in rows) else '<meta http-equiv="refresh" content="15">'
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">' + refresh +
            '<title>任务进度</title><style>'
            ':root{--bg:#eeeae0;--paper:#fffdf8;--ink:#26372e;--muted:#68756b;--line:#d7ded4;--accent:#355c46}'
            '@media(prefers-color-scheme:dark){:root{--bg:#1c201d;--paper:#252a26;--ink:#e6ebe4;--muted:#9aa69c;--line:#3a423c;--accent:#7fb08f}}'
            'body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 "Microsoft YaHei",system-ui,sans-serif}'
            'main{max-width:760px;margin:auto;padding:28px 16px}h1{font:normal 30px "STSong",serif;margin:0 0 4px}'
            'section{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:14px 0}'
            '.head{display:flex;justify-content:space-between;gap:12px}.bar{height:12px;background:var(--line);border-radius:6px;overflow:hidden;margin:8px 0}'
            '.bar i{display:block;height:100%;background:var(--accent)}p{margin:0;color:var(--muted);font-size:13px}.note{margin-top:4px}'
            '</style></head><body><main><h1>任务进度</h1>'
            f'<p>更新于 {datetime.fromtimestamp(now).strftime("%H:%M:%S")}，每 15 秒自动刷新。</p>'
            + bars + '</main></body></html>')


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    while True:
        now = time.time()
        rows = [status(job, now) for job in json.loads(JOBS.read_text(encoding="utf-8"))]
        OUT.write_text(render(rows, now), encoding="utf-8")
        if all(r["finished"] for r in rows):
            break
        time.sleep(15)


if __name__ == "__main__":
    main()
