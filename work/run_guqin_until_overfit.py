"""Train pass by pass and stop once held-out validation loss rises.

Each pass trains one epoch over the train+test windows (story captions,
random start and length), resuming from the previous pass's LoRA weights,
then scores all validation windows. Stops after the loss has risen for
`--patience` passes in a row (or at `--max_passes`) and reports the best.
Note: each pass restarts AdamW moments; the learning rate is constant.
"""

import argparse
import json
import subprocess
from pathlib import Path

WORK = Path(__file__).resolve().parent
ROOT = WORK / "guqin_aligned_412"
PYTHON = WORK / "stable-audio-3" / ".venv" / "Scripts" / "python.exe"
COMMON = ["--model", "medium-base", "--duration", "60", "--batch_size", "1", "--rank", "16",
          "--adapter_type", "dora-rows", "--base_precision", "bf16", "--lr", "0.0001",
          "--num_workers", "0", "--demo_every", "0", "--mask_probs", "0.1", "0.3", "0.6"]


def run(cmd: list[str], log: Path) -> str:
    with log.open("a", encoding="utf-8") as stream:
        result = subprocess.run([str(PYTHON), str(WORK / "train_guqin_lora.py"), *cmd], cwd=WORK,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace")
        stream.write(result.stdout)
    if result.returncode:
        raise SystemExit(f"train_guqin_lora.py failed ({result.returncode}); see {log}")
    return result.stdout


def validate(checkpoint: Path | None, args, log: Path) -> float:
    cmd = [*COMMON, "--seed", "42", "--steps", "1", "--encoded_dir", str(args.train_dir),
           "--save_dir", str(args.out / "val_tmp"), "--val_dir", str(args.val_dir), "--val_batches", str(args.val_batches)]
    if checkpoint:
        cmd += ["--lora_checkpoint", str(checkpoint)]
    line = next(l for l in run(cmd, log).splitlines() if l.startswith("VALIDATION "))
    return json.loads(line[len("VALIDATION "):])["val/avg_loss"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=Path, default=ROOT / "latents_story_train")
    parser.add_argument("--val_dir", type=Path, default=ROOT / "latents_story_val")
    parser.add_argument("--out", type=Path, default=ROOT / "story_run")
    parser.add_argument("--steps_per_pass", type=int, default=2479)
    parser.add_argument("--val_batches", type=int, default=276)
    parser.add_argument("--min_seconds", type=float, default=20)
    parser.add_argument("--max_passes", type=int, default=15)
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    history_path = args.out / "history.jsonl"
    history = [json.loads(l) for l in history_path.read_text(encoding="utf-8").splitlines()] if history_path.exists() else []
    log = args.out / "run.log"
    if not history:
        base = validate(None, args, log)
        history.append({"pass": 0, "val": base, "checkpoint": None})
        history_path.write_text(json.dumps(history[-1]) + "\n", encoding="utf-8")
        print(f"PASS_DONE 0 val {base:.5f} (base model)", flush=True)
    while history[-1]["pass"] < args.max_passes:
        n = history[-1]["pass"] + 1
        save = args.out / f"pass{n:02d}"
        cmd = [*COMMON, "--seed", str(42 + n), "--steps", str(args.steps_per_pass),
               "--checkpoint_every", str(args.steps_per_pass), "--log_every", "100",
               "--min_seconds", str(args.min_seconds), "--encoded_dir", str(args.train_dir),
               "--save_dir", str(save), "--name", f"guqin-story-pass{n}"]
        if history[-1]["checkpoint"]:
            cmd += ["--lora_checkpoint", history[-1]["checkpoint"]]
        run(cmd, log)
        checkpoint = max(save.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
        val = validate(checkpoint, args, log)
        history.append({"pass": n, "val": val, "checkpoint": str(checkpoint)})
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(history[-1]) + "\n")
        print(f"PASS_DONE {n} val {val:.5f}", flush=True)
        best = min(h["val"] for h in history)
        rises = 0
        for prev, cur in zip(history, history[1:]):
            rises = rises + 1 if cur["val"] > prev["val"] else 0
        if rises >= args.patience:
            break
    best = min(history, key=lambda h: h["val"])
    (args.out / "best.json").write_text(json.dumps(best, indent=1), encoding="utf-8")
    print(f"STOPPED after pass {history[-1]['pass']}; best pass {best['pass']} val {best['val']:.5f}", flush=True)


if __name__ == "__main__":
    main()
