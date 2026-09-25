"""Train the guqin latent LM on pre-encoded SAME-L windows.

Uses the piece-level split already in guqin_aligned_412 (latents_train vs
latents_validation / latents_test), so test openings are unseen compositions.
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from guqin_latent_lm import LatentLM

ROOT = Path(__file__).resolve().parent / "guqin_aligned_412"


def load_split(name: str) -> list[torch.Tensor]:
    windows = []
    for path in sorted((ROOT / f"latents_{name}").glob("*.npy")):
        if path.name == "silence.npy":
            continue
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        valid = sum(meta["padding_mask"])
        windows.append(torch.from_numpy(np.load(path).astype(np.float32)[:, :valid].T.copy()))
    return windows


def batches(windows, batch, length, generator):
    while True:
        idx = torch.randint(len(windows), (batch,), generator=generator)
        out = []
        for i in idx.tolist():
            w = windows[i]
            start = torch.randint(w.shape[0] - length + 1, (1,), generator=generator).item()
            out.append(w[start:start + length])
        yield torch.stack(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "latent_lm"))
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    train, val = load_split("train"), load_split("validation")
    stacked = torch.cat(train)
    mean, std = stacked.mean(0), stacked.std(0).clamp_min(1e-3)
    torch.save({"mean": mean, "std": std}, out / "stats.pt")
    norm = lambda ws: [(w - mean) / std for w in ws]
    train, val = norm(train), norm(val)
    print(f"train windows {len(train)} frames {stacked.shape[0]}; val windows {len(val)}", flush=True)

    device = "cuda"
    model = LatentLM(d_model=args.d_model, layers=args.layers, heads=args.d_model // 64).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / args.warmup) * (
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps))))
    gen = torch.Generator().manual_seed(args.seed)
    data = batches(train, args.batch, args.length, gen)
    val_gen = torch.Generator().manual_seed(0)
    val_batches = [next(batches(val, args.batch, args.length, val_gen)) for _ in range(8)]
    log = (out / "log.jsonl").open("a", encoding="utf-8")
    started, running = time.time(), []
    for step in range(1, args.steps + 1):
        model.train()
        x = next(data).to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model.loss(x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - 0.999)
        running.append(loss.item())
        if step % 100 == 0:
            rate = step / (time.time() - started)
            print(f"step {step} loss {np.mean(running):.4f} {rate:.2f} it/s", flush=True)
            log.write(json.dumps({"step": step, "train": float(np.mean(running))}) + "\n")
            running = []
        if step % args.eval_every == 0:
            ema.eval()
            torch.manual_seed(0)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                val_loss = np.mean([ema.loss(v.to(device), noise_prob=0).item() for v in val_batches])
            print(f"step {step} val {val_loss:.4f}", flush=True)
            log.write(json.dumps({"step": step, "val": float(val_loss)}) + "\n")
            log.flush()
        if step % args.save_every == 0 or step == args.steps:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "args": vars(args),
                        "step": step}, out / f"step{step:06d}.pt")
    log.close()


if __name__ == "__main__":
    main()
