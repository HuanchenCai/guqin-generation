"""Generate listening samples from the guqin latent LM.

1. Continuations: 6 test-split openings (unseen compositions, distinct pieces
   and performers), 10 s prompt + 50 s generated. For reference, the SA3
   pilot adapter trained on the same 329-piece split continues the same
   openings.
2. Endless play: one 5-minute run per temperature with a sliding context.
3. Free start: 60 s with no opening at all.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from guqin_latent_lm import LatentLM

WORK = Path(__file__).resolve().parent
ROOT = WORK / "guqin_aligned_412"
FPS = 44100 / 4096
PROMPT = int(round(10 * FPS))


def best_checkpoint(run: Path) -> Path:
    vals = {}
    for line in (run / "log.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "val" in row:
            vals[row["step"]] = row["val"]
    saved = {int(p.stem[4:]): p for p in run.glob("step*.pt")}
    step = min((s for s in saved if s in vals), key=vals.get)
    print(f"best checkpoint step {step} val {vals[step]:.4f}", flush=True)
    return saved[step]


def pick_openings(n=6) -> list[dict]:
    with (ROOT / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["split"] == "test"]
    encoded = {json.loads(p.read_text(encoding="utf-8"))["relpath"]: p.with_suffix(".npy")
               for p in (ROOT / "latents_test").glob("*.json")}
    chosen, artists, pieces = [], set(), set()
    for r in sorted(rows, key=lambda r: (r["artist"], r["file"])):
        key = r["piece"].rstrip("0123456789")
        if r["file"] in encoded and r["artist"] not in artists and key not in pieces and r["file"].endswith("_002.flac"):
            chosen.append({**r, "latent": str(encoded[r["file"]])})
            artists.add(r["artist"])
            pieces.add(key)
        if len(chosen) == n:
            break
    return chosen


@torch.no_grad()
def generate(model, prompt, frames, temperature=1.0, sigma=0.1, window=512, steps=32):
    """prompt: (B, P, 256) normalised; returns (B, P+frames, 256)."""
    seq = prompt.clone()
    for _ in range(frames):
        ctx = seq[:, -(window - 1):]
        noisy = ctx + sigma * torch.randn_like(ctx)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.hidden(noisy, torch.full((seq.shape[0],), sigma, device=seq.device))
        nxt = model.sample_next(h[:, -1].float(), steps=steps, temperature=temperature)
        seq = torch.cat([seq, nxt[:, None]], dim=1)
    return seq


def main() -> None:
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "latent_lm_d512"
    out = WORK.parent / "outputs" / "古琴潜空间续写"
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = best_checkpoint(run)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt["args"]
    model = LatentLM(d_model=args["d_model"], layers=args["layers"], heads=args["d_model"] // 64).cuda()
    model.load_state_dict(ckpt["ema"])
    model.eval()
    stats = torch.load(run / "stats.pt")
    mean, std = stats["mean"].cuda(), stats["std"].cuda()

    from stable_audio_3.model import AutoencoderModel
    ae = AutoencoderModel.from_pretrained("same-l", device="cuda")

    def write(name, latents):
        z = (latents * std + mean).T[None].float()
        audio = ae.decode(z, chunked=True)[0].float().cpu().T.numpy()
        sf.write(audio_dir / name, audio, 44100, format="FLAC", subtype="PCM_16")
        rms = 20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-9)
        return {"file": name, "seconds": round(len(audio) / 44100, 1), "rms_dbfs": round(float(rms), 1),
                "clipped_percent": round(float(np.mean(np.abs(audio) >= 0.999) * 100), 3)}

    records = {"checkpoint": ckpt_path.name, "continuations": [], "endless": [], "free": []}
    openings = pick_openings()
    prompts = []
    for i, o in enumerate(openings, 1):
        z = torch.from_numpy(np.load(o["latent"]).astype(np.float32).T).cuda()
        prompt = (z[PROMPT:2 * PROMPT] - mean) / std   # 10-20 s of the window
        prompts.append(prompt)
        records["continuations"].append({**write(f"open_{i:02d}.flac", prompt), "piece": o["piece"],
                                         "artist": o["artist"], "source": o["source"]})
    torch.manual_seed(2026)
    seq = generate(model, torch.stack(prompts), int(50 * FPS))
    for i, s in enumerate(seq, 1):
        records["continuations"][i - 1]["latent_lm"] = write(f"c{i:02d}_latent_lm.flac", s)
    for temperature in (0.8, 1.0):
        torch.manual_seed(7)
        seq = generate(model, prompts[0][None], int(300 * FPS), temperature=temperature)
        records["endless"].append({"temperature": temperature, **write(f"endless_t{temperature}.flac", seq[0])})
    torch.manual_seed(11)
    free = generate(model, torch.zeros(3, 0, 256, device="cuda"), int(60 * FPS))
    for i, s in enumerate(free, 1):
        records["free"].append(write(f"free_{i}.flac", s))
    (out / "样本记录.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(records, ensure_ascii=False, indent=1), flush=True)

    # Reference: SA3 pilot adapter trained on the same 329-piece split.
    del ae
    torch.cuda.empty_cache()
    from stable_audio_3 import StableAudioModel
    sa3 = StableAudioModel.from_pretrained("medium", device="cuda")
    sa3.load_lora([str(ROOT / "pilot_continuation" / "epoch=1-step=2217.safetensors")])
    sa3.set_lora_strength(1.0)
    for i, rec in enumerate(records["continuations"], 1):
        context, rate = sf.read(audio_dir / rec["file"], dtype="float32", always_2d=True)
        with torch.inference_mode():
            result = sa3.generate(prompt="", duration=60, steps=8, seed=2026,
                                  inpaint_audio=(rate, torch.from_numpy(context.T.copy())),
                                  inpaint_mask_start_seconds=10, inpaint_mask_end_seconds=60,
                                  chunked_decode=True)
        audio = result[0].detach().float().cpu().T.numpy()
        sf.write(audio_dir / f"c{i:02d}_sa3.flac", audio, rate, format="FLAC", subtype="PCM_16")
        rec["sa3"] = {"file": f"c{i:02d}_sa3.flac"}
    (out / "样本记录.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
