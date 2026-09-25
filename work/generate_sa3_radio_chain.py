"""Test whether audio-only guqin continuation stays coherent across chunks."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from stable_audio_3 import StableAudioModel


ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "outputs" / "古琴夜间训练"
SOURCE = ROOT / "guqin_holdout_reference.flac"
RATE = 44100


def metrics(audio):
    mono = audio.mean(axis=0)
    seconds = mono[: len(mono) // RATE * RATE].reshape(-1, RATE)
    rms = np.sqrt(np.mean(seconds.astype(np.float64) ** 2, axis=1))
    return {"mean_rms_dbfs": round(20 * np.log10(max(float(np.sqrt(np.mean(mono.astype(np.float64) ** 2))), 1e-8)), 2),
            "near_silent_seconds_fraction": round(float(np.mean(rms < 0.001)), 3),
            "clipped_fraction": round(float(np.mean(np.abs(mono) >= 0.999)), 5)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small-music")
    parser.add_argument("--adapter")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--chunks", type=int, default=5)
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--level_dbfs", type=float)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    source, rate = torchaudio.load(str(SOURCE))
    context = source[:, 5 * rate : 15 * rate].mean(0, keepdim=True).repeat(2, 1)
    if rate != RATE:
        context = torchaudio.functional.resample(context, rate, RATE)
    context = context.cpu()
    model = StableAudioModel.from_pretrained(args.model, device="cuda")
    if args.adapter:
        model.load_lora([args.adapter])
    assembled = None
    rows = []
    for index in range(args.chunks):
        mask_start = 10 if index == 0 else 8
        seed = 94017 + index
        started = time.monotonic()
        with torch.inference_mode():
            result = model.generate(
                prompt="", duration=args.duration, steps=8, seed=seed,
                sample_size=model.model_config["sample_size"],
                inpaint_audio=(RATE, context), inpaint_mask_start_seconds=mask_start,
                inpaint_mask_end_seconds=args.duration, chunked_decode=True,
            )
        chunk = result[0].detach().float().cpu().numpy()
        generated = chunk[:, mask_start * RATE :]
        raw_signal = metrics(generated)
        gain_db = 0.0
        if args.level_dbfs is not None:
            rms = np.sqrt(np.mean(generated.astype(np.float64) ** 2))
            requested = 10 ** (args.level_dbfs / 20) / max(float(rms), 1e-8)
            peak_limited = 0.98 / max(float(np.max(np.abs(generated))), 1e-8)
            gain = min(requested, peak_limited, 2.0)
            gain_db = round(20 * np.log10(gain), 2)
            fade = min(2 * RATE, generated.shape[-1])
            curve = np.full(generated.shape[-1], gain, dtype=np.float32)
            curve[:fade] = np.linspace(1.0, gain, fade, dtype=np.float32)
            chunk[:, mask_start * RATE :] *= curve[None, :]
            generated = chunk[:, mask_start * RATE :]
        row = {"chunk": index + 1, "seed": seed, "mask_start_seconds": mask_start,
               "generation_seconds": round(time.monotonic() - started, 2),
               "raw_signal": raw_signal, "level_gain_db": gain_db,
               "signal": metrics(generated)}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if assembled is None:
            assembled = chunk
        else:
            assembled = np.concatenate([assembled[:, :-2 * RATE], chunk[:, 8 * RATE :]], axis=1)
        context = torch.from_numpy(assembled[:, -10 * RATE :].copy())
        if row["signal"]["near_silent_seconds_fraction"] > 0.4 or row["signal"]["clipped_fraction"] > 0.01:
            row["stop_reason"] = "signal check failed"
            break
    file = f"电台接续_{args.tag}_{round(assembled.shape[1] / RATE)}秒.flac"
    sf.write(OUT / file, assembled.T, RATE, format="FLAC", subtype="PCM_16")
    seconds = assembled.shape[1] / RATE
    excerpt_starts = [20, max(20, int(seconds // 2) - 10), max(20, int(seconds) - 30)]
    excerpt_files = []
    for number, start in enumerate(excerpt_starts, 1):
        excerpt = assembled[:, start * RATE : (start + 20) * RATE]
        name = f"电台接续_{args.tag}_选段{number}.flac"
        sf.write(OUT / name, excerpt.T, RATE, format="FLAC", subtype="PCM_16")
        excerpt_files.append({"file": name, "start_seconds": start})
    report = {"model": args.model, "adapter": args.adapter, "prompt": "", "level_dbfs": args.level_dbfs,
              "source": str(SOURCE),
              "source_start_seconds": 5, "file": file, "total_seconds": round(seconds, 2),
              "chunks": rows, "excerpts": excerpt_files}
    (OUT / f"电台接续_{args.tag}_记录.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"file": file, "total_seconds": round(seconds, 2)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
