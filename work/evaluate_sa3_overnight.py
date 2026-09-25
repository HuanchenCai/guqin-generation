"""Generate matched, audio-only guqin continuations for listening tests."""

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
SOURCES = {
    "较新录音": (ROOT / "guqin_v2" / "audio" / "source_01_part_04.flac", 4),
    "未见录音": (ROOT / "guqin_holdout_reference.flac", 5),
}


def prepare_context(path: Path, start: int):
    audio, rate = torchaudio.load(str(path))
    audio = audio[:, start * rate : (start + 10) * rate]
    audio = audio.mean(dim=0, keepdim=True).repeat(2, 1)
    if rate != 44100:
        audio = torchaudio.functional.resample(audio, rate, 44100)
        rate = 44100
    return audio, rate


def describe(audio: np.ndarray, rate: int):
    mono = audio.mean(axis=1)
    part = mono[10 * rate :]
    blocks = part[: len(part) // rate * rate].reshape(-1, rate)
    rms = np.sqrt(np.mean(blocks.astype(np.float64) ** 2, axis=1))
    return {
        "seconds": round(len(mono) / rate, 2),
        "mean_rms_dbfs": round(20 * np.log10(max(float(np.sqrt(np.mean(part.astype(np.float64) ** 2))), 1e-8)), 2),
        "quiet_second_fraction": round(float(np.mean(rms < 0.001)), 3),
        "clipped_fraction": round(float(np.mean(np.abs(part) >= 0.999)), 5),
        "peak": round(float(np.max(np.abs(part))), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small-music")
    parser.add_argument("--adapter")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--duration", type=int, default=35)
    parser.add_argument("--sources", nargs="+", default=list(SOURCES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[84917, 94017])
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    model = StableAudioModel.from_pretrained(args.model, device="cuda")
    if args.adapter:
        model.load_lora([args.adapter])
        model.set_lora_strength(args.strength)
    rows = []
    for source_name in args.sources:
        path, start = SOURCES[source_name]
        context, rate = prepare_context(path, start)
        for seed in args.seeds:
            begin = time.monotonic()
            with torch.inference_mode():
                result = model.generate(
                    prompt="", duration=args.duration, steps=args.steps, seed=seed,
                    inpaint_audio=(rate, context), inpaint_mask_start_seconds=10,
                    inpaint_mask_end_seconds=args.duration, chunked_decode=True,
                )
            audio = result[0].detach().float().cpu().T.numpy()
            file = f"{source_name}_{args.tag}_{seed}.flac"
            sf.write(OUT / file, audio, rate, format="FLAC", subtype="PCM_16")
            row = {"file": file, "source": source_name, "source_path": str(path),
                   "source_start_seconds": start, "context_seconds": 10,
                   "generated_seconds": args.duration - 10, "model": args.model,
                   "adapter": args.adapter, "adapter_strength": args.strength,
                   "seed": seed, "steps": args.steps,
                   "prompt": "", "generation_seconds": round(time.monotonic() - begin, 2),
                   "signal": describe(audio, rate)}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    (OUT / f"{args.tag}_记录.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
