"""Audio-led, resumable MusicGen continuation probe without mood prompts."""

import json
import time
from pathlib import Path

import torch
import torchaudio
from transformers import AutoProcessor, MusicgenForConditionalGeneration

ROOT = Path(__file__).parent
SOURCE = ROOT / "guqin_v2" / "audio" / "source_01_part_04.flac"
MODEL = ROOT / "models" / "musicgen-small"
RUN = ROOT / "guqin_radio_probe"
OUTPUT = ROOT.parent / "outputs" / "古琴连续生成"
RATE = 32000
STEPS = 6
CONTEXT_SEC = 10
FADE_SEC = 0.05
LABEL = "A single solo Chinese seven-string guqin, no voice or other instruments."


def rms(values):
    return round(float(torch.sqrt(values.float().square().mean())), 5)


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    master = RUN / "master.wav"
    state_path = RUN / "state.json"
    if state_path.exists() and master.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        audio, sr = torchaudio.load(str(master))
        assert sr == RATE
        completed = len(state["chunks"])
    else:
        audio, sr = torchaudio.load(str(SOURCE))
        audio = audio.mean(0, keepdim=True)[:, :CONTEXT_SEC * sr]
        audio = torchaudio.functional.resample(audio, sr, RATE)
        state = {"source": str(SOURCE), "source_seconds": CONTEXT_SEC,
                 "condition": LABEL, "chunks": [], "sample_rate": RATE}
        completed = 0
    if completed < STEPS:
        processor = AutoProcessor.from_pretrained(str(MODEL), local_files_only=True)
        model = MusicgenForConditionalGeneration.from_pretrained(
            str(MODEL), local_files_only=True, torch_dtype=torch.float16).to("cuda").eval()
        for step in range(completed, STEPS):
            prompt = audio[:, -CONTEXT_SEC * RATE:].squeeze(0).numpy()
            seed = 9301 + step
            torch.manual_seed(seed)
            inputs = processor(audio=prompt, sampling_rate=RATE, text=[LABEL],
                               padding=True, return_tensors="pt").to("cuda")
            inputs["input_values"] = inputs["input_values"].half()
            started = time.monotonic()
            with torch.inference_mode():
                values = model.generate(**inputs, do_sample=True, guidance_scale=3,
                                        max_new_tokens=950)
            generated = values[0].detach().cpu().float()
            if generated.ndim == 1:
                generated = generated.unsqueeze(0)
            new = generated[:, CONTEXT_SEC * RATE:]
            if new.shape[-1] < 5 * RATE:
                raise RuntimeError(f"Too little new audio in chunk {step + 1}")
            fade = int(FADE_SEC * RATE)
            before = rms(audio[:, -5 * RATE:])
            after = rms(new)
            mix = torch.linspace(0, 1, fade).unsqueeze(0)
            audio[:, -fade:] = audio[:, -fade:] * (1 - mix) + new[:, :fade] * mix
            audio = torch.cat((audio, new[:, fade:]), dim=-1)
            row = {"chunk": step + 1, "seed": seed, "added_seconds": round((len(new[0]) - fade) / RATE, 2),
                   "rms_before": before, "rms_new": after,
                   "elapsed_seconds": round(time.monotonic() - started, 1)}
            state["chunks"].append(row)
            state["total_seconds"] = round(audio.shape[-1] / RATE, 2)
            torchaudio.save(str(master), audio, RATE)
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(row, ensure_ascii=False), flush=True)
    preview = OUTPUT / "音频接续_连续约两分钟.mp3"
    torchaudio.save(str(preview), audio, RATE, format="mp3")
    (OUTPUT / "生成记录.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {preview} ({audio.shape[-1] / RATE:.2f}s)", flush=True)


if __name__ == "__main__":
    main()
