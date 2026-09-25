"""Extend MusicGen in codec-token space, avoiding repeated waveform re-encoding."""

import json
import time
import argparse
import subprocess
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
import imageio_ffmpeg
from transformers import AutoProcessor, MusicgenForConditionalGeneration

ROOT = Path(__file__).parent
SOURCE = ROOT / "guqin_v2" / "audio" / "source_01_part_04.flac"
MODEL = ROOT / "models" / "musicgen-small"
RUN = ROOT / "guqin_token_chain"
OUTPUT = ROOT.parent / "outputs" / "古琴连续生成"
RATE = 32000
STEPS = 6
CONTEXT_TOKENS = 600  # 12 seconds at MusicGen's 50 Hz codec-token rate
LABEL = "A single solo Chinese seven-string guqin, no voice or other instruments."


def rms(x):
    return round(float(torch.sqrt(x.float().square().mean())), 5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("instrument", "audio_only"), default="instrument")
    parser.add_argument("--steps", type=int, default=STEPS)
    args = parser.parse_args()
    run = RUN if args.condition == "instrument" else ROOT / "guqin_token_chain_audio_only"
    run.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(str(MODEL), local_files_only=True)
    model = MusicgenForConditionalGeneration.from_pretrained(
        str(MODEL), local_files_only=True, torch_dtype=torch.float16).to("cuda").eval()
    if (run / "state.json").exists() and (run / "all_codes.pt").exists():
        report = json.loads((run / "state.json").read_text(encoding="utf-8"))
        all_codes = torch.load(run / "all_codes.pt", map_location="cpu", weights_only=True)
    else:
        source, rate = torchaudio.load(str(SOURCE))
        source = source.mean(0, keepdim=True)[:, :10 * rate]
        source = torchaudio.functional.resample(source, rate, RATE)
        with torch.inference_mode():
            initial = model.audio_encoder.encode(source.unsqueeze(0).half().to("cuda"))
        all_codes = initial.audio_codes.detach().cpu().long()
        report = {"source": str(SOURCE), "method": "MusicGen codec-token continuation without intermediate waveform re-encoding",
                  "condition": LABEL if args.condition == "instrument" else "audio only; no text prompt",
                  "sample_rate": RATE, "chunks": []}

    original_decode = model.audio_encoder.decode
    captured = {}

    def capture_decode(audio_codes, *args, **kwargs):
        captured["codes"] = audio_codes.detach().cpu().long()
        return original_decode(audio_codes, *args, **kwargs)

    model.audio_encoder.decode = capture_decode
    for step in range(len(report["chunks"]), args.steps):
        context = all_codes[0, 0, :, -CONTEXT_TOKENS:]
        context_len = context.shape[-1]
        inputs = (processor(text=[LABEL], padding=True, return_tensors="pt").to("cuda")
                  if args.condition == "instrument" else model.get_unconditional_inputs(num_samples=1))
        torch.manual_seed(10301 + step)
        started = time.monotonic()
        captured.clear()
        with torch.inference_mode():
            output = model.generate(**inputs, decoder_input_ids=context.to("cuda"),
                                    do_sample=True, max_new_tokens=950,
                                    **({"guidance_scale": 3} if args.condition == "instrument" else {}))
        generated_codes = captured["codes"]
        got = generated_codes[0, 0]
        agreement = float((got[:, :context_len] == context).float().mean())
        if agreement < 0.99:
            raise RuntimeError(f"Prompt token mismatch at step {step + 1}: {agreement:.4f}")
        new_codes = generated_codes[..., context_len:]
        if new_codes.shape[-1] < 250:
            raise RuntimeError(f"Too little new audio at step {step + 1}: {new_codes.shape}")
        all_codes = torch.cat((all_codes, new_codes), dim=-1)
        waveform = output[0].detach().cpu().float().reshape(1, -1)
        new_waveform = waveform[:, int(context_len / 50 * RATE):]
        row = {"chunk": step + 1, "seed": 10301 + step, "new_tokens": new_codes.shape[-1],
               "new_seconds": round(new_codes.shape[-1] / 50, 2), "rms_new": rms(new_waveform),
               "prompt_token_agreement": round(agreement, 5),
               "elapsed_seconds": round(time.monotonic() - started, 1)}
        report["chunks"].append(row)
        report["total_seconds"] = round(all_codes.shape[-1] / 50, 2)
        torch.save(all_codes, run / "all_codes.pt")
        (run / "state.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    model.audio_encoder.decode = original_decode
    duration_min = round(all_codes.shape[-1] / 50 / 60)
    if args.condition == "instrument":
        filename = "音频符号接续_连续约两分钟.mp3" if args.steps <= 6 else f"音频符号接续_连续约{duration_min}分钟.mp3"
    else:
        filename = "纯音频接续_无文字提示.mp3" if args.steps <= 6 else f"纯音频接续_连续约{duration_min}分钟.mp3"
    path = OUTPUT / filename
    if all_codes.shape[-1] <= 7000:
        with torch.inference_mode():
            decoded = model.audio_encoder.decode(all_codes.to("cuda"), audio_scales=[None]).audio_values
        audio = decoded[0].detach().cpu().float().reshape(1, -1)
        torchaudio.save(str(path), audio, RATE, format="mp3")
        actual_seconds = audio.shape[-1] / RATE
    else:
        wav = run / "decoded.wav"
        total_tokens = all_codes.shape[-1]
        with sf.SoundFile(str(wav), "w", samplerate=RATE, channels=1, subtype="PCM_16") as output_file:
            for start in range(0, total_tokens, 1000):
                end = min(total_tokens, start + 1000)
                left = max(0, start - 200)
                right = min(total_tokens, end + 200)
                with torch.inference_mode():
                    decoded = model.audio_encoder.decode(all_codes[..., left:right].to("cuda"),
                                                         audio_scales=[None]).audio_values
                piece = decoded[0].detach().cpu().float().reshape(-1)
                piece = piece[(start - left) * 640:(end - left) * 640]
                output_file.write(piece.numpy())
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                   "-i", str(wav), "-codec:a", "libmp3lame", "-b:a", "128k", str(path)]
        subprocess.run(command, check=True)
        actual_seconds = sf.info(str(wav)).duration
    if args.condition == "instrument":
        record_name = "音频符号接续记录.json" if args.steps <= 6 else f"音频符号接续_连续约{duration_min}分钟_记录.json"
    else:
        record_name = "纯音频接续记录.json" if args.steps <= 6 else f"纯音频接续_连续约{duration_min}分钟_记录.json"
    (OUTPUT / record_name).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {path} ({actual_seconds:.2f}s)", flush=True)


if __name__ == "__main__":
    main()
