"""Small audio-only MusicGen LoRA experiment for solo-guqin continuation."""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
import torchaudio
from peft import LoraConfig, get_peft_model
from transformers import MusicgenForConditionalGeneration


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "musicgen-small"
AUDIO_PATH = ROOT / "guqin_v2" / "audio"
TOKEN_PATH = ROOT / "musicgen_guqin_tokens"
RUN_PATH = ROOT / "musicgen_guqin_lora"
RATE = 32000
SEGMENT_SECONDS = 15


def make_tokens():
    TOKEN_PATH.mkdir(parents=True, exist_ok=True)
    model = MusicgenForConditionalGeneration.from_pretrained(
        str(MODEL_PATH), local_files_only=True, torch_dtype=torch.float16
    ).to("cuda").eval()
    rows = []
    for path in sorted(AUDIO_PATH.glob("*.flac")):
        waveform, rate = torchaudio.load(str(path))
        waveform = waveform.mean(0, keepdim=True)
        if rate != RATE:
            waveform = torchaudio.functional.resample(waveform, rate, RATE)
        for part in range(3):
            clip = waveform[:, part * SEGMENT_SECONDS * RATE : (part + 1) * SEGMENT_SECONDS * RATE]
            if clip.shape[-1] < 10 * RATE:
                continue
            filename = f"{path.stem}_{part:02}.pt"
            target = TOKEN_PATH / filename
            if not target.exists():
                with torch.inference_mode():
                    codes = model.audio_encoder.encode(clip[None].half().to("cuda")).audio_codes
                if codes.shape[0] != 1 or codes.shape[1] != 1:
                    raise RuntimeError(f"Unexpected code shape: {codes.shape}")
                torch.save(codes[0, 0].cpu().short(), target)
            rows.append({"file": filename, "source": path.stem.split("_part_")[0]})
        print(f"Encoded {path.name}", flush=True)
    (TOKEN_PATH / "index.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} training segments", flush=True)


def train(steps, window, lr):
    torch.manual_seed(4107)
    random.seed(4107)
    RUN_PATH.mkdir(parents=True, exist_ok=True)
    entries = json.loads((TOKEN_PATH / "index.json").read_text(encoding="utf-8"))
    # All ten confirmed solo recordings train; the external reference remains unseen.
    samples = [(row, torch.load(TOKEN_PATH / row["file"], map_location="cpu", weights_only=True).long())
               for row in entries]
    model = MusicgenForConditionalGeneration.from_pretrained(
        str(MODEL_PATH), local_files_only=True, torch_dtype=torch.bfloat16
    ).to("cuda")
    model.config.decoder.decoder_start_token_id = model.generation_config.decoder_start_token_id
    model.config.decoder.pad_token_id = model.generation_config.pad_token_id
    model.requires_grad_(False)
    target = r"decoder\.model\.decoder\.layers\.\d+\.self_attn\.(q_proj|v_proj)"
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                           target_modules=target, bias="none"))
    model.train()
    model.text_encoder.eval()
    model.audio_encoder.eval()
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=0.01)
    history = []
    started = time.monotonic()
    order = list(range(len(samples)))
    random.shuffle(order)
    for step in range(steps):
        if step and step % len(order) == 0:
            random.shuffle(order)
        row, codes = samples[order[step % len(order)]]
        offset = random.randrange(max(1, codes.shape[-1] - window))
        labels = codes[:, offset : offset + window].T.contiguous()[None].to("cuda")
        # Zero text condition makes the training target pure audio continuation.
        hidden = torch.zeros((1, 1, model.base_model.model.config.text_encoder.hidden_size),
                             device="cuda", dtype=torch.bfloat16)
        mask = torch.zeros((1, 1), device="cuda", dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        output = model(encoder_outputs=(hidden,), attention_mask=mask, labels=labels, use_cache=False)
        loss = output.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        optimizer.step()
        record = {"step": step + 1, "loss": round(float(loss.detach()), 4), "source": row["source"]}
        history.append(record)
        if step == 0 or (step + 1) % 10 == 0:
            print(json.dumps(record), flush=True)
        if (step + 1) % 40 == 0 or step + 1 == steps:
            model.save_pretrained(str(RUN_PATH / "adapter"))
            (RUN_PATH / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {"steps": steps, "window_tokens": window, "learning_rate": lr,
               "seconds": round(time.monotonic() - started, 1),
               "initial_mean_loss": round(sum(x["loss"] for x in history[:20]) / min(20, len(history)), 4),
               "final_mean_loss": round(sum(x["loss"] for x in history[-20:]) / min(20, len(history)), 4),
               "training_segments": len(samples), "audio_condition": "unconditional/no text",
               "base_model": "facebook/musicgen-small"}
    (RUN_PATH / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("tokens", "train"))
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--window", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.0001)
    args = parser.parse_args()
    make_tokens() if args.mode == "tokens" else train(args.steps, args.window, args.lr)
