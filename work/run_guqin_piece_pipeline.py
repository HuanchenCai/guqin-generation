"""Finish aligned encoding and run matched piece-level Stable Audio 3 pilots."""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


WORK = Path(__file__).resolve().parent
ROOT = WORK / "guqin_aligned_412"
PYTHON = WORK / "stable-audio-3" / ".venv" / "Scripts" / "python.exe"
STATUS = ROOT / "pipeline_status.json"


def update(phase: str, **details: object) -> None:
    payload = {"phase": phase, "time": datetime.now().astimezone().isoformat(), **details}
    STATUS.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(payload, flush=True)


def run(phase: str, command: list[str], log_name: str) -> None:
    update(phase, command=command)
    log = ROOT / log_name
    with log.open("a", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=WORK.parent, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        update("failed", failed_phase=phase, exit_code=result.returncode, log=str(log))
        raise SystemExit(result.returncode)


def main() -> None:
    first = ROOT / "latents_batch_1"
    second = ROOT / "latents_batch_2"
    update("waiting_for_first_encoding", encoded=len(list(first.glob("*.json"))), expected=2143)
    last_count, last_change = -1, time.monotonic()
    while True:
        count = len(list(first.glob("*.json")))
        if count != last_count:
            last_count, last_change = count, time.monotonic()
        if count == 2143 and "Done" in (ROOT / "encode_batch_1.log").read_text(
                encoding="utf-8", errors="replace")[-5000:]:
            break
        if time.monotonic() - last_change > 900:
            update("failed", failed_phase="first_encoding_stalled", encoded=count)
            raise SystemExit(2)
        time.sleep(30)

    if len(list(second.glob("*.json"))) != 621:
        run("encoding_second_batch", [str(PYTHON),
            str(WORK / "stable-audio-3" / "scripts" / "pre_encode_dataset.py"),
            "--model", "same-l", "--data_dir", str(ROOT / "encode_batch_2" / "audio"),
            "--output_path", str(second), "--sample_size", "2641920",
            "--batch_size", "1", "--model_half", "--chunked"], "encode_batch_2.log")

    run("merging_latents", [str(PYTHON), str(WORK / "merge_guqin_latents.py")], "merge.log")
    run("splitting_pilot_latents", [str(PYTHON), str(WORK / "split_guqin_latents.py")], "split.log")

    pilots = (("default", ("0.1", "0.8", "0.1")),
              ("continuation", ("0.1", "0.3", "0.6")))
    for name, mask in pilots:
        destination = ROOT / f"pilot_{name}"
        checkpoints = list(destination.glob("*.ckpt"))
        if checkpoints:
            update(f"pilot_{name}_already_complete", checkpoint=str(checkpoints[-1]))
            continue
        run(f"training_pilot_{name}", [str(PYTHON), str(WORK / "train_guqin_lora.py"),
            "--model", "medium-base", "--encoded_dir", str(ROOT / "latents_train"),
            "--save_dir", str(destination), "--name", f"guqin-piece-{name}",
            "--duration", "60", "--steps", "2217", "--batch_size", "1",
            "--rank", "16", "--adapter_type", "dora-rows", "--base_precision", "bf16",
            "--lr", "0.0001", "--seed", "42", "--checkpoint_every", "2217",
            "--log_every", "100", "--demo_every", "0", "--num_workers", "0",
            "--mask_probs", *mask], f"pilot_{name}.log")
    update("pilots_complete")

    # The held-out groups are only for model-choice pilots. The user's final
    # model must see all 412 approved complete recordings, including their
    # formerly held-out compositions.
    destination = ROOT / "full_412"
    if not list(destination.glob("*step=2764*.ckpt")):
        run("training_full_412", [str(PYTHON), str(WORK / "train_guqin_lora.py"),
            "--model", "medium-base", "--encoded_dir", str(ROOT / "latents"),
            "--save_dir", str(destination), "--name", "guqin-all-412-continuation",
            "--duration", "60", "--steps", "2764", "--batch_size", "1",
            "--rank", "16", "--adapter_type", "dora-rows", "--base_precision", "bf16",
            "--lr", "0.0001", "--seed", "42", "--checkpoint_every", "1382",
            "--log_every", "100", "--demo_every", "0", "--num_workers", "0",
            "--mask_probs", "0.1", "0.3", "0.6"], "full_412.log")
    update("complete", source_recordings=412, aligned_windows=2764)


if __name__ == "__main__":
    main()
