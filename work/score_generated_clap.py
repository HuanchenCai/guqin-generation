"""Compare generated clips to the solo-guqin source distribution using CLAP."""

import csv
from pathlib import Path

import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

from audit_solo_clap import LABELS, MODEL_ID, ROOT, load_ten_seconds


OUTPUT = ROOT.parent / "outputs" / "古琴单乐器试验"


def main() -> None:
    processor = ClapProcessor.from_pretrained(MODEL_ID)
    model = ClapModel.from_pretrained(MODEL_ID).to("cuda").eval()
    text = processor(text=list(LABELS.values()), return_tensors="pt", padding=True)
    with torch.inference_mode():
        text_features = torch.nn.functional.normalize(
            model.get_text_features(**{k: v.cuda() for k, v in text.items()}), dim=-1
        )
    rows = []
    paths = sorted((ROOT.parent / "outputs" / "古琴试听对比").glob("*.flac"))
    paths += sorted(OUTPUT.glob("*.flac"))
    paths += sorted((ROOT.parent / "outputs" / "古琴第二版试听").glob("*.flac"))
    paths += sorted((ROOT.parent / "outputs" / "古琴SFT试听").glob("*.flac"))
    for path in paths:
        audio = load_ten_seconds(path)
        inputs = processor(audio=[audio], sampling_rate=48000, return_tensors="pt")
        with torch.inference_mode():
            features = torch.nn.functional.normalize(
                model.get_audio_features(**{k: v.cuda() for k, v in inputs.items()}), dim=-1
            )
            values = (features @ text_features.T).squeeze(0).cpu().numpy()
        row = {"clip": path.name, **dict(zip(LABELS.keys(), np.round(values, 4)))}
        rows.append(row)
        print(row, flush=True)
    with (OUTPUT / "声音标签模型对比.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
