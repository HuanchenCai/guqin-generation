import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import ASTFeatureExtractor, ASTForAudioClassification


WORK = Path(__file__).parent
OUTPUT = WORK.parent / "outputs" / "古琴新素材筛选"
with (OUTPUT / "候选曲目与年代.csv").open(encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
reel, sr = sf.read(OUTPUT / "较晚录音候选审听.flac", dtype="float32", always_2d=True)

torch.set_num_threads(4)
model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
extractor = ASTFeatureExtractor.from_pretrained(model_id)
model = ASTForAudioClassification.from_pretrained(model_id).eval()
labels = model.config.id2label
wanted = ("speech", "singing", "flute", "drum", "guitar", "zither", "plucked")
ids = [int(i) for i, label in labels.items() if any(term in label.lower() for term in wanted)]

for row in rows:
    start = int((float(row["合辑开始秒"]) + 2.5) * sr)
    x = reel[start : start + 10 * sr].mean(axis=1)
    x = resample_poly(x, 16000, sr).astype(np.float32)
    features = extractor(x, sampling_rate=16000, return_tensors="pt")
    with torch.inference_mode():
        scores = torch.sigmoid(model(**features).logits)[0].numpy()
    ranked = sorted(((labels[i], float(scores[i])) for i in ids), key=lambda pair: pair[1], reverse=True)
    row["AST分类前五_仅辅助判断"] = "; ".join(f"{k}:{v:.3f}" for k, v in ranked[:5])
    row["AST_语音最高分"] = round(max((float(scores[i]) for i in ids if "speech" in labels[i].lower()), default=0), 4)
    row["AST_歌唱最高分"] = round(max((float(scores[i]) for i in ids if "singing" in labels[i].lower()), default=0), 4)
    row["AST_长笛最高分"] = round(max((float(scores[i]) for i in ids if "flute" in labels[i].lower()), default=0), 4)
    row["AST_鼓最高分"] = round(max((float(scores[i]) for i in ids if "drum" in labels[i].lower()), default=0), 4)
    print(row["序号"], row["文件名"], ranked[:5], flush=True)

with (OUTPUT / "候选曲目与年代.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
