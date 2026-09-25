"""Summarize amplitude drift at different SA3 continuation lengths."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf


OUT = Path(__file__).resolve().parent.parent / "outputs" / "古琴夜间训练"
FILES = {
    35: "未见录音_Medium123段1000步_94017.flac",
    60: "未见录音_长度诊断60秒_94017.flac",
    120: "未见录音_长度诊断120秒_94017.flac",
    300: "电台接续_Medium123段1000步单次_300秒.flac",
}
INTERVALS = [(10, 30), (30, 60), (60, 120), (120, 300)]


def describe(part: np.ndarray) -> dict:
    mono = part.mean(axis=1, dtype=np.float64)
    rms = float(np.sqrt(np.mean(mono**2)))
    return {"mean_rms_dbfs": round(20 * np.log10(max(rms, 1e-8)), 2),
            "clipped_fraction": round(float(np.mean(np.abs(mono) >= 0.999)), 5)}


def main() -> None:
    results = []
    for duration, name in FILES.items():
        audio, rate = sf.read(OUT / name, always_2d=True, dtype="float32")
        if rate != 44100 or len(audio) != duration * rate:
            raise AssertionError(f"Unexpected audio length: {name}")
        rows = []
        for start, end in INTERVALS:
            if start >= duration:
                continue
            stop = min(end, duration)
            rows.append({"start_seconds": start, "end_seconds": stop,
                         **describe(audio[start * rate:stop * rate])})
        results.append({"duration_seconds": duration, "file": name,
                        "source": "held-out Cai Deyun, seconds 5–15 as context",
                        "seed": 94017, "model": "SA3 Medium + 123-clip LoRA, 1000 steps",
                        "generated_only": describe(audio[10 * rate:]), "intervals": rows})
    target = OUT / "续写长度诊断.json"
    target.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
