"""Create one-page human review excerpts for uncertain solo-guqin recordings."""

import json
from pathlib import Path

import soundfile as sf


ROOT = Path(__file__).resolve().parent / "sa3_overnight"
OUT = ROOT.parent.parent / "outputs" / "古琴夜间训练"


def main():
    manifest = json.loads((ROOT / "corpus_manifest.json").read_text(encoding="utf-8"))
    source_by_file = {row.get("file"): row for row in manifest["large_files"] if row.get("file")}
    flagged = json.loads((ROOT / "strict_solo_corpus.json").read_text(encoding="utf-8"))["excluded_large_files"]
    scores = {row["file"]: row for row in json.loads((ROOT / "solo_clap_audit.json").read_text(encoding="utf-8"))}
    rows = []
    for index, file in enumerate(flagged, 1):
        source = ROOT / "large_audio" / file
        with sf.SoundFile(source) as handle:
            start = max(0, (handle.frames - 10 * handle.samplerate) // 2)
            handle.seek(start)
            audio = handle.read(10 * handle.samplerate, dtype="float32", always_2d=True)
            rate = handle.samplerate
        output_name = f"素材复核_{index:02d}.flac"
        sf.write(OUT / output_name, audio, rate, format="FLAC", subtype="PCM_16")
        original = source_by_file[file]
        score = scores[file]
        solo = max(score["solo_guqin"], score["solo_zither"])
        other_keys = ("guqin_flute", "guqin_voice", "ensemble", "bass_drums", "speech")
        predicted = max(other_keys, key=lambda key: score[key])
        rows.append({"file": output_name, "source": original["source"], "performer": original["performer"],
                     "title": original["title"], "model_flag": predicted,
                     "score_margin": round(score[predicted] - solo, 3)})
    (OUT / "素材复核清单.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared {len(rows)} review snippets", flush=True)


if __name__ == "__main__":
    main()
