"""Prepare the same lower-noise corpus as audio for SAME-L encoding."""

import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "sa3_overnight"
DEST = ROOT / "quieter_audio"


def main():
    audit = json.loads((ROOT / "noise_audit.json").read_text(encoding="utf-8"))
    DEST.mkdir(parents=True, exist_ok=True)
    count = 0
    for group in ("small", "large"):
        for row in audit[group]:
            if row["quiet_highpass_relative_db"] > -35:
                continue
            source = ROOT / f"{group}_audio" / row["file"]
            target = DEST / f"{group}_{row['file']}"
            if not target.exists():
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copy2(source, target)
            target.with_suffix(".txt").write_text("", encoding="utf-8")
            count += 1
    print(f"Prepared {count} clips", flush=True)


if __name__ == "__main__":
    main()
