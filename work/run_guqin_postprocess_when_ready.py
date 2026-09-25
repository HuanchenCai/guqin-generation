"""Create the listening page once the overnight training pipeline finishes."""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


WORK = Path(__file__).resolve().parent
ROOT = WORK / "guqin_aligned_412"
STATUS = ROOT / "pipeline_status.json"
POST_STATUS = ROOT / "postprocess_status.json"


def update(phase: str, **details: object) -> None:
    POST_STATUS.write_text(json.dumps({"phase": phase,
        "time": datetime.now().astimezone().isoformat(), **details},
        ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    update("waiting_for_training")
    while True:
        if STATUS.exists():
            state = json.loads(STATUS.read_text(encoding="utf-8"))
            if state["phase"] == "failed":
                update("blocked_by_training_failure", training=state)
                raise SystemExit(2)
            if state["phase"] == "complete":
                break
        time.sleep(30)
    update("generating_comparison")
    with (ROOT / "postprocess.log").open("a", encoding="utf-8") as stream:
        result = subprocess.run([sys.executable, str(WORK / "generate_guqin_piece_comparison.py")],
                                cwd=WORK.parent, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        update("failed", exit_code=result.returncode)
        raise SystemExit(result.returncode)
    update("complete", page=str(WORK.parent / "outputs" / "古琴曲目分组对照" / "index.html"))


if __name__ == "__main__":
    main()
