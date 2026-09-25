from pathlib import Path
from pypdf import PdfReader


ROOT = Path(r"Y:\Music\古琴曲")
OUT = Path(__file__).parent

for name in ("目录.pdf", "绝响国鹏辑近世琴人音像遗珍.pdf"):
    source = ROOT / name
    reader = PdfReader(source)
    rows = []
    for number, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        rows.append(f"\n=== PAGE {number} ===\n{text}\n")
    target = OUT / (source.stem + "_extracted.txt")
    target.write_text("".join(rows), encoding="utf-8")
    print(f"{name}: pages={len(reader.pages)}, extracted_chars={sum(map(len, rows))}, output={target}")
