"""Verify every referenced listening sample is present and playable."""

from html.parser import HTMLParser
from pathlib import Path

import soundfile as sf


OUT = Path(__file__).resolve().parent.parent / "outputs" / "古琴夜间训练"


class AudioSources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []

    def handle_starttag(self, tag, attrs):
        if tag == "audio":
            self.sources.append(dict(attrs).get("src"))


def main():
    parser = AudioSources()
    parser.feed((OUT / "index.html").read_text(encoding="utf-8"))
    for source in parser.sources:
        path = OUT / source
        if not path.is_file():
            raise FileNotFoundError(path)
        info = sf.info(path)
        if info.duration < 5 or info.samplerate != 44100:
            raise RuntimeError(f"Unexpected audio: {path} {info}")
    print(f"Verified {len(parser.sources)} audio players", flush=True)


if __name__ == "__main__":
    main()
