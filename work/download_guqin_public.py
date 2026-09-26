"""Download openly licensed guqin recordings into a review-only staging folder.

Nothing here enters training until the user has listened and approved it.
Sources were checked on 2026-09-26: Wikimedia Commons Category:Guqin_music
(Charlie Huang; CC BY / CC BY-SA / public domain) and two licensed Internet
Archive items. Items without an explicit open license are not downloaded.
"""

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

WORK = Path(__file__).resolve().parent
STAGE = WORK / "guqin_public_candidates"
MANIFEST = WORK / "guqin_public_candidates.json"
UA = {"User-Agent": "guqin-research/0.1 (huanchen@kth.se)"}
IA_FILES = [
    ("QiuFengCi", "QiuFengCi.flac"),
    ("Lost_Sounds_of_the_Tao-3748", "Lo_Ka_Ping_-_01_-_Teals_Descending_Upon_The_Level_Sand.ogg"),
]


def get_json(url: str) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def download(url: str, target: Path) -> None:
    """Polite download: pause between files and back off on HTTP 429."""
    if target.exists():
        return
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                target.write_bytes(r.read())
            time.sleep(3)
            return
        except urllib.error.HTTPError as error:
            if error.code != 429:
                raise
            time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"Rate limited too long: {url}")


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text or "")).strip()


def main() -> None:
    STAGE.mkdir(exist_ok=True)
    rows = []
    api = ("https://commons.wikimedia.org/w/api.php?action=query&generator=categorymembers"
           "&gcmtitle=Category:Guqin_music&gcmtype=file&gcmlimit=200&prop=imageinfo"
           "&iiprop=url|size|extmetadata|mediatype&format=json")
    for page in get_json(api)["query"]["pages"].values():
        info = page["imageinfo"][0]
        if info.get("mediatype") != "AUDIO":
            continue
        meta = info.get("extmetadata", {})
        license_name = clean(meta.get("LicenseShortName", {}).get("value"))
        if not re.search(r"public domain|CC BY|CC0", license_name, re.I):
            continue
        name = "commons_" + page["title"][5:].replace(" ", "_")
        download(info["url"], STAGE / name)
        rows.append({"file": name, "source": "Wikimedia Commons", "url": info["descriptionurl"],
                     "license": license_name, "license_url": clean(meta.get("LicenseUrl", {}).get("value")),
                     "author": clean(meta.get("Artist", {}).get("value")) or "Charlie Huang",
                     "description": clean(meta.get("ImageDescription", {}).get("value"))[:300]})
    for identifier, filename in IA_FILES:
        meta = get_json(f"https://archive.org/metadata/{identifier}")["metadata"]
        name = f"ia_{filename}"
        download(f"https://archive.org/download/{identifier}/{urllib.parse.quote(filename)}", STAGE / name)
        creator = meta.get("creator")
        rows.append({"file": name, "source": "Internet Archive", "url": f"https://archive.org/details/{identifier}",
                     "license": meta.get("licenseurl"), "license_url": meta.get("licenseurl"),
                     "author": ", ".join(sorted(set(creator))) if isinstance(creator, list) else creator,
                     "description": clean(meta.get("description"))[:300]})
    MANIFEST.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(rows)} files in {STAGE}")


if __name__ == "__main__":
    main()
