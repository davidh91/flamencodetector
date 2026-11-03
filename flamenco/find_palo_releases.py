"""Dry run: for each album folder, find the beets candidate with palos in its titles.

Does NOT import or change any file. Writes a JSON report for review:

    python -m flamenco.find_palo_releases <collection> data/palo_report.json

Selection rule: among candidates with distance <= MAX_DISTANCE, pick the one
with the most track titles that contain a palo. Tie-break on lowest distance.
"""

import json
import sys
from pathlib import Path

from beets import config, logging, plugins
from beets.autotag import Source, tag_album
from beets.library import Item
from beetsplug import chroma

from flamenco.palos import has_palo

AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".flac"}
MAX_DISTANCE = 0.30  # beets shows "strong" matches below ~0.04, so 0.30 is lenient
TOP_N = 5  # candidates per folder kept in the report

log = logging.getLogger("beets")


def album_folders(root: Path) -> list[Path]:
    """Folders that directly contain audio files."""
    folders = {p.parent for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS}
    return sorted(folders)


def describe(match) -> dict:
    info = match.info
    titles = [t.title for t in info.tracks]
    return {
        "source": info.data_source,
        "id": info.album_id,
        "artist": info.artist,
        "album": info.album,
        "year": info.year,
        "distance": round(float(match.distance), 3),
        "tracks": len(titles),
        "palo_titles": sum(has_palo(t) for t in titles),
        "titles": titles,
    }


def scan_folder(folder: Path) -> dict:
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)
    items = [Item.from_path(str(p)) for p in paths]

    # Same as the chroma plugin does at import time: fingerprint + AcoustID lookup.
    for item in items:
        chroma.acoustid_match(log, item.path)

    proposal = tag_album(Source.from_items(items))
    candidates = [describe(m) for m in proposal.candidates]
    eligible = [c for c in candidates if c["distance"] <= MAX_DISTANCE]
    best = max(eligible, key=lambda c: (c["palo_titles"], -c["distance"]), default=None)

    return {
        "folder": str(folder),
        "files": len(paths),
        "files_with_palo": sum(has_palo(p.stem) for p in paths),
        "chosen": best,
        "candidates": sorted(candidates, key=lambda c: c["distance"])[:TOP_N],
    }


def main(root: str, out: str) -> None:
    config.read()
    plugins.load_plugins()

    folders = album_folders(Path(root))
    report = []
    for n, folder in enumerate(folders, 1):
        print(f"[{n}/{len(folders)}] {folder.relative_to(root)}", flush=True)
        try:
            report.append(scan_folder(folder))
        except Exception as exc:  # keep going; one bad folder must not stop the run
            print(f"    ERROR: {exc}", flush=True)
            report.append({"folder": str(folder), "error": str(exc)})
        Path(out).write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"Wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
