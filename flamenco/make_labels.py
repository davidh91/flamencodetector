"""Build data/labels.csv: one row per recording, with the most informative title.

    python -m flamenco.make_labels data/labels.csv data/duplicates.json \
        <collection 1> data/palo_report.json [<collection 2> data/palo_report_2.json ...] \
        [--extra-labels data/extra_labels.csv] [--skip-dirs data/skip_dirs.txt]

Title source per file, best first:
  1. extra     - a label from --extra-labels (for example from a published
                 tracklist); its `source` column becomes the title_source
  2. filename  - the filename contains a palo
  3. database  - a matched beets candidate (distance <= MAX_DISTANCE) has a palo
                 in the title that matches this file
  4. none      - no palo found; palo_text stays empty (label by hand or skip)

--skip-dirs: folders (relative to a collection, one per line) that hold the
same recordings as another folder, so they are skipped.
Duplicates (same recording, found by find_duplicates.py) keep one copy: the
one with the best label. Copies with different palos are printed for review.

The original files are never changed. palo_text is raw text; mapping it to
the training classes is the job of labels.py.
"""

import argparse
import csv
import difflib
import json
import os
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import mutagen

from flamenco.labels import palo_from_title
from flamenco.palos import has_palo, normalize

AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".flac"}
MIN_TITLE_SIMILARITY = 0.6

LEADING_NUMBER_RE = re.compile(r"^\s*(\d+)\s*[-.]?\s*")
FIELDS = ["path", "collection", "folder", "artist", "album", "title", "title_source", "palo_text"]
SOURCE_RANK = {"filename": 1, "database": 2, "none": 3}  # extra labels (any other source): 0


def bare_title(title: str) -> str:
    """'Samara (Bulerías)' -> 'samara': for comparing titles without the palo."""
    return normalize(title.split("(")[0]).strip()


def read_tags(path: Path) -> tuple[str, str]:
    try:
        tags = mutagen.File(path, easy=True) or {}
    except mutagen.MutagenError:
        tags = {}
    return tags.get("artist", [""])[0], tags.get("album", [""])[0]


def database_title(title: str, candidate_titles: list[str]) -> str | None:
    """The candidate title with a palo that best matches this file's title."""
    with_palo = [t for t in candidate_titles if has_palo(t)]
    scored = [
        (difflib.SequenceMatcher(None, bare_title(title), bare_title(t)).ratio(), t)
        for t in with_palo
    ]
    best = max(scored, default=(0.0, None))
    return best[1] if best[0] >= MIN_TITLE_SIMILARITY else None


def label_file(file: Path, root: Path, chosen: dict | None, extra: dict[str, dict]) -> dict:
    folder = str(file.parent.relative_to(root))
    title = LEADING_NUMBER_RE.sub("", file.stem)
    artist, album = read_tags(file)
    row = {"path": str(file), "collection": root.name, "folder": folder, "artist": artist, "album": album}

    if found := extra.get(str(file)):
        return row | {"title": found["title"], "title_source": found["source"], "palo_text": found["palo_text"]}
    if has_palo(title):
        return row | {"title": title, "title_source": "filename", "palo_text": title}
    if chosen and (db_title := database_title(title, chosen["titles"])):
        return row | {"title": db_title, "title_source": "database", "palo_text": db_title}
    return row | {"title": title, "title_source": "none", "palo_text": ""}


def label_collection(root_dir: Path, report: dict, extra: dict, skip_dirs: set[str]) -> tuple[list[dict], int]:
    rows, unreadable = [], 0
    for file in sorted(root_dir.rglob("*")):
        if file.suffix.lower() not in AUDIO_EXTENSIONS or file.name.startswith("._"):
            continue
        # The disk can store "í" as "i" + accent (NFD); compare in one form (NFC).
        if unicodedata.normalize("NFC", str(file.parent.relative_to(root_dir))) in skip_dirs:
            continue
        if not os.path.exists(file):  # broken exFAT names: listed, but cannot be opened
            unreadable += 1
            continue
        chosen = report.get(str(file.parent), {}).get("chosen")
        rows.append(label_file(file, root_dir, chosen, extra))
    return rows, unreadable


def best_copy(row: dict) -> tuple:
    """Sort key: a copy with a usable palo first, then the most reliable title source."""
    return (palo_from_title(row["palo_text"]) is None, SOURCE_RANK.get(row["title_source"], 0), row["path"])


def drop_duplicates(rows: list[dict], groups: list[list[str]]) -> list[dict]:
    by_path = {row["path"]: row for row in rows}
    drop = set()
    for group in groups:
        copies = sorted((by_path[p] for p in group if p in by_path), key=best_copy)
        palos = {palo_from_title(c["palo_text"]) for c in copies} - {None}
        if len(palos) > 1:
            # The sources disagree; we cannot tell which is right, so drop all copies.
            print(f"  CONFLICT {sorted(palos)}, dropped: " + " | ".join(c["path"] for c in copies))
            drop.update(c["path"] for c in copies)
        else:
            drop.update(c["path"] for c in copies[1:])
    return [row for row in rows if row["path"] not in drop]


def read_extra_labels(path: str | None) -> dict[str, dict]:
    """{path: {"title", "palo_text", "source"}} from a CSV with these columns plus "path"."""
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return {row["path"]: row for row in csv.DictReader(f)}


def read_skip_dirs(path: str | None) -> set[str]:
    if not path:
        return set()
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return {unicodedata.normalize("NFC", line.strip()) for line in lines if line.strip() and not line.startswith("#")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("duplicates")
    parser.add_argument("collections", nargs="+", help="pairs: <collection folder> <palo report json>")
    parser.add_argument("--extra-labels", help="CSV with path,title,palo_text,source; wins over the other sources")
    parser.add_argument("--skip-dirs", help="text file: one folder per line, relative to its collection")
    args = parser.parse_args()
    if len(args.collections) % 2:
        parser.error("collections must be pairs of <folder> <report>")
    extra, skip_dirs = read_extra_labels(args.extra_labels), read_skip_dirs(args.skip_dirs)

    rows = []
    for root, report_path in zip(args.collections[::2], args.collections[1::2]):
        report = {r["folder"]: r for r in json.loads(Path(report_path).read_text())}
        found, unreadable = label_collection(Path(root), report, extra, skip_dirs)
        print(f"{root}: {len(found)} files, {unreadable} unreadable (skipped)")
        rows += found

    groups = json.loads(Path(args.duplicates).read_text())["groups"]
    before = len(rows)
    rows = drop_duplicates(rows, groups)
    print(f"Dropped {before - len(rows)} duplicate copies")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    counts = defaultdict(int)
    for row in rows:
        counts[row["title_source"]] += 1
    print(f"Wrote {len(rows)} rows to {args.out}: {dict(counts)}")


if __name__ == "__main__":
    main()
