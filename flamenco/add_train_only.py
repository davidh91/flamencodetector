"""Add tracks from extra folders to the TRAINING set only.

    python -m flamenco.add_train_only data/split.csv data/duplicates_extra.json data/split_extra.csv \
        <folder 1> [<folder 2> ...] [--min-similarity 0.90] [--overrides data/overrides.csv] [--source extra]

Use it for material that helps the model learn but is not a fair test, for
example practice recordings where every palo has the same musicians and the
same studio: then only compás and tonality tell the palos apart.

- Train only. Validation and test stay the same, so the scores stay
  comparable with runs without the extra tracks. The input split is not changed.
- The files directly in each folder. The palo comes from the file name
  (labels.palo_from_title), except for the files in --overrides (a CSV with
  the columns stem,palo): a folder name such as "Solea" can be part of a file
  name, and then the parser sees two palos.
- Duplicates (Chromaprint pairs from find_duplicates.py):
  * a copy of a track that is already in the split is dropped (a copy of a
    validation or test track would be a leak), at the threshold of the pairs file;
  * inside the new tracks a copy needs --min-similarity. Very repetitive
    audio (palmas only) reaches 0.80 for different tracks, so use a higher
    value there. Of real copies only the first is kept, but when the copies
    have different palos all are dropped: the same recording with two
    labels cannot teach the model anything.
"""

import argparse
import json
import os
import unicodedata
from pathlib import Path

import pandas as pd

from flamenco.find_duplicates import groups
from flamenco.labels import CLASSES, palo_from_title

AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".flac"}


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def read_overrides(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    df = pd.read_csv(path)
    return {nfc(stem): palo for stem, palo in zip(df["stem"], df["palo"])}


def palo_of(path: str, overrides: dict[str, str]) -> str | None:
    title = nfc(Path(path).stem)
    return overrides.get(title) or palo_from_title(title)


def extra_tracks(folders: list[str], overrides: dict[str, str], source: str) -> pd.DataFrame:
    rows = []
    for folder in map(Path, folders):
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() not in AUDIO_EXTENSIONS or f.name.startswith("._"):
                continue
            palo = palo_of(str(f), overrides)
            if palo in CLASSES:
                rows.append({"path": str(f), "album": nfc(f"{folder.parent.name}/{folder.name}"), "artist": "",
                             "title": nfc(f.stem), "title_source": source, "palo": palo, "split": "train"})
    empty = {folder for folder in folders if not any(Path(r["path"]).parent == Path(folder) for r in rows)}
    assert not empty, f"folders without tracks (renamed?): {empty}"
    return pd.DataFrame(rows)


def drop_duplicates(extra: pd.DataFrame, split: pd.DataFrame, pairs: list[dict], min_similarity: float,
                    overrides: dict[str, str]) -> pd.DataFrame:
    in_split = dict(zip(split["path"].map(nfc), split["split"]))
    candidates = set(extra["path"].map(nfc))
    pairs = [(nfc(p["a"]), nfc(p["b"]), p["similarity"]) for p in pairs]
    drop = set()

    for a, b, _ in pairs:  # already in the data: never add a second copy
        for new, old in [(a, b), (b, a)]:
            if new in candidates and old in in_split:
                drop.add(new)
                print(f"  drop copy of a {in_split[old]} track: {Path(new).name}")

    strict = [(a, b) for a, b, score in pairs if score >= min_similarity]
    for group in groups(strict):
        copies = sorted(p for p in group if p in candidates)
        if not copies:
            continue
        palos = {palo_of(p, overrides) for p in group} - {None}
        if len(palos) > 1:
            drop.update(copies)
            print(f"  drop {len(copies)}: one recording with two palos ({' / '.join(sorted(palos))}): "
                  f"{Path(copies[0]).name}")
        else:
            drop.update(copies[1:])
            for p in copies[1:]:
                print(f"  drop duplicate: {Path(p).name}  (= {Path(copies[0]).name})")
    return extra[~extra["path"].map(nfc).isin(drop)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split")
    parser.add_argument("duplicates", help="pairs JSON from find_duplicates.py, over the split AND the new folders")
    parser.add_argument("out")
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--min-similarity", type=float, default=0.80, help="duplicate threshold inside the new tracks")
    parser.add_argument("--overrides", help="CSV with stem,palo for files whose name the parser gets wrong")
    parser.add_argument("--source", default="extra", help="the title_source of the new rows")
    args = parser.parse_args()

    split = pd.read_csv(args.split)
    overrides = read_overrides(args.overrides)
    extra = extra_tracks(args.folders, overrides, args.source)
    print(f"{len(extra)} new tracks with a palo in CLASSES")
    pairs = json.loads(Path(args.duplicates).read_text())["pairs"]
    extra = drop_duplicates(extra, split, pairs, args.min_similarity, overrides)

    unreadable = extra[~extra["path"].map(os.path.exists)]
    for path in unreadable["path"]:
        print(f"  skip unreadable: {path}")
    extra = extra[extra["path"].map(os.path.exists)]

    table = pd.DataFrame({
        "train before": split[split["split"] == "train"]["palo"].value_counts(),
        "new": extra["palo"].value_counts(),
    }).reindex(CLASSES).fillna(0).astype(int)
    table["train after"] = table.sum(axis=1)
    print(table.to_string())

    pd.concat([split, extra], ignore_index=True).to_csv(args.out, index=False)
    print(f"Wrote {len(split)} + {len(extra)} tracks to {args.out}")


if __name__ == "__main__":
    main()
