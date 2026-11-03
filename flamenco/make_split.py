"""Split the labelled tracks into train / validation / test, by album.

    python -m flamenco.make_split data/labels.csv data/split.csv

- Only tracks whose palo is one of labels.CLASSES are used.
- An album (collection + folder) is never in two sets, so the model cannot
  recognize a singer, guitarist or studio sound from training at test time.
  Duplicates across albums are already removed by make_labels.py.
- Stratified: each set gets about the same share of each palo.
- Big albums make a group split uneven (one album with 26 fandangos is
  one group), and StratifiedGroupKFold puts such groups in fold 0 for every seed.
  So we search over seeds AND over which folds become test and validation,
  and keep the most balanced result. The search is deterministic, so the
  split is reproducible. Keep data/split.csv fixed, so that later models
  (MERT) use exactly the same test set.
"""

import itertools
import sys

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from flamenco.labels import CLASSES, palo_from_title

N_FOLDS = 6  # 1 fold test + 1 fold validation + 4 folds train = ~17 / 17 / 67 %
N_SEEDS = 50
TARGET = {"train": 4 / 6, "validation": 1 / 6, "test": 1 / 6}


def folds_for(df: pd.DataFrame, seed: int) -> pd.Series:
    """Fold number (0 .. N_FOLDS-1) per track for one seed."""
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold = pd.Series(-1, index=df.index)
    for n, (_, idx) in enumerate(splitter.split(df, df["palo"], groups=df["album"])):
        fold.iloc[idx] = n
    return fold


def to_split(fold: pd.Series, test_fold: int, val_fold: int) -> pd.Series:
    return fold.map(lambda f: "test" if f == test_fold else "validation" if f == val_fold else "train")


def imbalance(palo: pd.Series, split: pd.Series) -> float:
    """Worst distance, over all palos and sets, between the real and the target share."""
    shares = pd.crosstab(palo, split, normalize="index")
    return max(abs(shares[name] - target).max() for name, target in TARGET.items())


def main(labels_csv: str, out: str) -> None:
    df = pd.read_csv(labels_csv, keep_default_na=False)
    df["palo"] = df["palo_text"].map(palo_from_title)
    df = df[df["palo"].isin(CLASSES)].reset_index(drop=True)
    df["album"] = df["collection"] + "/" + df["folder"]

    best = None  # (imbalance, seed, test_fold, val_fold)
    for seed in range(N_SEEDS):
        fold = folds_for(df, seed)
        for test_fold, val_fold in itertools.permutations(range(N_FOLDS), 2):
            score = imbalance(df["palo"], to_split(fold, test_fold, val_fold))
            if best is None or score < best[0]:
                best = (score, seed, test_fold, val_fold)
    score, seed, test_fold, val_fold = best
    df["split"] = to_split(folds_for(df, seed), test_fold, val_fold)
    print(f"Best: seed {seed}, test fold {test_fold}, validation fold {val_fold} "
          f"(worst share error {score:.3f})\n")

    # Safety check: no album may appear in more than one set.
    leaks = df.groupby("album")["split"].nunique()
    assert (leaks == 1).all(), f"albums in two sets: {list(leaks[leaks > 1].index)}"

    df[["path", "album", "artist", "title", "title_source", "palo", "split"]].to_csv(out, index=False)

    table = pd.crosstab(df["palo"], df["split"], margins=True)[["train", "validation", "test", "All"]]
    print(table.to_string())
    print()
    print(df.groupby("split")["album"].nunique().rename("albums").to_string())
    print(f"\nWrote {len(df)} tracks to {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
