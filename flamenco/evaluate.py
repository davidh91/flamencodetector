"""Evaluate a trained model per track: report per palo, confusion matrix, mistakes.

    python -m flamenco.evaluate data/manifest.csv models/ast-palo/best validation
    python -m flamenco.evaluate data/manifest.csv models/ast-palo/best test   # only at the very end
    python -m flamenco.evaluate data/manifest24k.csv models/mert-palo/best validation

Uses the same fixed windows and track aggregation (mean of the clip
probabilities) as the validation during training, so the numbers match.
Applies data/label_fixes.csv (see train_ast.read_manifest).
Writes one row per track to <model>/predictions_<split>.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from transformers import ASTFeatureExtractor, ASTForAudioClassification

from flamenco import train_mert
from flamenco.labels import CLASSES
from flamenco.train_ast import FixedWindows, read_manifest

BATCH_SIZE = 8


def predict_tracks(model, data: FixedWindows, device: str) -> np.ndarray:
    """Probabilities per track (rows in the order of the tracks), averaged over its windows."""
    probs = []
    with torch.no_grad():
        for batch in DataLoader(data, batch_size=BATCH_SIZE):
            logits = model(input_values=batch["input_values"].to(device)).logits
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    probs = np.concatenate(probs)
    return np.stack([probs[data.track_ids == t].mean(0) for t in np.unique(data.track_ids)])


def load_model(model_dir: str):
    """AST is a normal transformers model; MERT is our own class (see train_mert.py)."""
    if (Path(model_dir) / train_mert.INFO_FILE).exists():
        return train_mert.load_model(model_dir)
    model = ASTForAudioClassification.from_pretrained(model_dir)
    assert [model.config.id2label[i] for i in range(len(CLASSES))] == CLASSES
    return model, ASTFeatureExtractor.from_pretrained(model_dir)


def main(manifest: str, model_dir: str, split: str) -> None:
    tracks = read_manifest(manifest)
    tracks = tracks[tracks["split"] == split].reset_index(drop=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, extractor = load_model(model_dir)
    model = model.to(device).eval()

    probs = predict_tracks(model, FixedWindows(tracks, extractor), device)
    tracks["pred"] = [CLASSES[i] for i in probs.argmax(-1)]
    tracks["confidence"] = probs.max(-1)
    for i, palo in enumerate(CLASSES):
        tracks[f"p_{palo}"] = probs[:, i].round(3)

    print(f"{split}: {len(tracks)} tracks\n")
    print(classification_report(tracks["palo"], tracks["pred"], labels=CLASSES, digits=3, zero_division=0))

    # Rows = the true palo, columns = what the model said. The diagonal is correct.
    matrix = confusion_matrix(tracks["palo"], tracks["pred"], labels=CLASSES)
    print("confusion matrix (rows = true, columns = predicted)")
    print(pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_string(), "\n")

    # Shortcut check: is the model much better on tracks from one source (for
    # example old recordings with known labels) than on the rest?
    for source, group in tracks.groupby("title_source"):
        correct = (group["palo"] == group["pred"]).mean()
        print(f"{source}: {len(group)} tracks, accuracy {correct:.3f}")
    print(f"track macro-F1: {f1_score(tracks['palo'], tracks['pred'], labels=CLASSES, average='macro'):.3f}\n")

    wrong = tracks[tracks["palo"] != tracks["pred"]].sort_values("confidence", ascending=False)
    print(f"mistakes ({len(wrong)}), most confident first:")
    for _, t in wrong.iterrows():
        print(f"  {t['palo']:>9} -> {t['pred']:<9} {t['confidence']:.2f}  {t['artist']} - {t['title']}")

    out = Path(model_dir) / f"predictions_{split}.csv"
    tracks.drop(columns=["npy"]).to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
