"""Fine-tune AST (Audio Spectrogram Transformer) to classify the palo.

    python -m flamenco.train_ast data/manifest.csv models/ast-palo
    python -m flamenco.train_ast data/manifest.csv models/smoke --max-steps 10   # quick test

Data:
  train       each epoch takes CROPS_PER_TRACK random 10 s windows per track
              (data augmentation); near-silent windows are drawn again.
  validation  fixed windows over each track; the score is per TRACK: we
              average the clip probabilities of a track (labels are weak:
              they belong to the track, not to each window).
  test        not used here. It stays unseen until the final evaluation.

Model: the AudioSet head (527 labels) is replaced by a new head with one
output per palo; the transformer body keeps what it learned and is trained
further with a small learning rate.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import ASTFeatureExtractor, ASTForAudioClassification, Trainer, TrainingArguments

from flamenco.labels import CLASSES

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
CLIP_SECONDS = 10.24  # the 1024 frames (10 ms each) that AST reads; MERT uses the same windows
CROPS_PER_TRACK = 4  # random windows per track per epoch
EVAL_WINDOWS = 12  # max fixed windows per validation track
MIN_RMS = 0.01  # below this (full scale = 1.0) a window counts as silence
LABEL2ID = {palo: i for i, palo in enumerate(CLASSES)}


def read_manifest(path: str) -> pd.DataFrame:
    """The manifest with the label fixes applied (label_fixes.csv in the same folder).
    The fixes stay apart from labels.csv, so the split never moves. A fix to a
    palo outside CLASSES (e.g. "taranta") removes the track."""
    df = pd.read_csv(path)
    fixes_csv = Path(path).parent / "label_fixes.csv"
    if not fixes_csv.exists():
        return df
    fixes = pd.read_csv(fixes_csv)
    unknown = set(fixes["path"]) - set(df["path"])
    assert not unknown, f"label fixes for tracks that are not in the manifest: {unknown}"
    df["palo"] = df["path"].map(dict(zip(fixes["path"], fixes["palo"]))).fillna(df["palo"])
    dropped = ~df["palo"].isin(CLASSES)
    print(f"{len(fixes)} label fix(es) applied, {dropped.sum()} track(s) removed (palo not in CLASSES)")
    return df[~dropped].reset_index(drop=True)


def load_window(npy: str, start: int, length: int) -> np.ndarray:
    audio = np.load(npy, mmap_mode="r")  # reads only the part we slice
    window = np.asarray(audio[start:start + length], dtype=np.float32) / 32768
    return np.pad(window, (0, length - len(window)))  # short track: pad with silence


class RandomCrops(Dataset):
    """Training data: a new random window from each track every time.
    The .npy files must have the sample rate of the extractor."""

    def __init__(self, tracks: pd.DataFrame, extractor):
        self.tracks = tracks.reset_index(drop=True)
        self.extractor = extractor
        self.rate = extractor.sampling_rate
        assert (tracks["rate"] == self.rate).all(), "the .npy sample rate does not match the model"
        self.clip = int(CLIP_SECONDS * self.rate)

    def __len__(self) -> int:
        return len(self.tracks) * CROPS_PER_TRACK

    def __getitem__(self, i: int) -> dict:
        track = self.tracks.iloc[i % len(self.tracks)]
        last_start = max(0, int(track["seconds"] * self.rate) - self.clip)
        for _ in range(5):  # try a few times to avoid silence
            start = int(torch.randint(0, last_start + 1, ()).item())
            window = load_window(track["npy"], start, self.clip)
            if np.sqrt(np.mean(window**2)) >= MIN_RMS:
                break
        features = self.extractor(window, sampling_rate=self.rate, return_tensors="np")
        return {"input_values": features["input_values"][0], "labels": LABEL2ID[track["palo"]]}


class FixedWindows(Dataset):
    """Validation data: evenly spaced windows; `track_ids` maps each window to its track."""

    def __init__(self, tracks: pd.DataFrame, extractor):
        self.extractor = extractor
        self.rate = extractor.sampling_rate
        assert (tracks["rate"] == self.rate).all(), "the .npy sample rate does not match the model"
        self.clip = int(CLIP_SECONDS * self.rate)
        self.items = []  # (npy, start, label, track index)
        for t, track in tracks.reset_index(drop=True).iterrows():
            last_start = max(0, int(track["seconds"] * self.rate) - self.clip)
            n = min(EVAL_WINDOWS, last_start // self.clip + 1)
            for start in np.linspace(0, last_start, n).astype(int):
                self.items.append((track["npy"], int(start), LABEL2ID[track["palo"]], t))
        self.track_ids = np.array([item[3] for item in self.items])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        npy, start, label, _ = self.items[i]
        window = load_window(npy, start, self.clip)
        features = self.extractor(window, sampling_rate=self.rate, return_tensors="np")
        return {"input_values": features["input_values"][0], "labels": label}


def make_metrics(track_ids: np.ndarray):
    """Clip-level and track-level scores. The Trainer does not shuffle eval data,
    so prediction i belongs to window i of the FixedWindows dataset."""

    def compute(pred) -> dict:
        logits, labels = pred.predictions, pred.label_ids
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        clip_pred = probs.argmax(-1)

        tracks = np.unique(track_ids)
        track_probs = np.stack([probs[track_ids == t].mean(0) for t in tracks])
        track_labels = np.array([labels[track_ids == t][0] for t in tracks])
        track_pred = track_probs.argmax(-1)
        return {
            "clip_accuracy": accuracy_score(labels, clip_pred),
            "clip_macro_f1": f1_score(labels, clip_pred, average="macro"),
            "track_accuracy": accuracy_score(track_labels, track_pred),
            "track_macro_f1": f1_score(track_labels, track_pred, average="macro"),
        }

    return compute


class WeightedTrainer(Trainer):
    """Cross-entropy with class weights: a mistake on a rare palo costs more."""

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("output_dir")
    parser.add_argument("--epochs", type=float, default=10)
    parser.add_argument("--lr", type=float, default=3e-5)
    # AST attends over 1212 tokens per clip, so memory grows fast: on an 18 GB Mac
    # a batch of 8 does not fit (batch 4 = 9.3 GB). 4 x 2 accumulation steps acts like 8.
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=-1, help="for a quick test run")
    return parser


def make_trainer(model, extractor, args: argparse.Namespace, **trainer_kwargs) -> Trainer:
    """Data, class weights and training settings. train_mert.py uses this too,
    so that AST and MERT differ only in the model."""
    df = read_manifest(args.manifest)
    train, val = df[df["split"] == "train"], df[df["split"] == "validation"]
    smoke = args.max_steps > 0
    if smoke:  # small validation set, so a test run finishes fast
        val = val.groupby("palo").head(2)

    # Class weight = how much rarer than average: bulerias (many) < 1 < alegrias (few).
    counts = train["palo"].value_counts().reindex(CLASSES).to_numpy()
    class_weights = torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32)
    print("class weights:", dict(zip(CLASSES, class_weights.numpy().round(2))))

    train_data = RandomCrops(train, extractor)
    val_data = FixedWindows(val, extractor)
    print(f"train: {len(train)} tracks -> {len(train_data)} crops/epoch; "
          f"validation: {len(val)} tracks -> {len(val_data)} windows")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_steps=0.1,  # first 10% of the steps: learning rate ramps up from 0
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulate,
        per_device_eval_batch_size=8,
        eval_strategy="steps" if smoke else "epoch",
        eval_steps=args.max_steps if smoke else None,
        save_strategy="steps" if smoke else "epoch",
        save_steps=args.max_steps if smoke else None,
        save_total_limit=2,  # the best checkpoint is always kept as well
        # Not load_best_model_at_end: in this transformers version it skips the
        # rename of old AST weight names and silently loads only the head.
        # We reload the best checkpoint ourselves after training.
        metric_for_best_model="track_macro_f1",
        logging_steps=5 if smoke else 20,
        dataloader_num_workers=2,
        report_to="none",
        seed=42,
    )
    return WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=make_metrics(val_data.track_ids),
        class_weights=class_weights,
        **trainer_kwargs,
    )


def main() -> None:
    args = make_parser().parse_args()
    extractor = ASTFeatureExtractor.from_pretrained(MODEL_ID)
    model = ASTForAudioClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(CLASSES),
        label2id=LABEL2ID,
        id2label={i: p for p, i in LABEL2ID.items()},
        ignore_mismatched_sizes=True,  # drop the 527-label head, make a new 6-label head
    )
    trainer = make_trainer(model, extractor, args)
    trainer.train()

    checkpoint = trainer.state.best_model_checkpoint
    print(f"best checkpoint: {checkpoint} (track_macro_f1 {trainer.state.best_metric:.3f})")
    trainer.model = ASTForAudioClassification.from_pretrained(checkpoint).to(trainer.args.device)
    best = Path(args.output_dir) / "best"
    trainer.save_model(best)
    extractor.save_pretrained(best)
    print("validation of the saved model:", trainer.evaluate())
    print(f"saved the best model to {best}")


if __name__ == "__main__":
    main()
