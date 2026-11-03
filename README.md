# Flamenco palo detector

Upload a flamenco recording and the model tells you its **palo** (the flamenco
form). It also shows how sure it is, window by window, over the whole track.

**Demo:** _(Hugging Face Space link)_ · **Model:** fine-tuned
[MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M)

I built this project to learn audio fine-tuning with Hugging Face end to end:
data labelling, a leak-free split, training, evaluation, and a web app.
I know flamenco well, so I could check the labels and the model's mistakes by ear.

_(screenshot of the app: docs/app.png)_

## The task

Six palos, which differ in rhythm (compás), tonality, and character:

| palo | compás | mode |
|---|---|---|
| bulerías | 12 beats, fast | Phrygian / major |
| soleá | 12 beats, slow | Phrygian |
| alegrías | 12 beats | major |
| siguiriya | 12 beats in an uneven 5-beat pattern | Phrygian |
| tangos | 4 beats | Phrygian |
| fandango | 3 beats or free | Phrygian + major |

This is hard for a model because bulerías, soleá and alegrías share the same
accent pattern (3, 6, 8, 10, 12), and siguiriya uses the same accents in a different order.

## Results

All scores are per **track** on recordings from albums that the model never
saw in training. A track score is the mean of the probabilities of its 10 s windows.

| model | training tracks | validation macro-F1 | test macro-F1 | test accuracy |
|---|---|---|---|---|
| AST (AudioSet) | 315 | 0.717 | 0.570 | 0.662 |
| MERT-v1-95M | 315 | 0.676 | 0.719 | 0.770 |
| **MERT-v1-95M (demo model)** | **403** | **0.732** | **0.681** | **0.730** |

Test set of the demo model, per palo (74 tracks, 9 albums):

| | bulerías | fandango | soleá | tangos | siguiriya | alegrías |
|---|---|---|---|---|---|---|
| F1 | 0.93 | 0.56 | 0.70 | 0.60 | 0.70 | 0.60 |
| tracks | 22 | 11 | 16 | 10 | 9 | 6 |

**How to read these numbers.** The test set has 74 tracks from only 9 albums,
and the errors of one album are correlated (the same singer, guitarist and studio).
So the real uncertainty is large:

- MERT vs AST on the test set: MERT is better on 7 of 9 albums. A bootstrap
  over albums gives a 95% interval of [+0.01, +0.25] for the macro-F1 difference.
  This is the one clear result.
- The two MERT rows are **not** different: the interval of the difference is
  [−0.18, +0.11]. On validation the order was the opposite. This shows why a
  model chosen on a small validation set can look better than it is.

## How it works

```
audio ─ ffmpeg ─> mono 24 kHz ─> 10.24 s windows ─> MERT (12 layers)
      ─> learned weighted sum of the layers ─> mean over time ─> linear head ─> 6 palos
track score = mean of the window probabilities
```

- **Model:** MERT-v1-95M, a self-supervised music model. Transformers has no
  classifier for it, so `flamenco/train_mert.py` adds one: a softmax-weighted sum
  of the 12 layer outputs (as in the MERT paper), mean pooling over time, and a
  linear layer. The CNN feature encoder stays frozen.
- **Training:** 10 epochs, 4 random 10 s crops per track per epoch, class-weighted
  cross-entropy, AdamW with a learning rate of 3e-5 for the pretrained body and
  1e-3 for the new head and layer weights. The best epoch is chosen by the
  validation track macro-F1. About 2 hours on an 18 GB Apple Silicon Mac (MPS).
- **AST baseline:** `MIT/ast-finetuned-audioset-10-10-0.4593` with a new 6-class
  head, the same data, and the same training loop (`flamenco/train_ast.py`).

## Data

A private collection of flamenco recordings (commercial CDs). The audio, the
track list and the labels are **not** published. Only the code and this summary are.

| palo | train | validation | test |
|---|---|---|---|
| bulerías | 113 | 21 | 22 |
| fandango | 78 | 17 | 11 |
| soleá | 66 | 13 | 16 |
| tangos | 48 | 10 | 10 |
| siguiriya | 55 | 11 | 9 |
| alegrías | 43 | 6 | 6 |
| **tracks** | **403** | **78** | **74** |
| albums | 57 | 14 | 9 |
| hours | 29.0 | 5.2 | 4.3 |

## Method decisions

The parts I would point a reviewer to:

- **Split by album** (`make_split.py`). An album is never in two sets. Otherwise the
  model can recognise a singer or a studio sound and not the palo. The split is
  stratified by palo, searched over seeds and folds for balance, and then fixed,
  so that every model uses the same test set.
- **Duplicate detection with audio fingerprints** (`find_duplicates.py`). The same
  recording often appears on several compilations. Chromaprint fingerprints find
  the copies, so that one recording cannot be in train *and* test. If the copies
  have different labels, all copies are dropped. The threshold is higher for very
  repetitive audio (palmas only), where different recordings look alike.
- **Labels from the titles** (`labels.py`, `make_labels.py`). The palo is parsed from
  the file name, or from a matched release in the MusicBrainz / Discogs database
  (beets). The parser handles accents, spelling variants (seguiriya / siguiriya),
  cut-off names, and "soleá por bulerías" vs "bulerías".
- **Test-set discipline.** Models and settings were compared on validation only.
  After a first test run of the AST baseline, I checked **all 74 test labels by
  ear** (1 was wrong and was fixed). All test scores above use the fixed labels,
  and each model was evaluated on the test set once, after it was chosen.
- **Honest uncertainty.** McNemar's test and a bootstrap over **albums**, not
  over tracks, because the errors within an album are correlated.

## What I learned from the errors

- **AST used fandango as a "dumping ground".** Half of its validation errors were
  predicted as fandango. MERT, a music model, does not have this problem.
- **The model counts claps; it does not hear the compás.** A 30 s recording of
  my own bulerías palmas (accents only, ~95 claps per minute) was classified
  as soleá. The training bulerías have 2–3× more claps per minute. Mean pooling
  over time keeps "how much happens", but it loses "in which order", and the
  order of the accents is what separates the palos. A head that keeps the time
  order (1D-CNN or GRU over the MERT frames) is the next step.
- **Unknown palos get the nearest known palo.** A taranta (not one of the 6
  classes) is classified as fandango, which is musically its closest relative.
  The timeline in the app shows that the windows disagree with each other, which
  is a better hint of "I do not know" than the top score.

## Limitations

- Only 6 palos. Every input gets one of them, also a sevillana or a taranta.
- Mostly cante with guitar from commercial recordings. Phone recordings, solo guitar
  and palmas-only audio are outside the training distribution.
- The test set is small (74 tracks, 9 albums), so the scores have wide error bars.
- The model weights are not public, because the training audio is copyrighted.
  You can use the model in the demo.

## Usage

Requirements: Python 3.12+, `ffmpeg`, and for duplicate detection the Chromaprint library.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest
```

The pipeline, for your own folder of flamenco recordings:

```bash
python -m flamenco.find_palo_releases <collection> data/palo_report.json     # optional: database titles
python -m flamenco.find_duplicates data/duplicates.json <collection>
python -m flamenco.make_labels data/labels.csv data/duplicates.json <collection> data/palo_report.json
python -m flamenco.make_split data/labels.csv data/split.csv
python -m flamenco.decode_audio data/split.csv data/audio24k data/manifest24k.csv 24000
python -m flamenco.train_mert data/manifest24k.csv models/mert --head-lr 1e-3
python -m flamenco.evaluate data/manifest24k.csv models/mert/best validation
```

`add_train_only.py` adds extra folders to the training set only (validation and
test stay unchanged). Every script explains its options in its docstring.

Run the app locally:

```bash
python -m flamenco.predict models/mert/best some_track.mp3   # command line, with a timeline
python app.py                                               # web app on http://127.0.0.1:7860
```

## Project layout

```
flamenco/       the package: labels, duplicates, split, decoding, training, evaluation, prediction
app.py          Gradio app: palo for the whole track + a per-window timeline
tests/          tests for the title parser
tools/windows/  helpers to repair file names on an exFAT drive
```
