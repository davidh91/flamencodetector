"""Predict the palo of an audio file with a trained model (used by the web app).

    python -m flamenco.predict models/mert-app/best Almoraima.mp3 [more files]

The steps must match the training and evaluation exactly, or the model gets
input it never saw (without an error):
  - decode with ffmpeg to mono at the sample rate of the model (decode_audio.read_pcm);
  - int16 -> float in [-1, 1], the same as train_ast.load_window;
  - 10.24 s windows, evenly spaced, the same as train_ast.FixedWindows;
  - the track score = the mean of the window probabilities, as in evaluate.py.
Two differences: we use ALL windows, not at most EVAL_WINDOWS (the app shows
a timeline), and we skip near-silent windows, as the training crops do.
"""

import sys
from dataclasses import dataclass

import numpy as np
import torch

from flamenco.decode_audio import read_pcm
from flamenco.evaluate import BATCH_SIZE, load_model
from flamenco.labels import CLASSES
from flamenco.train_ast import CLIP_SECONDS, MIN_RMS


@dataclass
class Prediction:
    scores: dict[str, float]  # {palo: probability} for the whole file
    seconds: float  # length of the audio
    times: np.ndarray  # start time of each window, in seconds
    probs: np.ndarray  # (window, palo) probabilities


class Predictor:
    """Loads the model once; call predict() for each file."""

    def __init__(self, model_dir: str):
        if torch.cuda.is_available():  # also on a ZeroGPU Space
            self.device = "cuda"
        else:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        model, self.extractor = load_model(model_dir)
        self.model = model.to(self.device).eval()
        self.rate = self.extractor.sampling_rate
        self.clip = int(CLIP_SECONDS * self.rate)

    def windows(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(start times in seconds, windows) with the spacing of FixedWindows."""
        audio = np.pad(audio, (0, max(0, self.clip - len(audio))))  # short audio: pad with silence
        last_start = len(audio) - self.clip
        starts = np.linspace(0, last_start, last_start // self.clip + 1).astype(int)
        windows = np.stack([audio[s:s + self.clip] for s in starts])
        loud = np.sqrt(np.mean(windows**2, axis=1)) >= MIN_RMS
        if loud.any():  # all silent: keep them all, the model must still answer
            starts, windows = starts[loud], windows[loud]
        return starts / self.rate, windows

    def read(self, path: str) -> np.ndarray:
        """int16 mono samples at the model's sample rate; ValueError if ffmpeg cannot decode the file."""
        pcm, error = read_pcm(path, self.rate)
        if error:
            raise ValueError(f"cannot decode {path}: {error}")
        return pcm

    def predict(self, path: str) -> Prediction:
        return self.predict_samples(self.read(path))

    def predict_samples(self, pcm: np.ndarray) -> Prediction:
        """The model part of predict(): the app runs only this part on the GPU."""
        times, windows = self.windows(pcm.astype(np.float32) / 32768)

        probs = []
        with torch.no_grad():
            for i in range(0, len(windows), BATCH_SIZE):
                features = self.extractor(list(windows[i:i + BATCH_SIZE]), sampling_rate=self.rate,
                                          return_tensors="pt")
                logits = self.model(input_values=features["input_values"].to(self.device)).logits
                probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        probs = np.concatenate(probs)
        scores = {palo: float(p) for palo, p in zip(CLASSES, probs.mean(0))}
        return Prediction(scores, len(pcm) / self.rate, times, probs)


def main(model_dir: str, *paths: str) -> None:
    predictor = Predictor(model_dir)
    for path in paths:
        prediction = predictor.predict(path)
        print(f"\n{path} ({prediction.seconds:.1f} s)")
        for palo, p in sorted(prediction.scores.items(), key=lambda item: -item[1]):
            print(f"  {palo:>9} {p:.3f}")
        print("  timeline:")
        for t, p in zip(prediction.times, prediction.probs):
            print(f"    {int(t) // 60}:{int(t) % 60:02d}  {CLASSES[p.argmax()]:<9} {p.max():.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
