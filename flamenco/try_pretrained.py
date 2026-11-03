"""Run a pre-trained AudioSet model on one audio file and print the top labels.

This model is NOT fine-tuned on palos. It only knows the 527 general
AudioSet labels (e.g. "Guitar", "Flamenco", "Speech", "Clapping").
"""

import sys

import librosa
import torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLE_RATE = 16_000  # AST expects 16 kHz mono audio
CLIP_SECONDS = 10


def main(path: str) -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    extractor = ASTFeatureExtractor.from_pretrained(MODEL_ID)
    model = ASTForAudioClassification.from_pretrained(MODEL_ID).to(device).eval()

    # Load 10 seconds from 30 s into the track (skips quiet intros).
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, offset=30, duration=CLIP_SECONDS)
    if len(audio) == 0:
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=CLIP_SECONDS)

    # Audio waveform -> log-mel spectrogram (the "image" the model sees).
    inputs = extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(device)
    print("input shape:", tuple(inputs["input_values"].shape))

    with torch.no_grad():
        logits = model(**inputs).logits[0]

    # AudioSet is multi-label, so we use sigmoid (not softmax).
    probs = torch.sigmoid(logits).cpu()
    top = torch.topk(probs, k=10)
    for score, idx in zip(top.values, top.indices):
        print(f"{score:.3f}  {model.config.id2label[idx.item()]}")


if __name__ == "__main__":
    main(sys.argv[1])
