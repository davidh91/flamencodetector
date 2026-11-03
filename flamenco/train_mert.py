"""Fine-tune MERT (a music model) to classify the palo, to compare with AST.

    python -m flamenco.train_mert data/manifest24k.csv models/mert-palo
    python -m flamenco.train_mert data/manifest24k.csv models/mert-palo-headlr --head-lr 1e-3
    python -m flamenco.train_mert data/manifest24k.csv models/smoke --max-steps 10   # quick test

Everything except the model comes from train_ast.py: the split, the 10.24 s
windows, the class weights, the metrics and the training settings. So the
comparison changes one thing only. MERT reads raw audio at 24 kHz, hence the
24 kHz manifest (see decode_audio.py).

Model: transformers has no MERT classifier, so we build one:
  1. each of the 12 transformer layers gives one vector per frame (75 per s).
     Lower layers hold more of the sound (timbre), higher layers more of the
     music (pitch, rhythm). A learned weighted sum mixes the layers, as in
     the MERT paper;
  2. the mean over time gives one vector per window;
  3. a linear layer maps that vector to one score per palo.
The CNN at the input (raw audio -> frames) stays frozen, the usual choice
when fine-tuning HuBERT-style models; it also saves memory.

--head-lr: a separate learning rate for the new parts (head and layer
weights). They start from nothing and need big steps; the pretrained body
needs small steps (--lr) to keep what it learned. Without --head-lr all
parts use --lr, like AST (run 1: the layer weights did not move).
"""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoFeatureExtractor, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

from flamenco.labels import CLASSES
from flamenco.train_ast import make_parser, make_trainer

MODEL_ID = "m-a-p/MERT-v1-95M"
REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"  # pinned: the model runs code from the Hub
INFO_FILE = "mert.json"  # marks a saved MERT model (evaluate.py looks for it)


class MERTClassifier(nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        # layerdrop=0: during training MERT skips a random layer now and then,
        # and then we would get 11 layer outputs instead of 12.
        self.mert = AutoModel.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True, layerdrop=0.0)
        for param in self.mert.feature_extractor.parameters():
            param.requires_grad = False

        # The MERT code does not fill `hidden_states` in transformers 5,
        # so a hook catches the output of each layer.
        self._layer_outputs = []
        for layer in self.mert.encoder.layers:
            layer.register_forward_hook(lambda module, inputs, output: self._layer_outputs.append(output))

        self.layer_weights = nn.Parameter(torch.zeros(len(self.mert.encoder.layers)))  # equal at the start
        self.head = nn.Linear(self.mert.config.hidden_size, num_labels)

    def forward(self, input_values: torch.Tensor, labels: torch.Tensor | None = None) -> SequenceClassifierOutput:
        self._layer_outputs.clear()
        self.mert(input_values)
        layers = torch.stack(self._layer_outputs)  # (layer, batch, frame, hidden)
        weights = torch.softmax(self.layer_weights, dim=0)
        mixed = (weights[:, None, None, None] * layers).sum(0)
        return SequenceClassifierOutput(logits=self.head(mixed.mean(1)))


def load_model(model_dir: str) -> tuple[MERTClassifier, object]:
    """Load a model saved by this script (the counterpart of from_pretrained)."""
    info = json.loads((Path(model_dir) / INFO_FILE).read_text())
    assert info["classes"] == CLASSES, f"model classes {info['classes']} != {CLASSES}"
    model = MERTClassifier(len(CLASSES))
    model.load_state_dict(load_file(Path(model_dir) / "model.safetensors"))
    return model, AutoFeatureExtractor.from_pretrained(model_dir)


def make_optimizer(model: MERTClassifier, lr: float, head_lr: float) -> torch.optim.Optimizer:
    """AdamW with two learning rates; the other settings are the Trainer defaults.
    The Trainer still adds the warmup and decay, scaled for each group."""
    new = [model.layer_weights, *model.head.parameters()]
    body = [param for param in model.mert.parameters() if param.requires_grad]
    groups = [{"params": body, "lr": lr}, {"params": new, "lr": head_lr}]
    return torch.optim.AdamW(groups, weight_decay=0.0)


def main() -> None:
    parser = make_parser()
    parser.add_argument("--head-lr", type=float, help="learning rate of the head and the layer weights")
    args = parser.parse_args()
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, revision=REVISION)
    model = MERTClassifier(len(CLASSES))
    optimizers = (make_optimizer(model, args.lr, args.head_lr), None) if args.head_lr else (None, None)
    trainer = make_trainer(model, extractor, args, optimizers=optimizers)
    trainer.train()

    # Our model is not a transformers model, so the Trainer saves only its
    # weights (model.safetensors). We load the best ones back into the model.
    checkpoint = trainer.state.best_model_checkpoint
    print(f"best checkpoint: {checkpoint} (track_macro_f1 {trainer.state.best_metric:.3f})")
    trainer.model.load_state_dict(load_file(Path(checkpoint) / "model.safetensors"))
    best = Path(args.output_dir) / "best"
    trainer.save_model(best)
    extractor.save_pretrained(best)
    (best / INFO_FILE).write_text(json.dumps({"model": MODEL_ID, "revision": REVISION, "classes": CLASSES}))
    print("layer weights:", torch.softmax(trainer.model.layer_weights, 0).detach().cpu().numpy().round(3))
    print("validation of the saved model:", trainer.evaluate())
    print(f"saved the best model to {best}")


if __name__ == "__main__":
    main()
