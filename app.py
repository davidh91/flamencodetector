try:
    # Only on a Hugging Face ZeroGPU Space: a GPU only while a @spaces.GPU function runs.
    # It must be imported before torch touches CUDA, so it comes first.
    import spaces
except ImportError:
    spaces = None

import os

import gradio as gr
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from flamenco.predict import CLASSES, Prediction, Predictor
from flamenco.train_ast import CLIP_SECONDS

# Locally: the model folder. On the Space: MODEL_REPO, a private Hub repo; snapshot_download
# reads it with the HF_TOKEN secret from the environment.
MODEL_DIR = snapshot_download(os.environ["MODEL_REPO"]) if "MODEL_REPO" in os.environ else "models/mert-app/best"
MAX_FILE_SIZE = "50mb"  # a 5-minute MP3 is ~10 MB; this also limits what reaches ffmpeg
CACHE_SECONDS = 3600  # Gradio keeps a copy of each upload; delete copies older than 1 hour, every hour

palo_predictor = Predictor(MODEL_DIR)


def on_gpu(fn):
    return spaces.GPU(duration=60)(fn) if spaces else fn


@on_gpu
def run_model(samples: np.ndarray) -> Prediction:
    return palo_predictor.predict_samples(samples)


# One fixed color per palo (colorblind-checked), so a palo keeps its color in every result.
PALO_COLORS = dict(zip(CLASSES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))


def timeline(prediction: Prediction) -> pd.DataFrame:
    """One row per (window, palo). The labels are zero-padded ("01:05"), so sorting them sorts by time."""
    rows = []
    for start, probs in zip(prediction.times, prediction.probs):
        label = f"{int(start) // 60:02d}:{int(start) % 60:02d}"
        rows += [{"start": label, "palo": palo, "probability": float(p)} for palo, p in zip(CLASSES, probs)]
    return pd.DataFrame(rows)


def predict(audio_path: str | None):
    if audio_path is None:
        raise gr.Error("File is empty. Choose a file to detect the palo")
    try:
        samples = palo_predictor.read(audio_path)
    except ValueError as e:
        raise gr.Error("Cannot read this file. Is it an audio file?") from e
    prediction = run_model(samples)
    if prediction.seconds < CLIP_SECONDS:
        gr.Warning(f"The audio is only {prediction.seconds:.1f} s. The model listens to {CLIP_SECONDS} s "
                   f"at a time, so the result is less reliable. Record at least 20 s.")
    return prediction.scores, timeline(prediction)


demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath"),
    outputs=[
        gr.Label(num_top_classes=3, label="palo (whole track)"),
        gr.BarPlot(
            x="start",
            y="probability",
            color="palo",
            color_map=PALO_COLORS,
            sort="x",
            y_lim=[0, 1],
            tooltip="all",
            x_title=f"window start (each window is {CLIP_SECONDS} s)",
            label="palo per window",
        ),
    ],
    api_name="predict_palo",
    title="Flamenco palo detector",
    description=(
        "Upload a flamenco recording, or use the microphone to record at least 20 s of flamenco music. "
        "The model picks one of six palos: bulerías, soleá, alegrías, siguiriya, tangos, fandango. "
        "A palo that is not in this list gets the closest of these six. "
        "Your audio is only used for this prediction, and the temporary copy is deleted within 1 hour."
    ),
    flagging_mode="never",  # the Flag button would save visitors' audio on the server
    delete_cache=(CACHE_SECONDS, CACHE_SECONDS),
)

demo.launch(max_file_size=MAX_FILE_SIZE)
