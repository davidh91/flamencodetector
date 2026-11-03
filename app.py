import gradio as gr
import pandas as pd

from flamenco.predict import CLASSES, Prediction, Predictor
from flamenco.train_ast import CLIP_SECONDS

palo_predictor = Predictor("models/mert-app/best")

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
        prediction = palo_predictor.predict(path=audio_path)
    except ValueError as e:
        raise gr.Error("Cannot read this file. Is it an audio file?") from e
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
)

demo.launch()
