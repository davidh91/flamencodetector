"""Decode every track in the split once to 16 kHz mono int16 .npy files.

    python -m flamenco.decode_audio data/split.csv data/audio16k data/manifest.csv          # AST
    python -m flamenco.decode_audio data/split.csv data/audio24k data/manifest24k.csv 24000  # MERT

MP3 decoding is slow, so training reads these arrays instead. int16 halves
the size of float32, and 16-bit is the resolution of a CD anyway.

We decode with ffmpeg: libsndfile (librosa's default) cannot read .m4a (AAC)
and gives up on damaged MP3s. The web app decodes uploads with ffmpeg too,
so training and inference see audio from the same decoder.
Already decoded files are skipped, so the script can be stopped and restarted.
Writes a manifest: the split rows plus the .npy path, the duration and the sample rate.
"""

import hashlib
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_RATE = 16_000  # what AST expects; MERT needs 24_000


def npy_name(path: str) -> str:
    """Stable, filesystem-safe name for a source path."""
    return hashlib.sha1(path.encode()).hexdigest()[:16] + ".npy"


def read_pcm(src: str, rate: int) -> tuple[np.ndarray, str]:
    """Mono int16 samples at `rate`, and an error text ("" if it worked).
    predict.py uses this too, so the app decodes exactly like the training data."""
    # -ac 1: mono, -ar: resample, s16le: raw 16-bit samples to stdout.
    cmd = ["ffmpeg", "-v", "error", "-i", src, "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"]
    result = subprocess.run(cmd, capture_output=True)
    pcm = np.frombuffer(result.stdout, dtype=np.int16)
    if result.returncode != 0 or len(pcm) == 0:
        return pcm, result.stderr.decode(errors="replace").strip()[-300:] or "no audio"
    return pcm, ""


def decode(job: tuple[str, str, int]) -> tuple[str, float | None, str]:
    src, dst, rate = job
    if Path(dst).exists():
        return src, len(np.load(dst, mmap_mode="r")) / rate, ""
    pcm, error = read_pcm(src, rate)
    if error:  # report and continue; one bad file must not stop the run
        return src, None, error
    tmp = dst + ".tmp.npy"
    np.save(tmp, pcm)
    Path(tmp).rename(dst)  # never leave a half-written .npy behind
    return src, len(pcm) / rate, ""


def main(split_csv: str, out_dir: str, manifest_csv: str, rate: int = SAMPLE_RATE) -> None:
    df = pd.read_csv(split_csv)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df["npy"] = [str(Path(out_dir) / npy_name(p)) for p in df["path"]]

    durations, errors = {}, []
    with ProcessPoolExecutor() as pool:
        jobs = [(src, dst, rate) for src, dst in zip(df["path"], df["npy"])]
        for n, (src, seconds, error) in enumerate(pool.map(decode, jobs, chunksize=4), 1):
            if error:
                errors.append((src, error))
            else:
                durations[src] = seconds
            if n % 50 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} decoded", flush=True)

    df["seconds"] = df["path"].map(durations)
    df["rate"] = rate  # training checks that it matches the model
    df = df[df["seconds"].notna()]
    df.to_csv(manifest_csv, index=False)
    for src, error in errors:
        print(f"ERROR {src}: {error}")
    print(f"Wrote {len(df)} tracks ({df['seconds'].sum() / 3600:.1f} h) to {manifest_csv}")


if __name__ == "__main__":
    main(*sys.argv[1:4], *[int(r) for r in sys.argv[4:5]])
