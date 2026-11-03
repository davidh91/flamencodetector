"""Find the same recording in different files, with Chromaprint fingerprints.

    python -m flamenco.find_duplicates data/duplicates.json \
        <collection 1> <collection 2>

Fingerprints are cached in data/fingerprints.json, so a second run is fast.
Two files are a duplicate when their durations are close and their
fingerprints agree on at least MIN_SIMILARITY of the bits (chance is ~0.5).
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import acoustid
import chromaprint
import numpy as np

AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".flac"}
CACHE = Path("data/fingerprints.json")
MAX_DURATION_DIFF = 15.0  # seconds; another master can cut applause or silence (París 1987: up to 13 s)
MAX_OFFSET = 30  # fingerprint frames (~0.12 s each) to slide when aligning
MIN_SIMILARITY = 0.80


def audio_files(roots: list[str]) -> list[str]:
    return sorted(
        str(p) for root in roots for p in Path(root).rglob("*")
        if p.suffix.lower() in AUDIO_EXTENSIONS and not p.name.startswith("._")
    )


def fingerprint(path: str) -> tuple[float, str] | None:
    try:
        duration, fp = acoustid.fingerprint_file(path)
    except (acoustid.FingerprintGenerationError, OSError) as exc:  # OSError: broken names on the exFAT drive
        print(f"    skip {path}: {exc}", flush=True)
        return None
    return duration, fp.decode()


def load_fingerprints(paths: list[str]) -> dict[str, tuple[float, np.ndarray]]:
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [p for p in paths if p not in cache]
    print(f"{len(paths)} files, {len(todo)} to fingerprint", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:  # fpcalc runs as a subprocess
        for n, (path, result) in enumerate(zip(todo, pool.map(fingerprint, todo)), 1):
            if result:
                cache[path] = result
            if n % 100 == 0:
                print(f"    {n}/{len(todo)}", flush=True)
                CACHE.write_text(json.dumps(cache))
    CACHE.write_text(json.dumps(cache))

    decoded = {}
    for path in paths:
        if path in cache:
            duration, fp = cache[path]
            raw, _ = chromaprint.decode_fingerprint(fp.encode())
            decoded[path] = (duration, np.array(raw, dtype=np.uint32))
    return decoded


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Share of equal bits at the best alignment of the two fingerprints."""
    best = 0.0
    for offset in range(-MAX_OFFSET, MAX_OFFSET + 1):
        x, y = (a[offset:], b) if offset >= 0 else (a, b[-offset:])
        n = min(len(x), len(y))
        if n < 50:
            continue
        diff_bits = np.unpackbits((x[:n] ^ y[:n]).view(np.uint8)).sum()
        best = max(best, 1 - diff_bits / (32 * n))
    return best


def groups(pairs: list[tuple[str, str]]) -> list[list[str]]:
    """Union-find: if A~B and B~C, then A, B, C are one group."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[find(a)] = find(b)
    clusters: dict[str, list[str]] = {}
    for x in parent:
        clusters.setdefault(find(x), []).append(x)
    return sorted(sorted(c) for c in clusters.values())


def main(out: str, roots: list[str]) -> None:
    prints = load_fingerprints(audio_files(roots))
    by_duration = sorted(prints, key=lambda p: prints[p][0])

    pairs = []
    for i, a in enumerate(by_duration):
        for b in by_duration[i + 1:]:
            if prints[b][0] - prints[a][0] > MAX_DURATION_DIFF:
                break  # sorted by duration, so no later file can be close enough
            score = similarity(prints[a][1], prints[b][1])
            if score >= MIN_SIMILARITY:
                pairs.append((a, b, round(score, 3)))

    result = {
        "pairs": [{"a": a, "b": b, "similarity": s} for a, b, s in pairs],
        "groups": groups([(a, b) for a, b, _ in pairs]),
    }
    Path(out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"{len(pairs)} duplicate pairs in {len(result['groups'])} groups -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
