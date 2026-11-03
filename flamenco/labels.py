"""Map raw palo text (from data/labels.csv) to one canonical palo.

    python -m flamenco.labels data/labels.csv     # print counts per palo

Rules:
  - If the text has a palo in (parentheses) or [brackets], only that part counts:
    "Canto a una serrana (fandango)" -> fandango, not serrana.
  - Cut-off filenames like "Tres Faroles Encendidos (fandan" match on prefix.
  - Text with two different palos ("Granaína y Malagueña", "Toná / Debla") -> None.
"""

import csv
import re
import sys
from collections import defaultdict

from flamenco.palos import normalize

# Order matters: compound and look-alike names first ("solea por bulerias"
# before "solea" and "bulerias", "taranto" before "taranta").
PALO_PATTERNS = [
    ("solea_por_bulerias", r"solea(res)?( por)? bulerias?|bulerias? por solea(s|res)?"),
    ("taranto", r"tarantos?"),
    ("taranta", r"tarantas?"),
    ("bulerias", r"bulerias?"),
    ("solea", r"soleas?|soleares|solear"),
    ("alegrias", r"alegrias?"),
    ("cantinas", r"cantinas?"),
    ("tanguillos", r"tanguillos?"),
    ("tangos", r"tangos?"),
    ("tientos", r"tientos?"),
    ("fandango", r"fandangos?"),
    ("siguiriya", r"s[ie]guiri(y|ll)as?|seguidillas? gitanas?"),
    ("malaguena", r"malaguenas?"),
    ("granaina", r"granainas?"),
    ("minera", r"mineras?"),
    ("cartagenera", r"cartageneras?"),
    ("rondena", r"rondenas?"),
    ("verdiales", r"verdiales?"),
    ("petenera", r"peteneras?"),
    ("farruca", r"farrucas?"),
    ("garrotin", r"garrotin"),
    ("guajira", r"guajiras?"),
    ("colombiana", r"colombianas?"),
    ("rumba", r"rumbas?"),
    ("sevillanas", r"sevillanas?"),
    ("zambra", r"zambras?"),
    ("zapateado", r"zapateados?"),
    ("martinete", r"martinetes?"),
    ("tona", r"tonas?"),
    ("debla", r"deblas?"),
    ("saeta", r"saetas?"),
    ("nana", r"nanas?"),
    ("cana", r"canas?"),
    ("polo", r"polos?"),
    ("serrana", r"serranas?"),
    ("mirabras", r"mirabras"),
    ("caracoles", r"caracoles"),
    ("romera", r"romeras?"),
    ("bamberas", r"bamberas?"),
    ("milonga", r"milongas?"),
    ("vidalita", r"vidalitas?"),
    ("villancico", r"villancicos?"),
    ("campanilleros", r"campanilleros?"),
    ("liviana", r"livianas?"),
    ("trilla", r"trillas?"),
    ("cabal", r"cabales?"),
    ("romance", r"romances?"),
]
# The palos we train on. Others (taranta, solea_por_bulerias, ...) have too few
# tracks, or come mostly from one source, for now.
CLASSES = ["bulerias", "fandango", "solea", "tangos", "siguiriya", "alegrias"]

COMPILED = [(name, re.compile(rf"\b(?:{pattern})\b")) for name, pattern in PALO_PATTERNS]

BRACKETS_RE = re.compile(r"[(\[]([^)\]]*)[)\]]?\s*$")  # last (...) or [...], may be unclosed
CUT_AFTER_POR_RE = re.compile(r"\bpor(\s+\w{0,4})?$")  # "... por" or "... por s"
MIN_PREFIX = 4  # "(fandan" -> fandango, but "(T" is too short to guess


def palos_in(text: str) -> set[str]:
    found = set()
    for name, pattern in COMPILED:
        if pattern.search(text):
            found.add(name)
            text = pattern.sub(" ", text)  # "solea por bulerias" must not also count as "solea"
    return found


def palo_from_title(title: str) -> str | None:
    """Return the canonical palo, or None if there is no palo or more than one."""
    text = normalize(title or "")
    if bracket := BRACKETS_RE.search(text):
        inside = bracket[1].strip()
        cut_off = not text.rstrip().endswith((")", "]"))
        if cut_off and CUT_AFTER_POR_RE.search(inside):
            return None  # "(bulerias por" may be "bulerias por solea"
        found = palos_in(inside)
        if not found and len(inside) >= MIN_PREFIX and cut_off:
            # Cut-off filename: match the start of a palo name. "sole" fits both
            # solea and solea_por_bulerias; the shortest name is the safest guess.
            prefix = inside.replace(" ", "_")
            matches = [name for name, _ in PALO_PATTERNS if name.startswith(prefix)]
            found = {min(matches, key=len)} if matches else set()
        if found:
            return found.pop() if len(found) == 1 else None
    found = palos_in(text)
    return found.pop() if len(found) == 1 else None


def main(labels_csv: str) -> None:
    tracks = defaultdict(int)
    albums = defaultdict(set)
    artists = defaultdict(set)
    with open(labels_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            palo = palo_from_title(row["palo_text"]) if row["palo_text"] else None
            tracks[palo] += 1
            albums[palo].add((row["collection"], row["folder"]))
            artists[palo].add(row["artist"] or row["folder"].split("/")[0])

    print(f"{'palo':20} {'tracks':>6} {'albums':>6} {'artists':>7}")
    for palo, n in sorted(tracks.items(), key=lambda kv: -kv[1]):
        print(f"{palo or '(none)':20} {n:>6} {len(albums[palo]):>6} {len(artists[palo]):>7}")


if __name__ == "__main__":
    main(sys.argv[1])
