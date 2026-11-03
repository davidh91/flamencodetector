"""Shared palo word matching for the data scripts."""

import re
import unicodedata

# Broad on purpose: here we only detect "this title names a palo".
# The mapping to our 7 training classes happens later, in labels.py.
PALO_WORDS = [
    "alegria", "bambera", "buleria", "campanillero", "cana", "cantina", "caracoles",
    "cartagenera", "colombiana", "debla", "fandango", "farruca", "garrotin", "granaina",
    "guajira", "jabera", "liviana", "malaguena", "mariana", "martinete", "milonga",
    "minera", "mirabras", "nana", "petenera", "polo", "romera", "rondena", "rumba",
    "saeta", "seguiriya", "serrana", "sevillana", "siguiriya", "solea", "solear",
    "tango", "tanguillo", "taranta", "taranto", "tiento", "tona", "trilla",
    "verdial", "vidalita", "villancico", "zambra", "zapateado",
]
PALO_RE = re.compile(r"\b(" + "|".join(PALO_WORDS) + r")(e?s)?\b")


def normalize(text: str) -> str:
    """Lowercase and strip accents: 'Soleá por Bulerías' -> 'solea por bulerias'."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def has_palo(title: str) -> bool:
    return PALO_RE.search(normalize(title or "")) is not None
