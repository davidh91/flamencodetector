
import pytest

from flamenco.labels import palo_from_title


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        # Invented titles: each one tests one way that palos appear in file names.
        ("Tangos del puerto viejo", "tangos"),
        ("Soleá por Bulerías", "solea_por_bulerias"),
        ("Solea por bulerias (live)", "solea_por_bulerias"),
        ("Bulerías por Soleá de la plaza", "solea_por_bulerias"),
        ("Soleá por Bulería del molino", "solea_por_bulerias"),
        ("Soleares del barrio", "solea"),
        ("Soleá de entrada y soleá grande del río", "solea"),
        ("ALEGRIAS DEL MAR", "alegrias"),
        ("Brisa de la bahía (Alegrías)", "alegrias"),
        ("Fandangos de la sierra", "fandango"),
        ("Fandangos Del Molino Viejo", "fandango"),
        ("Tarantas", "taranta"),
        ("Desde La Mina Se Ve (tarantos)", "taranto"),
        ("Romance Del Olivo (Tanguillos)", "tanguillos"),
        ("Campanas de la torre [Seguiriya]", "siguiriya"),
        ("Siguiriyas de cambio del herrero", "siguiriya"),
        ("Por La Puerta Del Sur (Bulerias por Soleares)", "solea_por_bulerias"),
        ("Por ti (soleá buleria)", "solea_por_bulerias"),
        ("CAMINO SIN VUELTA (SEGUIRILLA)", "siguiriya"),
        ("Que Lejos Está Tu Casa, Soleares", "solea"),
        ("Cantiñas - (Palabras Como Piedras)", "cantinas"),
        ("La Ventana [Rumba-Tango]", None),
        # The palo in brackets wins over a palo word in the song title.
        ("Canto a una serrana (fandang", "fandango"),
        # Cut-off filenames.
        ("Dos Lunas En El Agua (fandan", "fandango"),
        ("Al Pasar Las Horas (Buler", "bulerias"),
        ("Que Un Caballo En La Noche (T", None),
        ("QUE COSAS TIENE LA VIDA [SOLE", "solea"),  # shortest name wins
        ("Que cantes por mi pena (fandango", "fandango"),
        # Cut off after "por": could be "bulerías por soleá", so we cannot tell.
        ("Mírame a la cara (Bulería por S", None),
        ("Romance del camino (bulerias por", None),
        # Two different palos in one file: skip.
        ("Granaína y Malagueña Chica del puerto", None),
        ("Toná del pozo / La Debla del pozo", None),
        ("Solea y caña", None),
        # No palo.
        ("Bajo el puente (Rumba)", "rumba"),
        ("Bajo el puente", None),
        ("", None),
    ],
)
def test_palo_from_title(title, expected):
    assert palo_from_title(title) == expected
