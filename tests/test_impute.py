"""Test del nucleo puro dell'esperimento di censura imputazione (impute_core.py)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "airwaylab", "pipeline"))

from impute_core import error_stats, imputed_diameter   # noqa: E402


def test_imputed_diameter_taper():
    # figlio con meta' territorio del genitore, Murray nexp=3: d = 10 * 0.5^(1/3)
    d = imputed_diameter(10.0, 0.5, 3.0)
    assert abs(d - 10.0 * 0.5 ** (1 / 3)) < 1e-9
    assert d < 10.0                      # il taper non allarga


def test_imputed_diameter_clip_parent():
    # frac >= 1 non deve dare un figlio piu' largo del genitore
    assert imputed_diameter(8.0, 1.5, 3.0) == 8.0


def test_imputed_diameter_floor():
    # frazione minuscola -> clip al pavimento numerico
    assert imputed_diameter(5.0, 1e-6, 3.0, dfloor=0.3) == 0.3


def test_error_stats():
    # imputati sistematicamente +10% -> bias 0.1, tutti entro 20%
    pairs = [(10.0, 11.0), (5.0, 5.5), (2.0, 2.2)]
    s = error_stats(pairs)
    assert s['n'] == 3
    assert abs(s['bias_mediano'] - 0.1) < 1e-9
    assert s['entro_20pct'] == 1.0


def test_error_stats_ignora_misurato_nullo():
    s = error_stats([(0.0, 1.0), (4.0, 4.0)])
    assert s['n'] == 1 and s['errore_assoluto_mediano'] == 0.0
