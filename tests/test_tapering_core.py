"""Test del tapering (tapering_core). Puro."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "airwaylab", "pipeline"))

from tapering_core import (   # noqa: E402
    diameter_ratio_summary,
    taper_gradient,
    tapering_summary,
)


def test_ratio_albero_sano():
    # rapporti ~0.8 (rastremazione normale) -> nessuno "senza taper"
    r = diameter_ratio_summary([0.8] * 10)
    assert abs(r['taper_ratio_med'] - 0.8) < 1e-9
    assert r['frac_no_taper'] == 0.0 and r['n_pairs'] == 10


def test_ratio_bronchiectasie():
    # rami che restano larghi (ratio ~1.0) -> tapering perso in molti
    r = diameter_ratio_summary([1.0, 0.98, 1.05, 0.95, 0.99, 1.02])
    assert r['frac_no_taper'] >= 0.8


def test_ratio_troppo_pochi():
    assert diameter_ratio_summary([0.8, 0.8])['taper_ratio_med'] is None


def test_gradiente_decadimento_noto():
    # d = 10*exp(-0.02*L) -> b = -0.02/mm ; rate = (1-e^{-0.2})*100 ≈ 18.13 %/cm
    pts = [(L, 10.0 * math.exp(-0.02 * L)) for L in range(0, 120, 5)]
    g = taper_gradient(pts)
    assert abs(g['slope_per_mm'] - (-0.02)) < 1e-6
    assert abs(g['taper_rate_pct_per_cm'] - (1 - math.exp(-0.2)) * 100) < 1e-2
    assert g['r2'] == 1.0


def test_gradiente_nessuna_rastremazione():
    # calibro costante -> rate ~0
    pts = [(L, 5.0) for L in range(0, 100, 5)]
    g = taper_gradient(pts)
    assert abs(g['taper_rate_pct_per_cm']) < 1e-6


def test_summary_unisce():
    s = tapering_summary([0.8] * 8, [(L, 10 * math.exp(-0.02 * L)) for L in range(0, 100, 5)])
    assert s['taper_ratio_med'] is not None and s['taper_rate_pct_per_cm'] is not None
