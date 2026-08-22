"""Test del modulo puro section_metrics (Step 1: metriche di maschera sulla
sezione perpendicolare). Nessun dato paziente: fantocci sintetici."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "airwaylab", "pipeline"))

from section_metrics import (mask_ray_radii, mask_section,   # noqa: E402
                             overshoot_fraction)

T_Z = np.array([1.0, 0.0, 0.0])   # tangente lungo z (array zyx) -> piano (x, y)


def _cylinder(shape_zyx, iso, cy_mm, cx_mm, a_mm, b_mm):
    """Cilindro lungo z con sezione ellittica (semiassi a,b in mm su x,y)."""
    Nz, Ny, Nx = shape_zyx
    yy, xx = np.mgrid[0:Ny, 0:Nx].astype(float)
    ell = (((xx - cx_mm / iso) * iso / a_mm) ** 2
           + ((yy - cy_mm / iso) * iso / b_mm) ** 2) <= 1.0
    return np.repeat(ell[None], Nz, axis=0).astype(np.float32)


def test_circular_section_area_equivalent_and_feret():
    """Sezione circolare: d_eq, d_min e d_maj coincidono col diametro vero."""
    iso = 0.3
    D = 16.0
    m = _cylinder((5, 140, 140), iso, cy_mm=21.0, cx_mm=21.0, a_mm=D / 2, b_mm=D / 2)
    center = np.array([2.0, 21.0 / iso, 21.0 / iso])
    s = mask_section(m, center, T_Z, iso, r_est_mm=D / 2)
    assert abs(s['d_eq'] - D) < 1.0
    assert abs(s['d_min'] - D) < 1.5 and abs(s['d_maj'] - D) < 1.5


def test_elliptical_section_matches_axes_and_area_eq():
    """Sezione ellittica 26x12 mm (trachea a sciabola sintetica): il diametro
    AREA-equivalente e' sqrt(26*12)=17.66 mm, ben diverso dalla dimensione
    minima 12 mm — la distinzione al centro dello Step 1."""
    iso = 0.3
    maj, mnr = 26.0, 12.0
    m = _cylinder((5, 200, 200), iso, cy_mm=30.0, cx_mm=30.0,
                  a_mm=maj / 2, b_mm=mnr / 2)
    center = np.array([2.0, 30.0 / iso, 30.0 / iso])
    s = mask_section(m, center, T_Z, iso, r_est_mm=maj / 2)
    assert abs(s['d_eq'] - np.sqrt(maj * mnr)) < 1.0     # ~17.66
    assert abs(s['d_min'] - mnr) < 1.5                   # ~12
    assert abs(s['d_maj'] - maj) < 1.5                   # ~26
    # il d_eq NON deve essere scambiato per la dimensione minima
    assert s['d_eq'] - s['d_min'] > 3.0


def test_connected_component_contains_center():
    """Due lumi vicini: l'area conta SOLO la componente che contiene il centro,
    non la somma (evita di gonfiare con una figlia o struttura adiacente)."""
    iso = 0.3
    D = 10.0
    a = _cylinder((5, 200, 200), iso, cy_mm=30.0, cx_mm=22.0, a_mm=D / 2, b_mm=D / 2)
    b = _cylinder((5, 200, 200), iso, cy_mm=30.0, cx_mm=40.0, a_mm=D / 2, b_mm=D / 2)
    m = np.clip(a + b, 0, 1)
    center = np.array([2.0, 30.0 / iso, 22.0 / iso])     # dentro il primo lume
    s = mask_section(m, center, T_Z, iso, r_est_mm=D / 2)
    assert abs(s['d_eq'] - D) < 1.2                      # un solo lume, non due


def test_mask_ray_radii_on_circle():
    """I raggi alla parete della maschera su un cerchio sono ~ il raggio vero."""
    iso = 0.3
    R = 8.0
    m = _cylinder((5, 160, 160), iso, cy_mm=24.0, cx_mm=24.0, a_mm=R, b_mm=R)
    from section_metrics import perp_basis
    u, v = perp_basis(T_Z)
    center = np.array([2.0, 24.0 / iso, 24.0 / iso])
    rr = mask_ray_radii(m, center, u, v, iso, rmax_mm=16.0)
    assert abs(float(np.median(rr)) - R) < 0.5


def test_overshoot_fraction_discriminates():
    """Underfill globale: CT piu' larga in tutte le direzioni -> frazione ~1.
    Concordante: nessun overshoot -> 0. Fuga localizzata: pochi angoli."""
    n = 64
    ct = np.full(n, 10.0); mk = np.full(n, 8.0)
    assert overshoot_fraction(ct, mk) == 1.0
    assert overshoot_fraction(np.full(n, 6.0), mk) == 0.0
    loc = np.full(n, 6.0); loc[:4] = 20.0                # 4/64 angoli in fuga
    assert abs(overshoot_fraction(loc, mk) - 4.0 / n) < 1e-9
