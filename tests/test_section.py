"""Smoke tests on a synthetic tube phantom — no patient data required.

Builds a digital chest-like block (parenchyma at -850 HU) containing a
straight air-filled tube (lumen -1000 HU) with a soft-tissue wall (0 HU),
then checks that the half-max section analysis recovers the ground-truth
lumen diameter and flags the wall sectors as valid.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "airwaylab", "pipeline"))

from lumen import analyze_section, pca_tangent  # noqa: E402

ISO = 0.7  # mm


def make_tube(lumen_d_mm, wall_mm, shape=(60, 80, 80)):
    """Axial tube along z: lumen -1000, wall 0, parenchyma -850."""
    vol = np.full(shape, -850.0, dtype=np.float32)
    zc, yc, xc = shape[0] // 2, shape[1] // 2, shape[2] // 2
    yy, xx = np.meshgrid(np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
    r_mm = np.hypot((yy - yc) * ISO, (xx - xc) * ISO)
    r_lum = lumen_d_mm / 2
    for z in range(shape[0]):
        sl = vol[z]
        sl[r_mm <= r_lum + wall_mm] = 0.0
        sl[r_mm <= r_lum] = -1000.0
    return vol, np.array([zc, yc, xc], dtype=float)


def test_lumen_diameter_recovery():
    for d_true in (4.0, 8.0, 14.0):
        vol, center = make_tube(d_true, wall_mm=1.5)
        t = np.array([1.0, 0.0, 0.0])  # tube axis = z (zyx order)
        sec = analyze_section(vol, center, t, r_est_mm=d_true / 2, iso=ISO)
        assert sec is not None, f"no section for d={d_true}"
        err = abs(sec["d_eq"] - d_true)
        assert err < 0.8, f"d_true={d_true} measured={sec['d_eq']:.2f}"
        assert sec["quality"]["ax_ratio"] < 1.2


def test_wall_measured_and_sectors_valid():
    vol, center = make_tube(8.0, wall_mm=1.5)
    t = np.array([1.0, 0.0, 0.0])
    sec = analyze_section(vol, center, t, r_est_mm=4.0, iso=ISO)
    assert sec is not None and sec["wall_med"] is not None
    assert 0.8 < sec["wall_med"] < 2.5, f"wall={sec['wall_med']}"
    assert sec["wall_valid_frac"] > 0.8


def test_oblique_cut_flagged():
    # NOTE: the moment-based ax_ratio underestimates moderate obliquity
    # (a ~50 deg cut of a circular tube reads ~1.25, not the geometric 1.56),
    # which is why pipeline QC combines it with multi-point consistency.
    # A strongly oblique cut must fail or be clearly non-circular:
    vol, center = make_tube(8.0, wall_mm=1.5)
    t = np.array([1.0, 2.1, 0.0])
    t = t / np.linalg.norm(t)  # ~65 degrees off-axis
    sec = analyze_section(vol, center, t, r_est_mm=4.0, iso=ISO)
    assert sec is None or sec["quality"]["ax_ratio"] > 1.3


def test_pca_tangent_direction():
    path = np.stack([np.arange(20.0), np.zeros(20), np.zeros(20)], axis=1)
    t = pca_tangent(path, 10)
    assert abs(abs(t[0]) - 1.0) < 1e-6
