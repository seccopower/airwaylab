"""Test dei biomarcatori opportunistici (bodycomp_core). Puro."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "airwaylab", "pipeline"))

from bodycomp_core import (   # noqa: E402
    bodycomp_summary,
    bone_summary,
    fat_summary,
    muscle_summary,
)


def test_bone_normale_e_basso():
    ok = bone_summary({'T8': 160.0, 'T9': 150.0, 'T10': 145.0})
    assert ok['mean_hu'] == round((160 + 150 + 145) / 3, 1)
    assert ok['min_hu'] == 145.0 and ok['low_flag'] is False
    low = bone_summary({'T8': 90.0, 'T9': 130.0})
    assert low['low_flag'] is True          # minimo 90 < 110


def test_bone_vuoto():
    b = bone_summary({'T8': None})
    assert b['n'] == 0 and b['low_flag'] is None


def test_muscle():
    m = muscle_summary(n_vox=100000, mean_hu=35.0, vox_ml=0.001)
    assert m['muscle_ml'] == 100.0 and m['muscle_hu'] == 35.0
    assert muscle_summary(0, None, 0.001)['muscle_ml'] is None


def test_fat_rapporto():
    f = fat_summary(sat_vox=200000, vat_vox=100000, vox_ml=0.001)
    assert f['sat_ml'] == 200.0 and f['vat_ml'] == 100.0
    assert abs(f['vat_sat_ratio'] - 0.5) < 1e-9


def test_summary_combina():
    s = bodycomp_summary({'T8': 150.0}, (50000, 30.0, 0.001), (100000, 50000, 0.001))
    assert s['bone']['mean_hu'] == 150.0
    assert s['muscle']['muscle_ml'] == 50.0
    assert s['fat']['vat_sat_ratio'] == 0.5
