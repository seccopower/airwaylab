"""Test del nucleo puro arteria/vena (av_core.py). Nessun dato paziente."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "airwaylab", "pipeline"))

from av_core import (aggregate_by_lobe, av_ratio,   # noqa: E402
                     bvn_volumes, radius_for_csa)


def test_radius_for_csa():
    assert abs(radius_for_csa(np.pi * 4) - 2.0) < 1e-9   # area 4pi -> r 2
    assert abs(radius_for_csa(5.0) - np.sqrt(5 / np.pi)) < 1e-12


def test_bvn_volumes_thresholds():
    # raggi locali (mm) di 4 voxel; soglia BV5 -> r<1.2616, BV10 -> r<1.784
    edt = np.array([0.5, 1.0, 1.5, 2.0])
    vox_ml = 1.0                            # ml/voxel leggibile (evita arrotondamenti)
    out = bvn_volumes(edt, vox_ml, csa_list=(5.0, 10.0))
    assert out['tbv_ml'] == 4.0
    assert out['bv5_frac'] == 0.5          # 2 voxel su 4 sotto r5 (0.5,1.0)
    assert out['bv10_frac'] == 0.75        # 3 voxel su 4 sotto r10 (0.5,1.0,1.5)


def test_aggregate_by_lobe_skips_tiny():
    edt = np.array([0.5, 0.6, 2.0, 2.1, 0.4])
    lobe = np.array(['RLL', 'RLL', 'RUL', 'RUL', 'LING'], dtype=object)
    vox_ml = 1.0                            # 1 ml/voxel per un test leggibile
    per = aggregate_by_lobe(edt, lobe, vox_ml, min_ml=2.0)
    assert 'RLL' in per and 'RUL' in per    # 2 ml ciascuno
    assert 'LING' not in per                # 1 voxel = 1 ml < min_ml
    assert per['RLL']['bv5_frac'] == 1.0    # entrambi piccoli
    assert per['RUL']['bv5_frac'] == 0.0    # entrambi grandi


def test_av_ratio():
    assert av_ratio(200.0, 100.0) == 2.0
    assert av_ratio(100.0, 0.0) is None
