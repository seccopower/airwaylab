"""Test dei descrittori parenchimali (parenchyma_core). Puro."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "airwaylab", "pipeline"))

from parenchyma_core import (   # noqa: E402
    cluster_size_stats,
    heterogeneity,
    histogram_shape,
)


def test_histogram_shape_normale():
    rng = np.random.default_rng(0)
    hu = rng.normal(-850, 40, 50000)
    h = histogram_shape(hu)
    assert abs(h['mld_hu'] + 850) < 3
    assert abs(h['skewness']) < 0.1        # gaussiana: skew ~0
    assert abs(h['kurtosis']) < 0.2        # gaussiana: kurtosi ~0


def test_histogram_shape_pochi_voxel():
    assert histogram_shape([1, 2, 3])['mld_hu'] is None


def test_heterogeneity_omogeneo_vs_mosaico():
    omog = heterogeneity([-850] * 20)
    assert omog['het_sd_hu'] == 0.0
    mosaico = heterogeneity([-950, -750] * 10)       # due popolazioni = mosaico
    assert mosaico['het_sd_hu'] > 50


def test_cluster_pochi_e_frazione():
    st = cluster_size_stats([100, 50, 10], total_laa=200)
    assert st['n_clusters'] == 3
    assert abs(st['largest_frac'] - 0.5) < 1e-9      # 100/200
    assert st['D'] is None                            # < min_clusters -> niente fit


def test_cluster_D_legge_di_potenza():
    # dimensioni a legge di potenza -> D finito con buon fit
    sizes = [int(1000 / (k ** 1.5)) + 1 for k in range(1, 40)]
    st = cluster_size_stats(sizes)
    assert st['n_clusters'] == 39 and st['D'] is not None and st['r2'] > 0.9
