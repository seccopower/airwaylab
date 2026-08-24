"""Test della morfometria dell'albero (treestats_core). Puro."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "airwaylab", "pipeline"))

from treestats_core import (   # noqa: E402
    box_count_dimension,
    count_summary,
    geometric_sizes,
)


def test_count_summary_terminali_e_generazioni():
    # trachea(0) -> due figli(1); i due figli sono terminali
    br = [
        {'u': 0, 'v': 1, 'gen': 0, 'length': 30.0},
        {'u': 1, 'v': 2, 'gen': 1, 'length': 20.0},
        {'u': 1, 'v': 3, 'gen': 1, 'length': 18.0},
    ]
    s = count_summary(br)
    assert s['n_branches'] == 3
    assert s['n_terminals'] == 2          # v=2 e v=3 non sono padri
    assert s['max_gen'] == 1
    assert s['total_length_mm'] == 68.0
    assert s['count_by_gen'] == {0: 1, 1: 2}


def test_count_summary_vuoto():
    assert count_summary([])['n_branches'] == 0


def test_box_count_linea_dimensione_uno():
    # punti su una retta -> AFD ≈ 1
    P = np.stack([np.linspace(0, 100, 4000), np.zeros(4000), np.zeros(4000)], axis=1)
    r = box_count_dimension(P, [2, 4, 8, 16, 32])
    assert 0.85 <= r['afd'] <= 1.15
    assert r['r2'] >= 0.98


def test_box_count_piano_dimensione_due():
    # griglia su un piano -> AFD ≈ 2
    g = np.linspace(0, 100, 200)
    X, Y = np.meshgrid(g, g)
    P = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)
    r = box_count_dimension(P, [2, 4, 8, 16, 32])
    assert 1.8 <= r['afd'] <= 2.15


def test_box_count_pochi_punti():
    assert box_count_dimension(np.zeros((3, 3)), [2, 4, 8])['afd'] is None


def test_geometric_sizes():
    s = geometric_sizes(200.0, n=6, smallest=2.0)
    assert len(s) == 6 and s[0] == 2.0 and s[-1] <= 200 * 0.25 + 1e-6
    assert all(s[i] < s[i + 1] for i in range(len(s) - 1))
