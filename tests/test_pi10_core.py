"""Test di Pi10 (pi10_core). Puro."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "airwaylab", "pipeline"))

from pi10_core import airway_points, pi10_fit, wall_point   # noqa: E402


def test_wall_point_geometria():
    # lume d=4 (r_in=2), parete 1 -> r_out=3; WA = pi*(9-4)=5pi; Pi = 4pi
    pi_perim, wa = wall_point(4.0, 1.0)
    assert abs(wa - 5 * math.pi) < 1e-9
    assert abs(pi_perim - 4 * math.pi) < 1e-9


def test_airway_points_filtra_qc_e_valori():
    br = [
        {'qc': 'ok', 'd_mean': 4.0, 'wall': 1.0},
        {'qc': 'sotto-risoluzione', 'd_mean': 3.0, 'wall': 0.8},   # scartato
        {'qc': 'ok', 'd_mean': None, 'wall': 1.0},                 # scartato
        {'qc': 'ok', 'd_mean': 5.0, 'wall': 0.0},                  # scartato (parete 0)
    ]
    assert len(airway_points(br)) == 1


def test_pi10_fit_retta_esatta():
    # costruisco punti con sqrtWA = 0.5 + 0.3*Pi  -> pi10 = 0.5 + 3 = 3.5
    pts = [(pi, 0.5 + 0.3 * pi) for pi in range(2, 20)]
    r = pi10_fit(pts)
    assert abs(r['pi10'] - 3.5) < 1e-6
    assert abs(r['slope'] - 0.3) < 1e-6
    assert r['r2'] == 1.0
    assert r['n'] == len(pts)


def test_pi10_troppo_pochi():
    r = pi10_fit([(3.0, 1.0), (4.0, 1.2)])
    assert r['pi10'] is None and r['n'] == 2


def test_pi10_end_to_end_da_branches():
    # 12 vie aeree ok con geometria varia -> Pi10 numerico finito
    br = [{'qc': 'ok', 'd_mean': 2.0 + 0.5 * k, 'wall': 0.4 + 0.05 * k}
          for k in range(12)]
    r = pi10_fit(airway_points(br))
    assert r['pi10'] is not None and r['n'] == 12
