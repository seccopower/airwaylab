"""Test di Pi10 (pi10_core). Puro."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "airwaylab", "pipeline"))

from pi10_core import (   # noqa: E402
    airway_points,
    pi10_bootstrap,
    pi10_fit,
    pi10_loo,
    pi10_summary,
    wall_point,
)


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


# --- diagnostica -----------------------------------------------------------

def test_range_e_interpolazione():
    # Pi da 2 a 19: 10 mm e' DENTRO -> non estrapolazione
    pts = [(pi, 0.5 + 0.3 * pi) for pi in range(2, 20)]
    r = pi10_fit(pts)
    assert r['pi_min'] == 2.0 and r['pi_max'] == 19.0
    assert r['extrapolation'] is False


def test_estrapolazione_quando_target_fuori_range():
    # tutti i perimetri < 10 mm -> Pi10 a 10 e' un'estrapolazione (ma calcolabile)
    pts = [(2.0 + 0.5 * k, 0.5 + 0.3 * (2.0 + 0.5 * k)) for k in range(15)]  # Pi 2..9
    r = pi10_fit(pts)
    assert r['pi_max'] < 10.0
    assert r['extrapolation'] is True
    assert r['pi10'] is not None                    # estrapolato, non None
    assert r['n_below_target'] == r['n']            # tutti sotto i 10 mm


def test_bootstrap_deterministico_e_stretto_su_retta_esatta():
    pts = [(pi, 0.5 + 0.3 * pi) for pi in range(2, 20)]
    a = pi10_bootstrap(pts)
    b = pi10_bootstrap(pts)
    assert a == b                                   # stesso seed -> stessa CI
    # su una retta esatta ogni ricampionamento da' la stessa retta -> CI degenere a 3.5
    assert abs(a['ci_lo'] - 3.5) < 1e-6 and abs(a['ci_hi'] - 3.5) < 1e-6


def test_bootstrap_none_se_pochi():
    assert pi10_bootstrap([(3.0, 1.0), (4.0, 1.2)])['ci_lo'] is None


def test_loo_zero_su_retta_esatta():
    # togliendo un punto da una retta perfetta Pi10 non si muove
    pts = [(pi, 0.5 + 0.3 * pi) for pi in range(2, 20)]
    loo = pi10_loo(pts)
    assert abs(loo['loo_delta_max']) < 1e-6


def test_loo_intercetta_il_ramo_influente():
    # un outlier lontano sposta Pi10 quando lo si toglie -> delta non banale
    pts = [(pi, 0.5 + 0.3 * pi) for pi in range(2, 20)]
    pts.append((30.0, 50.0))                        # outlier ad alto leverage
    loo = pi10_loo(pts)
    assert abs(loo['loo_delta_max']) > 0.1


def test_summary_ha_i_sottoblocchi():
    pts = [(pi, 0.5 + 0.3 * pi) for pi in range(2, 20)]
    s = pi10_summary(pts)
    assert 'ci95' in s and 'loo' in s
    assert s['pi10'] is not None
    assert s['ci95']['n_boot'] > 0
