"""Test del nucleo puro della discordanza regionale (discordance.py).
Nessun dato paziente: strutture sintetiche."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "airwaylab", "pipeline"))

from discordance import (classify_lobe, lobe_of,   # noqa: E402
                         regional_summary)


def test_lobe_of_walks_to_lobar_ancestor():
    by_id = {
        'tr': {'aid': 'TRACHEA'}, 'rmb': {'aid': 'RMB'},
        'rll': {'aid': 'RLL'}, 'b9': {'aid': None}, 'b9b': {'aid': None},
    }
    parent = {'rmb': 'tr', 'rll': 'rmb', 'b9': 'rll', 'b9b': 'b9'}
    assert lobe_of('b9b', by_id, parent) == 'RLL'      # risale fino al lobo
    assert lobe_of('rll', by_id, parent) == 'RLL'
    assert lobe_of('tr', by_id, parent) == 'CENTRAL'   # trachea: nessun lobo
    assert lobe_of('assente', by_id, parent) == 'CENTRAL'


def test_regional_summary_separates_two_axes():
    delta_by_lobe = {
        'RLL': np.array([12.0, 15.0, 20.0, 2.0]),   # 3/4 > 10 mm
        'RUL': np.array([1.0, 2.0, 3.0, 4.0]),      # 0/4 > 10 mm
    }
    ba_by_lobe = {
        'RLL': [0.8, 0.9, 0.7],                      # nessun BA>1
        'RUL': [1.2, 1.5, 0.9, 1.1],                # 3/4 BA>1
    }
    s = regional_summary(delta_by_lobe, ba_by_lobe)
    assert s['RLL']['mismatch_frac'] == 0.75 and s['RLL']['ba_gt1_frac'] == 0.0
    assert s['RUL']['mismatch_frac'] == 0.0 and s['RUL']['ba_gt1_frac'] == 0.75
    assert s['RLL']['delta_med_mm'] == 13.5
    assert s['RUL']['n_voxel'] == 4 and s['RUL']['n_ba'] == 4


def test_regional_summary_handles_missing_side():
    s = regional_summary({'LUL': np.array([5.0])}, {})   # lobo senza BA
    assert s['LUL']['ba_med'] is None and s['LUL']['ba_gt1_frac'] is None
    assert s['LUL']['mismatch_frac'] == 0.0


def test_classify_lobe_decomposition():
    # mismatch di distanza dominante -> via aerea non rappresentata (NON occlusione)
    r = classify_lobe({'mismatch_frac': 0.7, 'ba_gt1_frac': 0.1})
    assert r['prevalenza'] == 'via aerea non rappresentata'
    assert r['mismatch_idx'] == 0.7 and 'occlus' not in r['prevalenza']
    # BA>1 dominante (etichetta neutra, non 'dilatazione'/'pruning')
    assert classify_lobe({'mismatch_frac': 0.1, 'ba_gt1_frac': 0.7})['prevalenza'] == 'rapporto bronco-arteria elevato'
    # entrambe alte
    assert classify_lobe({'mismatch_frac': 0.6, 'ba_gt1_frac': 0.6})['prevalenza'] == 'mista'
    # nessuna
    assert classify_lobe({'mismatch_frac': 0.1, 'ba_gt1_frac': 0.1})['prevalenza'] == 'nessuna discordanza elevata'
    # dati assenti
    assert classify_lobe({'mismatch_frac': None, 'ba_gt1_frac': None})['prevalenza'] == 'dati insufficienti'
