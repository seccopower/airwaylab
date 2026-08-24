"""Test del gradiente di pruning vascolare (vascular_gradient_core). Puro."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "airwaylab", "pipeline"))

from vascular_gradient_core import pruning_summary   # noqa: E402


def _shell(d, lung, small, allv):
    return {'d_center': d, 'lung_ml': lung, 'small_ml': small, 'all_ml': allv}


def test_polmone_sano_periferia_ricca():
    # densita' piccoli vasi ALTA in periferia (d piccolo), bassa al centro
    shells = [_shell(2, 100, 8, 12), _shell(7, 100, 7, 12),
              _shell(12, 100, 5, 12), _shell(20, 100, 2, 12), _shell(30, 100, 1, 12)]
    r = pruning_summary(shells, periph_mm=15)
    assert r['pruning_ratio'] > 1            # periferia piu' densa del centro
    assert r['gradient_per_mm'] < 0          # densita' cala con la distanza dalla pleura


def test_pruning_periferia_povera():
    # pruning: periferia impoverita -> densita' periferica bassa, ratio < 1
    shells = [_shell(2, 100, 1, 12), _shell(7, 100, 1, 12),
              _shell(12, 100, 2, 12), _shell(20, 100, 5, 12), _shell(30, 100, 6, 12)]
    r = pruning_summary(shells, periph_mm=15)
    assert r['pruning_ratio'] < 1
    assert r['gradient_per_mm'] > 0


def test_gusci_scartati_se_poco_polmone():
    shells = [_shell(2, 0.2, 1, 2), _shell(10, 100, 5, 10), _shell(20, 100, 4, 10),
              _shell(30, 100, 3, 10)]
    r = pruning_summary(shells, min_lung_ml=1.0)
    assert all(p['d_mm'] != 2.0 for p in r['profile'])   # il guscio da 0.2 ml e' fuori


def test_vuoto():
    r = pruning_summary([])
    assert r['pruning_ratio'] is None and r['profile'] == []
