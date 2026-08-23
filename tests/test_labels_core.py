"""Test del nucleo puro per l'individuazione dei bronchi principali (labels_core.py)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "airwaylab", "pipeline"))

from labels_core import choose_mains   # noqa: E402


def test_caso_pulito():
    # carena a gen1: due principali puri, sottoalberi grandi
    info = {
        'RMB': {'depth': 1, 'n_sub': 90, 'n_left': 2, 'n_right': 88},
        'LMB': {'depth': 1, 'n_sub': 85, 'n_left': 83, 'n_right': 2},
    }
    assert choose_mains(info) == ('RMB', 'LMB')


def test_scarta_moncone_e_trova_carena_profonda():
    # scenario casoDAS: figlio-moncone a sinistra (n_sub=1), tutto sotto un ramo
    # misto (br001), i VERI principali sono piu' in basso.
    info = {
        'br001': {'depth': 1, 'n_sub': 195, 'n_left': 95, 'n_right': 100},  # misto: non puro
        'br_stub': {'depth': 1, 'n_sub': 1, 'n_left': 1, 'n_right': 0},      # moncone: troppo piccolo
        'RMB_true': {'depth': 2, 'n_sub': 100, 'n_left': 3, 'n_right': 97},  # vero destro
        'LMB_true': {'depth': 2, 'n_sub': 92, 'n_left': 90, 'n_right': 2},   # vero sinistro
    }
    rmb, lmb = choose_mains(info)
    assert rmb == 'RMB_true' and lmb == 'LMB_true'


def test_preferisce_il_piu_prossimale():
    info = {
        'R_shallow': {'depth': 2, 'n_sub': 40, 'n_left': 1, 'n_right': 39},
        'R_deep': {'depth': 5, 'n_sub': 60, 'n_left': 1, 'n_right': 59},
        'L': {'depth': 2, 'n_sub': 40, 'n_left': 39, 'n_right': 1},
    }
    rmb, lmb = choose_mains(info)
    assert rmb == 'R_shallow' and lmb == 'L'


def test_nessun_candidato():
    info = {'a': {'depth': 1, 'n_sub': 3, 'n_left': 2, 'n_right': 1}}
    assert choose_mains(info) == (None, None)
