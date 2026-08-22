"""Test del nucleo puro dell'ensemble di incertezza (uncertainty_core.py).
Nessun dato paziente: liste sintetiche di repliche."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "airwaylab", "pipeline"))

from uncertainty_core import (   # noqa: E402
    label_stability,
    quantiles,
    rank_stability,
    variance_share,
    worst_rank_prob,
)


def test_quantiles_ignora_none():
    q = quantiles([1, 2, 3, 4, 5, None], qs=(50,))
    assert q[50] == 3.0
    assert quantiles([None], qs=(50,))[50] is None


def test_worst_rank_prob():
    reps = [
        {'RLL': 0.5, 'RUL': 0.1},
        {'RLL': 0.4, 'RUL': 0.2},
        {'RLL': 0.1, 'RUL': 0.3},   # qui RUL peggiore
    ]
    p = worst_rank_prob(reps, higher_is_worse=True)
    assert p['RLL'] == round(2 / 3, 3)
    assert p['RUL'] == round(1 / 3, 3)


def test_label_stability():
    base = {'RLL': 'A', 'RUL': 'B'}
    reps = [{'RLL': 'A', 'RUL': 'B'}, {'RLL': 'A', 'RUL': 'C'}, {'RLL': 'X', 'RUL': 'B'}]
    s = label_stability(reps, base)
    assert s['RLL'] == round(2 / 3, 3)
    assert s['RUL'] == round(2 / 3, 3)


def test_rank_stability():
    base_order = ['RLL', 'RUL']            # RLL peggiore (valore piu' alto)
    reps = [
        {'RLL': 0.9, 'RUL': 0.1},          # ordine coincide
        {'RLL': 0.8, 'RUL': 0.2},          # ordine coincide
        {'RLL': 0.1, 'RUL': 0.9},          # invertito
    ]
    r = rank_stability(reps, base_order, thresh=0.8)
    assert r['RLL']['frac'] == round(2 / 3, 3)
    assert r['RLL']['stabile'] is False    # 0.667 < 0.8
    r2 = rank_stability(reps[:2], base_order, thresh=0.8)
    assert r2['RLL']['stabile'] is True    # 1.0 >= 0.8


def test_variance_share():
    assert variance_share(2.0, 8.0) == 0.25
    assert variance_share(10.0, 8.0) == 1.0    # clip a 1
    assert variance_share(1.0, 0.0) is None
