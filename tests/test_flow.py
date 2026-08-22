"""Test del nucleo del modello di flusso 1D (flow_core) — esercita il codice
di produzione usato da flow.py, non una copia. Nessun dato paziente."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "airwaylab", "pipeline"))

from flow_core import (mass_error, poiseuille_R, r_completion,   # noqa: E402
                       solve_tree, terminal_pressures)


def test_symmetric_network_matches_analytic():
    """Radice R0 che si biforca in due rami identici R1, ciascuno chiuso da
    una resistenza terminale Re: la rete deve dare R_eq = R0 + (R1+Re)/2,
    flusso 50/50 e pressioni terminali uguali."""
    ch = lambda b: {'0': ['1', '2'], '1': [], '2': []}[b]
    R = {'0': 10.0, '1': 4.0, '2': 4.0}
    R_ext = {'1': 2.0, '2': 2.0}
    Qt = 1.0
    Req, Q = solve_tree('0', ch, R, R_ext, Qt)
    assert abs(Req['0'] - (10.0 + (4.0 + 2.0) / 2)) < 1e-12
    assert abs(Q['1'] - 0.5) < 1e-12 and abs(Q['2'] - 0.5) < 1e-12
    assert mass_error('0', ch, Q, Qt) < 1e-12
    P = terminal_pressures('0', ch, R, R_ext, Q)
    assert abs(P['1'] - P['2']) < 1e-12                 # pressioni terminali uguali


def test_asymmetric_flow_splits_by_conductance():
    """Due rami di resistenza diversa: il flusso si divide inversamente alla
    resistenza equivalente del sottoalbero, e la massa si conserva ovunque."""
    ch = lambda b: {'0': ['1', '2'], '1': ['3', '4'], '2': [],
                    '3': [], '4': []}.get(b, [])
    R = {'0': 1.0, '1': 2.0, '2': 3.0, '3': 5.0, '4': 5.0}
    R_ext = {'2': 1.0, '3': 1.0, '4': 1.0}
    Qt = 2.0
    Req, Q = solve_tree('0', ch, R, R_ext, Qt)
    # ramo 1: 2 + (5+1)||(5+1) = 2+3 = 5 ; ramo 2: 3+1 = 4
    assert abs(Req['1'] - 5.0) < 1e-12 and abs(Req['2'] - 4.0) < 1e-12
    # flusso inverso alle Req: Q1/Q2 = (1/5)/(1/4)
    assert abs(Q['1'] / Q['2'] - (0.2 / 0.25)) < 1e-9
    assert abs(Q['3'] - Q['4']) < 1e-12
    assert mass_error('0', ch, Q, Qt) < 1e-12
    # conservazione esplicita al nodo 1
    assert abs(Q['1'] - (Q['3'] + Q['4'])) < 1e-12


def test_series_resistance_adds():
    """Catena mono-figlio: le resistenze si sommano in serie."""
    ch = lambda b: {'0': ['1'], '1': ['2'], '2': []}.get(b, [])
    R = {'0': 1.0, '1': 2.0, '2': 3.0}
    Req, Q = solve_tree('0', ch, R, {'2': 4.0}, 1.0)
    assert abs(Req['0'] - (1 + 2 + 3 + 4)) < 1e-12
    assert abs(Q['2'] - 1.0) < 1e-12                    # tutto il flusso passa


def test_r_completion_positive_above_dstop():
    """Blocker della review: una foglia piu' larga del diametro acinare deve
    ricevere completamento periferico POSITIVO."""
    assert r_completion(0.6, 0.5, 3.0, 3.0) > 0.0
    assert r_completion(2.0, 0.5, 3.0, 3.0) > 0.0
    # foglia gia' acinare: nessun completamento
    assert r_completion(0.4, 0.5, 3.0, 3.0) == 0.0
    # monotona: piu' piccola la foglia, piu' resistenza periferica residua
    assert r_completion(1.0, 0.5, 3.0, 3.0) > r_completion(3.0, 0.5, 3.0, 3.0)


def test_pruned_topology_increases_Req():
    """Esperimento occlusione (review r3, major #7): potando un ramo dalla rete
    simmetrica con una children-function filtrata — lo stesso meccanismo usato
    da flow.py per i plug — la Req sale da R0+(R1+Re)/2 a R0+(R1+Re) e tutto il
    flusso passa dal ramo residuo."""
    ch_full = lambda b: {'0': ['1', '2'], '1': [], '2': []}[b]
    R = {'0': 10.0, '1': 4.0, '2': 4.0}
    R_ext = {'1': 2.0, '2': 2.0}
    Req_f, _ = solve_tree('0', ch_full, R, R_ext, 1.0)
    blocked = {'2'}
    ch_p = lambda b: [c for c in ch_full(b) if c not in blocked]
    Req_p, Q = solve_tree('0', ch_p, R, R_ext, 1.0)
    assert abs(Req_p['0'] - (10.0 + 4.0 + 2.0)) < 1e-12
    assert Req_p['0'] > Req_f['0']
    assert abs(Q['1'] - 1.0) < 1e-12
    assert mass_error('0', ch_p, Q, 1.0) < 1e-12


def test_poiseuille_scales_as_d_minus_4():
    """Dimezzando il diametro la resistenza sale di 16x (d^-4)."""
    r1 = poiseuille_R(0.01, 0.004)
    r2 = poiseuille_R(0.01, 0.002)
    assert abs(r2 / r1 - 16.0) < 1e-6


def test_pedley_iteration_converges_on_random_tree():
    """Rete profonda con resistenze dipendenti dal flusso (stile Pedley):
    il punto fisso deve convergere e conservare la massa."""
    rng = np.random.RandomState(0)
    # albero binario di 6 livelli
    nodes = ['0']
    ch_map = {}
    frontier = ['0']
    nid = 1
    for _ in range(5):
        nxt = []
        for p in frontier:
            kids = [str(nid), str(nid + 1)]
            nid += 2
            ch_map[p] = kids
            nodes += kids
            nxt += kids
        frontier = nxt
    for leaf in frontier:
        ch_map[leaf] = []
    ch = lambda b: ch_map.get(b, [])
    R0 = {n: 1.0 + rng.rand() for n in nodes}
    R_ext = {n: 1.0 + rng.rand() for n in frontier}
    Qt = 1.0
    Q = {n: Qt / len(frontier) if n in frontier else Qt for n in nodes}
    prev = None
    for _ in range(60):
        # resistenza cresce col flusso (surrogate non lineare)
        R = {n: R0[n] * (1.0 + 0.3 * Q.get(n, 0.0)) for n in nodes}
        Req, Q = solve_tree('0', ch, R, R_ext, Qt)
        if prev and max(abs(Q[n] - prev[n]) for n in nodes) / Qt < 1e-9:
            break
        prev = Q
    assert mass_error('0', ch, Q, Qt) < 1e-9
    P = terminal_pressures('0', ch, R, R_ext, Q)
    pv = list(P.values())
    assert max(pv) - min(pv) < 1e-6 * max(pv)          # pressioni terminali uniformi
