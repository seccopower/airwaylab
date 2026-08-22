"""Nucleo puro e testabile del modello di flusso 1D (vedi flow.py).

Nessun I/O, nessuna dipendenza dall'albero reale: solo la fisica della rete,
cosi' i test in tests/test_flow.py esercitano ESATTAMENTE il codice che flow.py
usa in produzione (niente costanti duplicate).
"""
import numpy as np

MU_AIR = 1.81e-5   # Pa*s


def poiseuille_R(L_m, d_m, mu=MU_AIR):
    """Resistenza di Poiseuille di un condotto (Pa*s/m^3)."""
    return 128.0 * mu * L_m / (np.pi * d_m ** 4)


def r_completion(d_leaf_mm, dstop_mm, Ld, nexp, mu=MU_AIR):
    """Resistenza dell'albero di conduzione a valle di una foglia, modellato
    come generazioni dicotomiche (d scala di 2^(-1/nexp), L=Ld*d, 2^g condotti
    in parallelo) finche' il genitore e' piu' largo del diametro acinare dstop.
    Positiva per ogni d_leaf > dstop; nulla se la foglia e' gia' acinare."""
    if d_leaf_mm <= dstop_mm:
        return 0.0
    h = 2.0 ** (-1.0 / nexp)
    r, d, par, g = 0.0, float(d_leaf_mm), 1.0, 0
    while d > dstop_mm and g < 60:
        d *= h
        par *= 2.0
        r += poiseuille_R(Ld * d * 1e-3, d * 1e-3, mu) / par
        g += 1
    return r


def solve_tree(root, children_of, R, R_ext, Q_tot):
    """Soluzione LINEARE di una rete ad albero per riduzione serie/parallelo.

    root        id del nodo radice
    children_of funzione id -> lista di id figli ([] per le foglie)
    R           dict id -> resistenza del ramo
    R_ext       dict id (foglia) -> resistenza terminale di completamento
    Q_tot       portata totale immessa nella radice

    Ritorna (Req, Q): impedenza equivalente e portata per ogni nodo. La caduta
    su un ramo condiviso e' R*Q(nodo) con Q(nodo)=somma dei figli (verificato
    dalla conservazione della massa). Con resistenze dipendenti dal flusso
    (Pedley) il chiamante itera questa soluzione fino a convergenza.
    """
    Req = {}

    def reduce(b):
        ks = children_of(b)
        if not ks:
            Req[b] = R[b] + R_ext.get(b, 0.0)
        else:
            Req[b] = R[b] + 1.0 / sum(1.0 / reduce(c) for c in ks)
        return Req[b]

    reduce(root)
    Q = {root: Q_tot}

    def split(b):
        ks = children_of(b)
        if not ks:
            return
        g = [1.0 / Req[c] for c in ks]
        gs = sum(g)
        for c, gi in zip(ks, g):
            Q[c] = Q[b] * gi / gs
            split(c)

    split(root)
    return Req, Q


def mass_error(root, children_of, Q, Q_tot):
    """Massimo errore relativo di conservazione della massa ai nodi interni."""
    err = 0.0
    stack = [root]
    while stack:
        b = stack.pop()
        ks = children_of(b)
        if ks:
            err = max(err, abs(Q[b] - sum(Q[c] for c in ks)) / Q_tot)
            stack.extend(ks)
    return err


def terminal_pressures(root, children_of, R, R_ext, Q):
    """Pressione a monte di ogni compartimento terminale (deve essere uniforme
    nella soluzione a pressione alveolare costante). Ritorna dict foglia->Pa."""
    P = {}

    def walk(b, p_in):
        p_out = p_in + R[b] * Q[b]
        ks = children_of(b)
        if not ks:
            P[b] = p_out + R_ext.get(b, 0.0) * Q[b]
        else:
            for c in ks:
                walk(c, p_out)

    walk(root, 0.0)
    return P
