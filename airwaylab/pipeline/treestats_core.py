"""Morfometria dell'albero: conteggi + dimensione frattale (nucleo puro).

"Quanto in profondita' e quanto complesso" arriva l'albero visibile — un indice di
completezza *e* di rimodellamento (nell'asma il conteggio periferico cala). Due
letture:

  1. CONTEGGI — n. rami, n. terminali, conteggio per generazione, lunghezza totale.
  2. AFD (Airway Fractal Dimension) — dimensione frattale via box-counting sullo
     scheletro 3D: per una serie di lati di cella s si contano le celle occupate
     N(s); AFD = pendenza di log N(s) su log(1/s). Un albero piu' ramificato/che
     riempie lo spazio ha AFD piu' alta; il rimodellamento la riduce.

CAVEAT centrale: conteggi e AFD dipendono dalla PROFONDITA' di segmentazione e dal
protocollo (spessore/dose/backend). NON sono confrontabili tra protocolli diversi;
il confronto sensato e' mela-con-mela (stesso backend/versione), es. pre/post dello
stesso paziente. La generazione, inoltre, si gonfia con i rami spuri (vedi tree.py).

Puro: numpy + aritmetica. Testato in tests/test_treestats_core.py.
"""
import math

import numpy as np


def count_summary(branches):
    """Conteggi dalla lista dei rami (ciascuno con u, v, gen, length). Un ramo e'
    TERMINALE se nessun ramo parte dal suo nodo distale (v non e' padre di nessuno)."""
    n = len(branches)
    if n == 0:
        return {'n_branches': 0, 'n_terminals': 0, 'max_gen': None,
                'total_length_mm': 0.0, 'count_by_gen': {}}
    parent_nodes = {b['u'] for b in branches}
    n_term = sum(1 for b in branches if b['v'] not in parent_nodes)
    gens = [b.get('gen', 0) for b in branches]
    by_gen = {}
    for g in gens:
        by_gen[int(g)] = by_gen.get(int(g), 0) + 1
    total_len = sum(float(b.get('length') or 0.0) for b in branches)
    return {'n_branches': n, 'n_terminals': n_term, 'max_gen': int(max(gens)),
            'total_length_mm': round(total_len, 1),
            'count_by_gen': {g: by_gen[g] for g in sorted(by_gen)}}


def _linfit(x, y):
    n = len(x)
    sx, sy = sum(x), sum(y)
    sxx = sum(v * v for v in x)
    sxy = sum(a * b for a, b in zip(x, y))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return None, None
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    ybar = sy / n
    ss_tot = sum((v - ybar) ** 2 for v in y)
    ss_res = sum((y[i] - (a + b * x[i])) ** 2 for i in range(n))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else None
    return b, r2


def box_count_dimension(points_mm, sizes):
    """Dimensione frattale box-counting. points_mm: array Nx3 (mm); sizes: lati di
    cella (mm). Ritorna {afd, r2, n_points, series:[(s, celle occupate)]}."""
    P = np.asarray(points_mm, dtype=float)
    if P.ndim != 2 or P.shape[0] < 8 or len(sizes) < 3:
        return {'afd': None, 'r2': None, 'n_points': int(P.shape[0]) if P.ndim == 2 else 0,
                'series': []}
    P = P - P.min(axis=0)
    xs, ys, series = [], [], []
    for s in sizes:
        cells = np.floor(P / float(s)).astype(np.int64)
        nocc = int(np.unique(cells, axis=0).shape[0])
        series.append((round(float(s), 3), nocc))
        xs.append(math.log(1.0 / s))
        ys.append(math.log(nocc))
    slope, r2 = _linfit(xs, ys)
    return {'afd': round(slope, 3) if slope is not None else None,
            'r2': round(r2, 3) if r2 is not None else None,
            'n_points': int(P.shape[0]), 'series': series}


def geometric_sizes(extent_mm, n=6, smallest=2.0, frac_largest=0.25):
    """Serie geometrica di lati di cella da `smallest` mm fino a `frac_largest`
    dell'estensione (regime utile del box-counting)."""
    largest = max(smallest * 2, extent_mm * frac_largest)
    if largest <= smallest:
        return [smallest]
    ratio = (largest / smallest) ** (1.0 / (n - 1))
    return [round(smallest * ratio ** k, 3) for k in range(n)]
