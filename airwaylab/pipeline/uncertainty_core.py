"""Nucleo puro delle statistiche dell'ensemble di incertezza (vedi uncertainty.py).

Solo aritmetica su liste di repliche: mediana/intervalli, probabilita' di rango
peggiore, stabilita' della classificazione e del rango, quota di varianza. Nessun
I/O, nessuna dipendenza dal modello. Testato in tests/test_uncertainty.py.
"""
import numpy as np


def quantiles(xs, qs=(5, 50, 95)):
    """Percentili richiesti (default 5/50/95) di una lista, ignorando i None."""
    a = np.asarray([x for x in xs if x is not None], dtype=float)
    if a.size == 0:
        return {q: None for q in qs}
    return {q: round(float(np.percentile(a, q)), 4) for q in qs}


def worst_rank_prob(replicates, higher_is_worse=True):
    """replicates: lista di dict {lobo -> valore}. Ritorna {lobo -> frazione di
    repliche in cui il lobo ha il valore PEGGIORE} (max se higher_is_worse)."""
    lobi = set()
    for r in replicates:
        lobi |= set(r)
    cnt = {lb: 0 for lb in lobi}
    n = 0
    for r in replicates:
        vals = {lb: v for lb, v in r.items() if v is not None}
        if not vals:
            continue
        n += 1
        best = max(vals, key=vals.get) if higher_is_worse else min(vals, key=vals.get)
        cnt[best] += 1
    return {lb: (round(cnt[lb] / n, 3) if n else None) for lb in lobi}


def label_stability(label_replicates, baseline):
    """label_replicates: lista di {lobo -> etichetta}; baseline: {lobo -> etichetta}.
    Ritorna {lobo -> frazione di repliche con etichetta uguale al baseline}."""
    lobi = set(baseline)
    cnt = {lb: 0 for lb in lobi}
    n = len(label_replicates)
    for r in label_replicates:
        for lb in lobi:
            if r.get(lb) == baseline.get(lb):
                cnt[lb] += 1
    return {lb: (round(cnt[lb] / n, 3) if n else None) for lb in lobi}


def rank_stability(replicates, baseline_order, thresh=0.8):
    """Per ogni lobo, frazione di repliche in cui mantiene LO STESSO rango del
    baseline (ordinamento per valore decrescente). replicates: lista di
    {lobo -> valore}; baseline_order: lista di lobi dal peggiore al migliore.
    Ritorna {lobo -> {'frac': f, 'stabile': f>=thresh}}."""
    pos = {lb: i for i, lb in enumerate(baseline_order)}
    cnt = {lb: 0 for lb in pos}
    n = 0
    for r in replicates:
        vals = {lb: v for lb, v in r.items() if v is not None and lb in pos}
        if not vals:
            continue
        n += 1
        order = sorted(vals, key=lambda k: -vals[k])
        rp = {lb: i for i, lb in enumerate(order)}
        for lb in pos:
            if rp.get(lb) == pos[lb]:
                cnt[lb] += 1
    return {lb: {'frac': (round(cnt[lb] / n, 3) if n else None),
                 'stabile': bool(n and cnt[lb] / n >= thresh)} for lb in pos}


def variance_share(var_subset, var_total):
    """Quota (0..1) di varianza spiegata da un sottoinsieme di perturbazioni."""
    if not var_total:
        return None
    return round(max(0.0, min(1.0, var_subset / var_total)), 3)
