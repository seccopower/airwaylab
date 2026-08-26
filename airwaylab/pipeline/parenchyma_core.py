"""Parenchima oltre la densita' media (nucleo puro).

MLD/LAA-950 riassumono male: la malattia delle piccole vie aeree nell'asma vive
nella DISOMOGENEITA' e nell'ORGANIZZAZIONE della bassa attenuazione, non nella media.
Tre letture interpretabili (niente radiomics opache):

  1. FORMA dell'istogramma HU — media, SD, asimmetria (skewness) e curtosi.
     Descrittori nominati e spiegabili. NB: calcolati sulla COMPONENTE AERATA
     (soglia aria), non sul polmone anatomico.
  2. ETEROGENEITA' regionale — deviazione delle medie di densita' calcolate a
     blocchi. E' disomogeneita' regionale della densita', NON una misura specifica
     di mosaic attenuation o air-trapping (che richiede l'espiratoria).
  3. CLUSTER LAA — distribuzione delle dimensioni dei componenti a bassa attenuazione:
     numero, frazione del cluster maggiore, ed esponente D (Mishima) dalla legge di
     potenza (OLS rango-dimensione, non MLE/xmin). D DESCRIVE la distribuzione
     dimensionale dei cluster; NON e' specifico per enfisema (Gupta non trovo'
     differenze nell'asma) e il sottocampionamento ×3 ne altera la topologia.

CAVEAT: dipendono da soglia HU, kernel/dose e volume inspiratorio → confronti solo a
parita' di protocollo. Su singola inspiratoria la bassa attenuazione non e'
air-trapping. Calcolati sulla griglia di analisi ×3 (nota nel runner).

Puro: numpy + aritmetica. Testato in tests/test_parenchyma_core.py.
"""
import math

import numpy as np


def histogram_shape(hu):
    """Forma dell'istogramma HU del polmone: media, SD, skewness, curtosi (Fisher)."""
    a = np.asarray(hu, dtype=float)
    if a.size < 100:
        return {'mld_hu': None, 'sd_hu': None, 'skewness': None,
                'kurtosis': None, 'n_vox': int(a.size)}
    m = float(a.mean())
    sd = float(a.std())
    if sd < 1e-6:
        skew = kurt = 0.0
    else:
        z = (a - m) / sd
        skew = float((z ** 3).mean())
        kurt = float((z ** 4).mean() - 3.0)
    return {'mld_hu': round(m, 1), 'sd_hu': round(sd, 1),
            'skewness': round(skew, 3), 'kurtosis': round(kurt, 3),
            'n_vox': int(a.size)}


def heterogeneity(block_means, min_blocks=8):
    """Disomogeneita' regionale dalle medie di densita' a blocchi: SD e IQR (HU)."""
    b = np.asarray([x for x in block_means if x is not None], dtype=float)
    if b.size < min_blocks:
        return {'het_sd_hu': None, 'het_iqr_hu': None, 'n_blocks': int(b.size)}
    iqr = float(np.percentile(b, 75) - np.percentile(b, 25))
    return {'het_sd_hu': round(float(b.std()), 1),
            'het_iqr_hu': round(iqr, 1), 'n_blocks': int(b.size)}


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


def cluster_size_stats(sizes, total_laa=None, min_clusters=10):
    """Statistiche della distribuzione delle dimensioni dei cluster LAA.
    sizes: conteggi voxel dei componenti; total_laa: voxel LAA totali (per la
    frazione). D = esponente della legge di potenza (pendenza log-log della
    distribuzione cumulativa: n. cluster >= dimensione)."""
    s = np.asarray([x for x in sizes if x and x > 0], dtype=float)
    n = int(s.size)
    if n == 0:
        return {'n_clusters': 0, 'largest_frac': None, 'D': None, 'r2': None}
    tot = float(total_laa) if total_laa else float(s.sum())
    largest = float(s.max()) / tot if tot else None
    D = r2 = None
    if n >= min_clusters:
        ss = np.sort(s)[::-1]
        x = list(np.log(ss))
        y = list(np.log(np.arange(1, n + 1)))   # rango = n. cluster >= dimensione
        slope, r2v = _linfit(x, y)
        if slope is not None:
            D = round(-slope, 3)
            r2 = round(r2v, 3) if r2v is not None else None
    return {'n_clusters': n,
            'largest_frac': round(largest, 3) if largest is not None else None,
            'D': D, 'r2': r2}
