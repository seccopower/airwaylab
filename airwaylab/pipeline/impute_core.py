"""Nucleo puro dell'esperimento di censura sull'imputazione dei diametri.

L'imputazione del modello di flusso (flow_model.assign_d) stima il diametro di un
ramo non misurato dal diametro del genitore e dalla frazione di territorio:
    d = min(d_genitore, max(dfloor, d_genitore * frac^(1/nexp)))
`imputed_diameter` E' quella formula (unica fonte: flow_model la richiama), cosi'
il test di censura misura l'imputazione REALE, non una copia.

`error_stats` riassume l'errore relativo (imputato-misurato)/misurato su una lista
di coppie. Puro: solo aritmetica/numpy. Testato in tests/test_impute.py.
"""
import numpy as np


def imputed_diameter(d_parent, frac, nexp, dfloor=0.3):
    """Diametro imputato di un ramo dal genitore e dalla frazione di territorio.
    Clip: mai sotto dfloor, mai sopra il genitore (il taper non allarga)."""
    d = float(d_parent) * (float(frac) ** (1.0 / float(nexp)))
    d = max(dfloor, d)
    d = min(float(d_parent), d)
    return d


def error_stats(pairs):
    """pairs: lista di (misurato, imputato). Ritorna statistiche dell'errore
    relativo r = (imputato - misurato)/misurato. Ignora coppie con misurato<=0."""
    meas = np.array([m for m, _ in pairs if m and m > 0], dtype=float)
    imp = np.array([i for m, i in pairs if m and m > 0], dtype=float)
    if meas.size == 0:
        return {'n': 0}
    rel = (imp - meas) / meas
    return {
        'n': int(meas.size),
        'bias_mediano': round(float(np.median(rel)), 3),               # sistematico (+ sovrastima)
        'errore_assoluto_mediano': round(float(np.median(np.abs(rel))), 3),
        'p5': round(float(np.percentile(rel, 5)), 3),
        'p95': round(float(np.percentile(rel, 95)), 3),
        'entro_20pct': round(float(np.mean(np.abs(rel) <= 0.20)), 3),   # quota entro +-20%
    }
