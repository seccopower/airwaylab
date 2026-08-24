"""Tapering delle vie aeree — rastremazione del calibro (nucleo puro).

Un albero bronchiale sano si assottiglia andando in periferia. La perdita di
rastremazione (rami che restano larghi a valle) e' il segno delle bronchiectasie;
alterazioni del tapering compaiono anche nell'asma. Due letture complementari, sui
soli rami misurabili (`qc == 'ok'`):

  1. RAPPORTO figlio/genitore del lume (locale): d_figlio/d_genitore a ogni
     biforcazione. Sano ~0.78-0.85 (Murray simmetrico: 2^(-1/3) ≈ 0.79). Vicino o
     sopra 1 = tapering perso. Robusto e scala-indipendente.

  2. GRADIENTE globale (per soggetto): regressione log-lineare del calibro sulla
     distanza cumulativa lungo l'albero, ln(d) = a + b·L. La pendenza b (per mm) da'
     il tasso di rastremazione, riportato come **% di riduzione del calibro per cm**:
     rate = (1 − e^{b·10})·100. Rate alto = si assottiglia bene; rate → 0 = non si
     assottiglia.

NB: entrambi dipendono da quali rami sono misurabili (protocollo/inspirazione); i
riferimenti di normalita' vanno stabiliti sulla coorte, a parita' di protocollo.

Puro: solo aritmetica. Testato in tests/test_tapering_core.py.
"""
import math

NO_TAPER_RATIO = 0.9      # rapporto figlio/genitore oltre cui il tapering e' "perso"
MIN_PAIRS = 6
MIN_POINTS = 8


def diameter_ratio_summary(ratios, loss_thresh=NO_TAPER_RATIO, min_pairs=MIN_PAIRS):
    """ratios: lista di d_figlio/d_genitore (>0). Ritorna mediana, frazione senza
    rastremazione (ratio > loss_thresh) e n. None se troppo pochi."""
    rs = [r for r in ratios if r and r > 0]
    n = len(rs)
    if n < min_pairs:
        return {'taper_ratio_med': None, 'frac_no_taper': None, 'n_pairs': n}
    rs_sorted = sorted(rs)
    mid = n // 2
    med = rs_sorted[mid] if n % 2 else 0.5 * (rs_sorted[mid - 1] + rs_sorted[mid])
    frac_loss = sum(1 for r in rs if r > loss_thresh) / n
    return {'taper_ratio_med': round(med, 3),
            'frac_no_taper': round(frac_loss, 3), 'n_pairs': n}


def taper_gradient(points, min_n=MIN_POINTS):
    """points: lista di (L_mm, d_mm) — distanza cumulativa dalla carena e calibro
    del ramo (d>0). Fit ln(d) = a + b·L; ritorna il tasso di rastremazione come
    % di riduzione del calibro per cm, la pendenza per mm, R² e n."""
    pts = [(L, math.log(d)) for (L, d) in points if d and d > 0 and L is not None]
    n = len(pts)
    if n < min_n:
        return {'taper_rate_pct_per_cm': None, 'slope_per_mm': None,
                'r2': None, 'n': n}
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return {'taper_rate_pct_per_cm': None, 'slope_per_mm': None,
                'r2': None, 'n': n}
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    ybar = sy / n
    ss_tot = sum((p[1] - ybar) ** 2 for p in pts)
    ss_res = sum((p[1] - (a + b * p[0])) ** 2 for p in pts)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else None
    rate = (1.0 - math.exp(b * 10.0)) * 100.0     # % riduzione calibro per cm
    return {'taper_rate_pct_per_cm': round(rate, 2),
            'slope_per_mm': round(b, 5),
            'r2': round(r2, 3) if r2 is not None else None, 'n': n}


def tapering_summary(ratios, points, **kw):
    """Combina le due letture in un dict unico."""
    out = {}
    out.update(diameter_ratio_summary(ratios,
               loss_thresh=kw.get('loss_thresh', NO_TAPER_RATIO),
               min_pairs=kw.get('min_pairs', MIN_PAIRS)))
    out.update(taper_gradient(points, min_n=kw.get('min_n', MIN_POINTS)))
    return out
