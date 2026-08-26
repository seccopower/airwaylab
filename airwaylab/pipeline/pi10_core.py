"""Pi10 — descrittore di parete a livello di soggetto (nucleo puro).

Pi10 (Nakano) = radice quadrata dell'area di parete (√WA) PREDETTA a un perimetro
interno del lume standardizzato di 10 mm. E' un modo di confrontare lo spessore di
parete TRA vie aeree di calibro diverso e TRA soggetti riportandolo tutto allo
stesso perimetro di riferimento: NON e' una "normalizzazione di WA%", e' il valore
letto dalla retta √WA–Pi in un punto fisso (Pi = 10 mm).

Metodo: per ogni via aerea misurabile (`qc == 'ok'`, con calibro e parete) si calcola
il punto (Pi, √WA), dove
  Pi  = perimetro interno del lume  = pi * d_lume            (approssimazione circolare)
  WA  = area della parete           = pi * (r_out^2 - r_in^2),  r_in = d_lume/2,
                                                                r_out = r_in + spessore
Si regredisce **√WA = a + b·Pi** su tutte le vie aeree del soggetto; Pi10 = a + b·10.
E' quindi una metrica a livello di SOGGETTO (un numero per esame), non per ramo.

DIAGNOSTICA (il numero da solo non basta ed e' facile da leggere male):
  - pi_min/pi_max: intervallo di perimetri realmente osservati. Se 10 mm cade FUORI
    da questo intervallo, Pi10 e' un'ESTRAPOLAZIONE, non una lettura interpolata —
    lo dichiariamo esplicitamente (`extrapolation`).
  - copertura: quanti punti stanno sopra/sotto il target (supporto vicino a 10 mm).
  - CI bootstrap: incertezza del valore per ricampionamento (deterministico, seed
    fisso: stessi punti -> stessa CI).
  - leave-one-out: quanto Pi10 si muove togliendo una via aerea alla volta (un solo
    ramo influente puo' spostare tutto).

Limiti: dipende da quali vie aeree sono misurabili (protocollo/inspirazione) e
dall'approssimazione circolare del perimetro; half-max sovrastima la parete sui rami
piccoli. Confrontabile solo a parita' di protocollo.

Puro: solo aritmetica + random (bootstrap seedato, riproducibile). Testato in
tests/test_pi10_core.py.
"""
import math
import random

MIN_AIRWAYS = 10          # sotto questo numero la regressione non e' stabile
BOOT_SEED = 12345         # seed fisso: la CI e' riproducibile dato lo stesso input
BOOT_N = 1000


def wall_point(d_lumen_mm, wall_mm):
    """Punto (Pi, WA) di una via aerea: perimetro interno e area di parete (mm, mm^2).
    Approssimazione circolare del lume."""
    r_in = d_lumen_mm / 2.0
    r_out = r_in + wall_mm
    wa = math.pi * (r_out * r_out - r_in * r_in)
    pi_perim = math.pi * d_lumen_mm
    return pi_perim, wa


def airway_points(branches):
    """Estrae i punti (Pi, √WA) dalle vie aeree misurabili.
    branches: iterabile di dict con 'qc', 'd_mean', 'wall'. Solo `qc == 'ok'` con
    calibro e parete positivi entra nella regressione."""
    pts = []
    for b in branches:
        if b.get('qc') != 'ok':
            continue
        d = b.get('d_mean')
        w = b.get('wall')
        if not d or not w or d <= 0 or w <= 0:
            continue
        pi_perim, wa = wall_point(d, w)
        if wa > 0:
            pts.append((pi_perim, math.sqrt(wa)))
    return pts


def _ols(points):
    """Regressione √WA = a + b·Pi. Ritorna (slope, intercept, r2) o (None, None, None)."""
    n = len(points)
    if n < 2:
        return None, None, None
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return None, None, None
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    ybar = sy / n
    ss_tot = sum((p[1] - ybar) ** 2 for p in points)
    ss_res = sum((p[1] - (a + b * p[0])) ** 2 for p in points)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else None
    return b, a, r2


def _pi10_value(points, target_pi):
    """Solo il valore Pi10 (a + b·target) da un fit; None se non fittabile."""
    b, a, _ = _ols(points)
    if b is None:
        return None
    return a + b * target_pi


def _percentile(sorted_vals, q):
    """Percentile lineare (q in [0,100]) su lista GIA' ordinata e non vuota."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (q / 100.0) * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def pi10_bootstrap(points, target_pi=10.0, n_boot=BOOT_N, seed=BOOT_SEED,
                   min_n=MIN_AIRWAYS):
    """CI 95% di Pi10 per bootstrap (ricampionamento con reinserimento).
    Deterministico: stesso input+seed -> stessa CI. None se punti insufficienti."""
    n = len(points)
    if n < min_n:
        return {'ci_lo': None, 'ci_hi': None, 'n_boot': 0}
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = [points[rng.randrange(n)] for _ in range(n)]
        v = _pi10_value(sample, target_pi)
        if v is not None:
            vals.append(v)
    if not vals:
        return {'ci_lo': None, 'ci_hi': None, 'n_boot': 0}
    vals.sort()
    return {'ci_lo': round(_percentile(vals, 2.5), 3),
            'ci_hi': round(_percentile(vals, 97.5), 3),
            'n_boot': len(vals)}


def pi10_loo(points, target_pi=10.0, min_n=MIN_AIRWAYS):
    """Sensibilita' leave-one-out: di quanto si muove Pi10 togliendo una via aerea.
    Ritorna il massimo scostamento assoluto e la sua SD. Serve n > min_n (dopo il
    taglio devono restare >= min_n punti). None se troppo pochi."""
    n = len(points)
    if n <= min_n:
        return {'loo_delta_max': None, 'loo_delta_sd': None, 'n': n}
    full = _pi10_value(points, target_pi)
    if full is None:
        return {'loo_delta_max': None, 'loo_delta_sd': None, 'n': n}
    deltas = []
    for i in range(n):
        sub = points[:i] + points[i + 1:]
        v = _pi10_value(sub, target_pi)
        if v is not None:
            deltas.append(v - full)
    if not deltas:
        return {'loo_delta_max': None, 'loo_delta_sd': None, 'n': n}
    mean = sum(deltas) / len(deltas)
    sd = math.sqrt(sum((d - mean) ** 2 for d in deltas) / len(deltas))
    return {'loo_delta_max': round(max(deltas, key=abs), 3),
            'loo_delta_sd': round(sd, 4), 'n': n}


def pi10_fit(points, target_pi=10.0, min_n=MIN_AIRWAYS):
    """Regressione √WA = a + b·Pi, lettura a Pi = target_pi, PIU' la diagnostica di
    base (range dei perimetri, estrapolazione, copertura). Se i punti sono meno di
    `min_n`, i valori di fit sono None (n e il range restano informativi)."""
    n = len(points)
    pis = [p[0] for p in points]
    pi_min = round(min(pis), 3) if pis else None
    pi_max = round(max(pis), 3) if pis else None
    # 10 mm dentro l'intervallo osservato? altrimenti e' estrapolazione
    if pi_min is None:
        extrapolation = None
    else:
        extrapolation = bool(target_pi < pi_min or target_pi > pi_max)
    n_below = sum(1 for x in pis if x < target_pi)
    frac_below = round(n_below / n, 3) if n else None

    base = {'pi10': None, 'slope': None, 'intercept': None, 'r2': None,
            'n': n, 'target_pi': target_pi,
            'pi_min': pi_min, 'pi_max': pi_max,
            'extrapolation': extrapolation,
            'frac_below_target': frac_below, 'n_below_target': n_below}
    if n < min_n:
        return base
    b, a, r2 = _ols(points)
    if b is None:
        return base
    base.update({'pi10': round(a + b * target_pi, 3), 'slope': round(b, 4),
                 'intercept': round(a, 3),
                 'r2': round(r2, 3) if r2 is not None else None})
    return base


def pi10_summary(points, target_pi=10.0, min_n=MIN_AIRWAYS,
                 n_boot=BOOT_N, seed=BOOT_SEED):
    """Fit + diagnostica completa (CI bootstrap + leave-one-out) in un dict unico,
    per il runner. I sotto-blocchi sono None-safe se i punti sono insufficienti."""
    res = pi10_fit(points, target_pi=target_pi, min_n=min_n)
    res['ci95'] = pi10_bootstrap(points, target_pi=target_pi, n_boot=n_boot,
                                 seed=seed, min_n=min_n)
    res['loo'] = pi10_loo(points, target_pi=target_pi, min_n=min_n)
    return res
