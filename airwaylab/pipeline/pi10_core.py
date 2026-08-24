"""Pi10 — metrica standard di rimodellamento parietale (nucleo puro).

Pi10 (Nakano) = radice quadrata dell'area di parete (√WA) predetta a un perimetro
interno del lume di 10 mm. E' lo standard di campo per confrontare la parete
bronchiale TRA vie aeree di calibro diverso e TRA soggetti: WA% da solo dipende dal
calibro, Pi10 lo normalizza.

Metodo: per ogni via aerea misurabile (`qc == 'ok'`, con calibro e parete) si calcola
il punto (Pi, √WA), dove
  Pi  = perimetro interno del lume  = pi * d_lume            (approssimazione circolare)
  WA  = area della parete           = pi * (r_out^2 - r_in^2),  r_in = d_lume/2,
                                                                r_out = r_in + spessore
Si regredisce **√WA = a + b·Pi** su tutte le vie aeree del soggetto; Pi10 = a + b·10.
E' quindi una metrica a livello di SOGGETTO (un numero per esame), non per ramo.

Limiti: dipende da quali vie aeree sono misurabili (protocollo/inspirazione) e
dall'approssimazione circolare del perimetro; half-max sovrastima la parete sui rami
piccoli. Confrontabile solo a parita' di protocollo.

Puro: solo aritmetica. Testato in tests/test_pi10_core.py.
"""
import math

MIN_AIRWAYS = 10          # sotto questo numero la regressione non e' stabile


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


def pi10_fit(points, target_pi=10.0, min_n=MIN_AIRWAYS):
    """Regressione lineare √WA = a + b·Pi e lettura a Pi = target_pi.

    Ritorna {pi10, slope, intercept, r2, n, target_pi}. Se i punti sono meno di
    `min_n`, pi10/slope/intercept/r2 sono None (n riportato comunque)."""
    n = len(points)
    if n < min_n:
        return {'pi10': None, 'slope': None, 'intercept': None,
                'r2': None, 'n': n, 'target_pi': target_pi}
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return {'pi10': None, 'slope': None, 'intercept': None,
                'r2': None, 'n': n, 'target_pi': target_pi}
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    ybar = sy / n
    ss_tot = sum((p[1] - ybar) ** 2 for p in points)
    ss_res = sum((p[1] - (a + b * p[0])) ** 2 for p in points)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else None
    pi10 = a + b * target_pi
    return {'pi10': round(pi10, 3), 'slope': round(b, 4),
            'intercept': round(a, 3), 'r2': round(r2, 3) if r2 is not None else None,
            'n': n, 'target_pi': target_pi}
