"""Gradiente di pruning vascolare (nucleo puro).

Il pruning e' la perdita dei piccoli vasi periferici (marker QCT di rimodellamento
del piccolo circolo). BV5/TBV globale la riassume; il GRADIENTE la localizza:
la densita' di piccoli vasi in funzione della distanza dalla pleura. In un polmone
sano i piccoli vasi popolano la periferia; nel pruning la densita' periferica cala.

Su gusci di distanza-dalla-pleura si calcola:
  - densita' di piccoli vasi   = ml piccoli vasi / ml polmone   (per guscio)
  - frazione BV5               = ml piccoli vasi / ml vasi totali (per guscio)
poi si aggregano PERIFERIA (dist <= periph_mm) vs CENTRO e si stima la pendenza
della densita' sulla distanza. `pruning_ratio` = densita' periferica / centrale:
< 1 o ridotto = pruning periferico.

CAVEAT: dipende dalla soglia dei vasi, da kernel/dose e dal volume inspiratorio, e
dall'approssimazione del campo di distanza sulla griglia ×3 → confronti solo a
parita' di protocollo.

Puro: aritmetica. Testato in tests/test_vascular_gradient_core.py.
"""


def _linfit(x, y):
    n = len(x)
    if n < 2:
        return None, None
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


def pruning_summary(shells, periph_mm=15.0, min_lung_ml=1.0):
    """shells: lista di dict {d_center (mm), lung_ml, small_ml, all_ml}.
    Ritorna profilo per guscio, densita' periferica/centrale, pruning_ratio,
    frazione BV5 periferica, pendenza densita'-distanza."""
    valid = [s for s in shells if s.get('lung_ml', 0) >= min_lung_ml]
    if not valid:
        return {'pruning_ratio': None, 'density_periph': None,
                'density_central': None, 'bv5_frac_periph': None,
                'gradient_per_mm': None, 'gradient_r2': None, 'profile': []}

    profile = []
    for s in valid:
        lung = s['lung_ml']
        profile.append({
            'd_mm': round(s['d_center'], 1),
            'small_density': round(s['small_ml'] / lung, 4),
            'bv5_frac': round(s['small_ml'] / s['all_ml'], 3) if s.get('all_ml') else None,
        })

    def agg(sel):
        lu = sum(s['lung_ml'] for s in sel)
        sm = sum(s['small_ml'] for s in sel)
        al = sum(s.get('all_ml', 0) for s in sel)
        return (sm / lu if lu else None, sm / al if al else None, lu)

    periph = [s for s in valid if s['d_center'] <= periph_mm]
    central = [s for s in valid if s['d_center'] > periph_mm]
    dp, bp, _ = agg(periph) if periph else (None, None, 0)
    dc, _, _ = agg(central) if central else (None, None, 0)
    ratio = (dp / dc) if (dp is not None and dc) else None

    slope, r2 = _linfit([s['d_center'] for s in valid],
                        [s['small_ml'] / s['lung_ml'] for s in valid])

    return {
        'pruning_ratio': round(ratio, 3) if ratio is not None else None,
        'density_periph': round(dp, 4) if dp is not None else None,
        'density_central': round(dc, 4) if dc is not None else None,
        'bv5_frac_periph': round(bp, 3) if bp is not None else None,
        'gradient_per_mm': round(slope, 5) if slope is not None else None,
        'gradient_r2': round(r2, 3) if r2 is not None else None,
        'periph_mm': periph_mm, 'profile': profile,
    }
