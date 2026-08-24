"""Tapering a livello di soggetto dal risultato della misura (compute standalone).

Legge out/tree_measured.json e calcola le due letture di rastremazione: rapporto
figlio/genitore del lume e gradiente globale del calibro sulla distanza cumulativa
dalla carena. Scrive out/tapering.json. Gira nel work dir dopo measure.py.
"""
import json
import os

from tapering_core import tapering_summary

TREE = 'out/tree_measured.json'

if not os.path.exists(TREE):
    print('tapering: manca tree_measured.json — salto')
    raise SystemExit(0)

branches = json.load(open(TREE)).get('branches', [])
by_v = {b['v']: b for b in branches}
parent_of = {b['id']: by_v.get(b['u']) for b in branches}   # ramo genitore o None

# lunghezza cumulativa dalla radice fino all'INIZIO di ogni ramo (memoizzata)
_cum = {}


def cum_start(b):
    bid = b['id']
    if bid in _cum:
        return _cum[bid]
    p = parent_of.get(bid)
    _cum[bid] = (cum_start(p) + float(p.get('length') or 0.0)) if p else 0.0
    return _cum[bid]


def ok(b):
    return b.get('qc') == 'ok' and b.get('d_mean') and b['d_mean'] > 0


# rapporti figlio/genitore (entrambi misurabili)
ratios = []
for b in branches:
    p = parent_of.get(b['id'])
    if ok(b) and p is not None and ok(p):
        ratios.append(b['d_mean'] / p['d_mean'])

# punti (distanza cumulativa al punto medio del ramo, calibro) per il gradiente
points = []
for b in branches:
    if ok(b):
        L = cum_start(b) + 0.5 * float(b.get('length') or 0.0)
        points.append((L, b['d_mean']))

res = tapering_summary(ratios, points)
res['schema_version'] = 1
json.dump(res, open('out/tapering.json', 'w'), indent=1)

rm = res.get('taper_ratio_med')
rt = res.get('taper_rate_pct_per_cm')
print('Tapering delle vie aeree:')
if rm is not None:
    print(f"  rapporto figlio/genitore mediano {rm}  "
          f"(senza rastremazione {100 * res['frac_no_taper']:.0f}% delle coppie, n={res['n_pairs']})")
else:
    print(f"  rapporto figlio/genitore: coppie insufficienti (n={res['n_pairs']})")
if rt is not None:
    print(f"  gradiente {rt} %/cm di riduzione del calibro  (R²={res['r2']}, n={res['n']})")
else:
    print(f"  gradiente: rami misurabili insufficienti (n={res['n']})")
