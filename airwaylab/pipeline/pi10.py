"""Pi10 a livello di soggetto dal risultato della misura (compute standalone).

Legge out/tree_measured.json, prende le vie aeree misurabili (`qc == 'ok'` con
calibro e parete), regredisce √WA su Pi e legge Pi10 a 10 mm. Scrive out/pi10.json.
Gira nel work dir dopo measure.py.
"""
import json
import os

from pi10_core import MIN_AIRWAYS, airway_points, pi10_fit
from provenance import provenance

TREE = 'out/tree_measured.json'

if not os.path.exists(TREE):
    print('pi10: manca tree_measured.json — salto')
    raise SystemExit(0)

branches = json.load(open(TREE)).get('branches', [])
pts = airway_points(branches)
res = pi10_fit(pts)
res['schema_version'] = 1
res['unit'] = 'mm (sqrt area di parete a Pi=10 mm)'
res['status'] = 'exploratory'

# quante vie aeree entrano nella regressione e quante restano fuori (e perche')
n_tot = len(branches)
n_qc_ok = sum(1 for b in branches if b.get('qc') == 'ok')
res['provenance'] = provenance(
    'pi10_airwaylab',
    params={'target_pi_mm': 10.0, 'min_airways': MIN_AIRWAYS,
            'wall': 'FWHM half-max', 'perimeter': 'approssimazione circolare (Pi=pi*d)'},
    denominators={'n_branches_totali': n_tot, 'n_qc_ok': n_qc_ok,
                  'n_punti_regressione': res['n']},
    exclusions={'qc_non_ok': n_tot - n_qc_ok,
                'qc_ok_senza_calibro_o_parete': n_qc_ok - res['n']})
json.dump(res, open('out/pi10.json', 'w'), indent=1)

if res['pi10'] is not None:
    print(f"Pi10 = {res['pi10']} mm  (√WA a Pi=10; n={res['n']} vie aeree, "
          f"R²={res['r2']}, slope={res['slope']})")
else:
    print(f"Pi10: non calcolabile — solo {res['n']} vie aeree misurabili "
          f"(servono ≥ {__import__('pi10_core').MIN_AIRWAYS}); confronta a parita' di protocollo")
