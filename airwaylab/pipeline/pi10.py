"""Pi10 a livello di soggetto dal risultato della misura (compute standalone).

Legge out/tree_measured.json, prende le vie aeree misurabili (`qc == 'ok'` con
calibro e parete), regredisce √WA su Pi e legge Pi10 a 10 mm CON la sua diagnostica
(range dei perimetri, estrapolazione, CI bootstrap, leave-one-out). Scrive
out/pi10.json. Gira nel work dir dopo measure.py.
"""
import json
import os

from pi10_core import MIN_AIRWAYS, airway_points, pi10_summary
from provenance import provenance

TREE = 'out/tree_measured.json'

if not os.path.exists(TREE):
    print('pi10: manca tree_measured.json — salto')
    raise SystemExit(0)

branches = json.load(open(TREE)).get('branches', [])
pts = airway_points(branches)
res = pi10_summary(pts)
res['schema_version'] = 2
res['unit'] = 'mm (sqrt area di parete a Pi=10 mm)'
res['status'] = 'exploratory'

# quante vie aeree entrano nella regressione e quante restano fuori (e perche')
n_tot = len(branches)
n_qc_ok = sum(1 for b in branches if b.get('qc') == 'ok')
res['provenance'] = provenance(
    'pi10_airwaylab',
    params={'target_pi_mm': 10.0, 'min_airways': MIN_AIRWAYS,
            'wall': 'FWHM half-max', 'perimeter': 'approssimazione circolare (Pi=pi*d)',
            'ci': 'bootstrap 95% (seed fisso)', 'loo': 'leave-one-out'},
    denominators={'n_branches_totali': n_tot, 'n_qc_ok': n_qc_ok,
                  'n_punti_regressione': res['n']},
    exclusions={'qc_non_ok': n_tot - n_qc_ok,
                'qc_ok_senza_calibro_o_parete': n_qc_ok - res['n']})
json.dump(res, open('out/pi10.json', 'w'), indent=1)

if res['pi10'] is not None:
    ci = res.get('ci95') or {}
    ci_txt = (f" · CI95 {ci['ci_lo']}–{ci['ci_hi']}"
              if ci.get('ci_lo') is not None else "")
    print(f"Pi10 = {res['pi10']} mm  (√WA a Pi=10; n={res['n']} vie aeree, "
          f"R²={res['r2']}, slope={res['slope']}{ci_txt})")
    print(f"  perimetri osservati Pi {res['pi_min']}–{res['pi_max']} mm; "
          f"{res['n_below_target']}/{res['n']} sotto i 10 mm")
    if res.get('extrapolation'):
        print("  ATTENZIONE: 10 mm e' FUORI dai perimetri osservati -> Pi10 e' "
              "un'ESTRAPOLAZIONE, non una lettura interpolata. Interpreta con cautela.")
    loo = res.get('loo') or {}
    if loo.get('loo_delta_max') is not None:
        print(f"  leave-one-out: spostamento max {loo['loo_delta_max']} mm "
              f"togliendo una via aerea (SD {loo['loo_delta_sd']})")
else:
    print(f"Pi10: non calcolabile — solo {res['n']} vie aeree misurabili "
          f"(servono ≥ {MIN_AIRWAYS}); confronta a parita' di protocollo")
